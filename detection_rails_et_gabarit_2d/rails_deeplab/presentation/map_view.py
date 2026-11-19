"""Offline map presentation, separate from calibrated positioning.

Africa uses the checked AfTS route. Belgium overlays accepted GPS observations
on official Infrabel geometry; it does not guess the exact track or route.
Raw GPS is never overwritten and no trace is sent to a map service.
"""
from pathlib import Path
from bisect import bisect_right
from datetime import datetime
import csv
import json
import math
import sys
import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer, Geod
from shapely.geometry import shape, Point, LineString
from shapely.ops import transform
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'localisation'))
from render_connected_video import along_path, font
from rail_localisation import RailNetwork

GEOD = Geod(ellps='WGS84')


def gps_quality(observations, nearest_distance_m=None):
    """Conservative display exclusions, not a GNSS accuracy estimate.

    Use the REAL UTC clock for accelerated video. Reject an isolated spike
    when both adjoining speeds exceed 100 m/s and bypassing it is plausible.
    Remaining impossible transitions break the displayed trail.
    """
    t = np.array([datetime.fromisoformat(o['utc_time'].replace('Z', '+00:00')).timestamp() for o in observations])
    lon = np.array([float(o['longitude_deg']) for o in observations])
    lat = np.array([float(o['latitude_deg']) for o in observations])
    if not np.isfinite(t).all() or not np.all(np.diff(t)>0): raise ValueError('Increasing real UTC times required')
    distance = np.array(GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])[2])
    dt = np.diff(t)
    speed = distance/dt
    status = np.array(['accepted_for_map_display']*len(t), dtype=object)
    for i in range(1, len(t)-1):
        bypass = GEOD.inv(lon[i-1], lat[i-1], lon[i+1], lat[i+1])[2]/(t[i+1]-t[i-1])
        if speed[i-1]>100 and speed[i]>100 and bypass<=100:
            status[i] = 'withheld_isolated_gps_jump'
    if nearest_distance_m is not None:
        for i, d in enumerate(nearest_distance_m):
            if d>200 and status[i]=='accepted_for_map_display': status[i]='withheld_far_from_reference_network'
    valid = status=='accepted_for_map_display'
    connect = np.zeros(len(t), dtype=bool)
    connect[1:] = valid[:-1] & valid[1:] & (speed<=100) & (dt<=30)
    # A point adjacent to an impossible transition may belong to a genuinely
    # different segment; do not invent a line through the gap.
    return status.tolist(), connect.tolist(), distance, dt


class MapView:
    def __init__(self, mode, localisation, network_path, width, height, proof_dir, clip_metadata=None):
        self.mode, self.width, self.height = mode, width, height
        self.context, self.parts, self.xy, self.times = [], [], [], []
        self.by_interval = {}
        self.statuses = []
        self.total = None
        self.gauge = None
        self.attribution = ''
        self.subtitle = 'Aucune position GPS fournie'
        self.line = ''
        self.audit = {}
        if mode == 'none':
            self.background = Image.new('RGB', (width, height), '#18222a')
            return
        if mode == 'metadata':
            if not clip_metadata:raise ValueError('Clip metadata position required')
            lon,lat=float(clip_metadata['longitude_deg']),float(clip_metadata['latitude_deg'])
            if (lon,lat)==(0,0):raise ValueError('Placeholder location cannot be plotted')
            obs=[dict(longitude_deg=lon,latitude_deg=lat)]
            network=RailNetwork.from_geopackage(network_path,obs,margin_m=3000)
            self.project=network.forward.transform
            self.xy=[self.project(lon,lat)];self.times=[]
            self.cx,self.cy=self.xy[0];self.scale=min(width-60,height-60)/4500
            self.background=Image.new('RGB',(width,height),'#18222a');d=ImageDraw.Draw(self.background)
            for edge in network.edges.values():
                pts=[self.pixel(p) for p in edge.xy.coords]
                d.line(pts,fill='#65867e',width=3)
            x,y=self.pixel(self.xy[0]);d.ellipse((x-9,y-9,x+9,y+9),outline='#dbe9e3',width=3)
            d.text((18,height-67),f'{lat:.4f}° N  ·  {lon:.4f}° E',font=font(18),fill='#dbe9e3')
            d.text((18,height-35),'Repère fixe, pas de trajet déduit',font=font(16),fill='#9fadb1')
            d.text((width-52,14),'N ↑',font=font(17),fill='#9cabb2')
            self.subtitle='Position du fichier · instant inconnu'
            self.attribution='AfTS-Db v3 · CC BY 4.0 · © OSM'
            self.audit=dict(mode='single_clip_metadata_position',coordinate_time_known=False,per_frame_gps_known=False,network_route_inferred=False)
            return
        localisation = Path(localisation)
        self.observations = list(csv.DictReader((localisation/'GPS_OBSERVATIONS.csv').open()))
        if len(self.observations)<2: raise ValueError('Too few GPS observations for map')
        self.times = [float(o['video_time_s']) for o in self.observations]
        if mode == 'africa':
            route = json.loads((localisation/'RESULTATS_LOCALISATION.json').read_text())
            if route['route_status']!='mapped_route_estimate_with_observation_checks':
                raise ValueError('Unresolved Africa route; display GPS only or explicitly no route')
            self.observations = route['matches']
            self.times = [float(o['video_time_s']) for o in self.observations]
            self.project = Transformer.from_crs(4326, route['projection_epsg'], always_xy=True).transform
            self.xy = [self.project(o['snapped_longitude_deg'], o['snapped_latitude_deg']) for o in self.observations]
            geometry = json.loads((localisation/'TRAJET_CARTOGRAPHIQUE.geojson').read_text())
            for feature in geometry['features']:
                if feature['properties'].get('kind')!='route_piece': continue
                pts = [self.project(*c) for c in feature['geometry']['coordinates']]
                self.parts.append(pts)
                self.by_interval.setdefault(int(feature['properties']['interval']), []).extend(pts)
            net = RailNetwork.from_geopackage(network_path, self.observations, margin_m=3000)
            self.context = [list(transform(self.project, net.to_wgs84(e.xy)).coords) for e in net.edges.values()]
            self.total = route['route_distance_estimate_m']
            self.gauge = route.get('gauge_selected_nominal_m')
            self.line = ' / '.join(sorted({e['line'] for e in route.get('route_edges', []) if e.get('line')}))
            self.subtitle = 'Position raccordée au réseau · estimée'
            self.attribution = 'AfTS-Db v3 · CC BY 4.0 · © OSM'
            self.audit = dict(mode='checked_route_interpolation', max_interpolation_gap_video_s=5, distance_estimate_m=self.total, no_new_calibration=True)
        elif mode == 'belgium':
            self.project = Transformer.from_crs(4326, 31370, always_xy=True).transform
            raw = json.loads(Path(network_path).read_text())
            lines, props = [], []
            for feature in raw['features']:
                geo = shape(feature['geometry'])
                geoms = [geo] if geo.geom_type=='LineString' else list(geo.geoms) if geo.geom_type=='MultiLineString' else []
                for line in geoms:
                    lines.append(transform(self.project, line)); props.append(feature['properties'])
            tree = STRtree(lines)
            self.xy = [self.project(float(o['longitude_deg']), float(o['latitude_deg'])) for o in self.observations]
            points = [Point(p) for p in self.xy]
            nearest = tree.nearest(points)
            distances = [points[i].distance(lines[int(j)]) for i,j in enumerate(nearest)]
            self.statuses, connected, _, _ = gps_quality(self.observations, distances)
            segment = []
            for i,p in enumerate(self.xy):
                if not connected[i] and segment:
                    if len(segment)>1: self.parts.append(segment)
                    segment=[]
                if self.statuses[i]=='accepted_for_map_display': segment.append(p)
            if len(segment)>1:self.parts.append(segment)
            self.context=[list(line.coords) for line in lines]
            self.subtitle='Trajet GPS sur le réseau · voie non certifiée'
            self.attribution='Réseau Infrabel · geosporen · CC0'
            audit=[]
            for i, o in enumerate(self.observations):
                audit.append(dict(frame_index=o['frame_index'], video_time_s=o['video_time_s'], utc_time=o['utc_time'], latitude_deg=o['latitude_deg'], longitude_deg=o['longitude_deg'], display_status=self.statuses[i], distance_to_network_m=distances[i], nearest_line_candidate=props[int(nearest[i])].get('linecalfa',''), trail_connected_to_previous=connected[i]))
            proof_dir=Path(proof_dir);proof_dir.mkdir(exist_ok=True,parents=True)
            with (proof_dir/'AUDIT_GPS_AFFICHAGE.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(audit[0]));writer.writeheader();writer.writerows(audit)
            self.audit=dict(mode='GPS_trace_on_official_network_not_map_matched_route', total_observations=len(audit), withheld=sum(s!='accepted_for_map_display' for s in self.statuses), max_speed_m_s=100, max_gap_real_s=30, max_distance_to_network_m=200, raw_positions_preserved=True, no_exact_track_assignment=True, total_distance_not_claimed=True)
        else:raise ValueError('Unsupported map mode')
        all_points=[p for part in self.parts for p in part]
        if not all_points:raise ValueError('No accepted map path')
        xmin,xmax=min(p[0] for p in all_points),max(p[0] for p in all_points)
        ymin,ymax=min(p[1] for p in all_points),max(p[1] for p in all_points)
        self.cx,self.cy=(xmin+xmax)/2,(ymin+ymax)/2
        self.scale=min((width-80)/max(100,xmax-xmin),(height-90)/max(100,ymax-ymin))*.90
        self.background=Image.new('RGB',(width,height),'#18222a')
        d=ImageDraw.Draw(self.background)
        for x in range(0,width,60):d.line([(x,0),(x,height)],fill='#202d35')
        for y in range(0,height,60):d.line([(0,y),(width,y)],fill='#202d35')
        for line in self.context:
            pts=[self.pixel(p) for p in line]
            if any(-100<x<width+100 and -100<y<height+100 for x,y in pts):d.line(pts,fill='#52626c',width=2)
        for line in self.parts:d.line([self.pixel(p) for p in line],fill='#58877d',width=5)
        start,end=self.pixel(all_points[0]),self.pixel(all_points[-1])
        for point,label in [(start,'Départ'),(end,'Fin')]:
            x,y=point;d.ellipse((x-4,y-4,x+4,y+4),fill='#ccd8dc')
            d.text((max(6,min(width-80,x+9)),max(8,min(height-30,y-28))),label,font=font(17),fill='#cfdbde')
        d.text((width-52,14),'N ↑',font=font(17),fill='#9cabb2')

    def pixel(self,p):return(self.width/2+(p[0]-self.cx)*self.scale,self.height/2-(p[1]-self.cy)*self.scale)

    def position(self,t):
        if self.mode=='none':return None,'gps_not_available'
        if self.mode=='metadata':return None,'clip_metadata_only_not_frame_observation'
        if not self.times[0]-1e-6<=t<=self.times[-1]+1e-6:return None,'outside_gps_coverage'
        i=max(0,min(len(self.times)-1,bisect_right(self.times,t+1e-6)-1))
        if self.mode=='belgium':
            if abs(self.times[i]-t)>2e-6:return None,'no_observation_for_frame'
            if self.statuses[i]!='accepted_for_map_display':return None,self.statuses[i]
            return self.xy[i],self.statuses[i]
        if abs(self.times[i]-t)<=1e-6:return self.xy[i],'mapped_at_gps_observation'
        if i>=len(self.times)-1 or self.times[i+1]-self.times[i]>5:return None,'withheld_gap'
        points=self.by_interval.get(i)
        if not points:
            # Repeated/quantized GPS coordinates create a zero-length interval,
            # hence no route_piece. Their constant interpolation is still defined.
            # This does not establish zero physical train speed.
            if math.dist(self.xy[i],self.xy[i+1])<=1e-6:
                return self.xy[i],'constant_between_equal_gps_positions'
            return None,'route_interval_unavailable'
        return along_path(points,(t-self.times[i])/(self.times[i+1]-self.times[i])),'interpolated_on_route'

    def image(self,t):
        im=self.background.copy();d=ImageDraw.Draw(im)
        if self.mode=='none':
            d.text((28,65),'Trajet indisponible',font=font(24,True),fill='#dce5e6')
            d.text((28,108),'Ce clip ne fournit pas de GPS.',font=font(18),fill='#98aab1')
            d.text((28,141),'Rails et objets restent analysés.',font=font(18),fill='#98aab1')
            return im,'gps_not_available'
        pos,status=self.position(t)
        if pos:
            x,y=self.pixel(pos);d.ellipse((x-10,y-10,x+10,y+10),fill='#dbe9e3',outline='#18222a',width=3)
        return im,status
