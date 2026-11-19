"""Render the existing 2D video and mapped train progress on one native timeline.

No new object/rail prediction is made. The video pane is a scaled view; all CSV
coordinates remain in the original image. Map interpolation is explicitly an
estimate and is never exported as an observed GPS or an object's location.
"""
from pathlib import Path
from bisect import bisect_right
from fractions import Fraction
from functools import lru_cache
import csv
import json
import math
import time
import av
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from handoff_bundle import read_csv, sha256, write_json


@lru_cache(maxsize=24)
def font(size, bold=False):
    candidates = [
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf' if bold else '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for name in candidates:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def along_path(points, fraction):
    """Interpolate within a projected polyline without cutting across its bends."""
    if not points:
        return None
    if len(points) == 1:
        return points[0]
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    remaining = min(1.0, max(0.0, fraction)) * sum(lengths)
    for a, b, length in zip(points, points[1:], lengths):
        if remaining <= length and length > 0:
            f = remaining / length
            return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
        remaining -= length
    return points[-1]


def pane_transform(width, height):
    """Letterbox instead of stretching a VGA/4:3 source into the 16:9 pane."""
    scale = min(1440 / width, 810 / height)
    w, h = round(width * scale), round(height * scale)
    return w, h, (1440-w)//2, 70+(810-h)//2


class RoutePanel:
    def __init__(self, localisation):
        self.route = json.loads((localisation / 'RESULTATS_LOCALISATION.json').read_text())
        if self.route['route_status'] != 'mapped_route_estimate_with_observation_checks':
            raise ValueError('A resolved mapped route is required for a map video')
        self.transform = Transformer.from_crs(4326, self.route['projection_epsg'], always_xy=True).transform
        self.observations = self.route['matches']
        self.times = [float(x['video_time_s']) for x in self.observations]
        geometry = json.loads((localisation / 'TRAJET_CARTOGRAPHIQUE.geojson').read_text())
        self.pieces, self.by_interval = [], {}
        for feature in geometry['features']:
            if feature['properties'].get('kind') != 'route_piece':
                continue
            pts = [self.transform(*c) for c in feature['geometry']['coordinates']]
            self.pieces.append(pts)
            self.by_interval.setdefault(int(feature['properties']['interval']), []).extend(pts)
        all_points = [p for part in self.pieces for p in part]
        if not all_points:
            raise ValueError('Empty mapped route')
        xmin, xmax = min(p[0] for p in all_points), max(p[0] for p in all_points)
        ymin, ymax = min(p[1] for p in all_points), max(p[1] for p in all_points)
        self.scale = min(350 / max(xmax - xmin, 1), 380 / max(ymax - ymin, 1))
        self.cx, self.cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        self.background = Image.new('RGB', (480, 890), '#101c2d')
        d = ImageDraw.Draw(self.background)
        d.text((30, 14), 'PROGRESSION SUR LA VOIE', font=font(23, True), fill='#e8f1fb')
        d.text((30, 52), 'Position raccordée à la carte', font=font(19), fill='#9db6cf')
        for part in self.pieces:
            d.line([self.pixel(p) for p in part], fill='#496984', width=9)
        for j, label in [(0, 'Départ'), (-1, 'Arrivée')]:
            obs = self.observations[j]
            pt = self.pixel(self.transform(obs['snapped_longitude_deg'], obs['snapped_latitude_deg']))
            d.ellipse((pt[0]-5, pt[1]-5, pt[0]+5, pt[1]+5), fill='#d5e6f5')
            d.text((min(370, max(12, pt[0]+9)), pt[1]-38), label, font=font(17), fill='#d5e6f5')
        d.text((405, 220), 'N ↑', font=font(20, True), fill='#9db6cf')
        # This map displays only the available rail geometry, without an invented basemap.
        name = self.route['route_edges'][0].get('line', 'Ligne cartographiée')
        d.text((30, 670), name[:38], font=font(19), fill='#e8f1fb')
        gauge = self.route.get('gauge_selected_nominal_m')
        d.text((30, 706), f'Écartement cartographié : {gauge:.3f} m' if gauge else 'Écartement non résolu', font=font(19), fill='#9db6cf')
        d.text((30, 744), 'Position interpolée entre les GPS.', font=font(18), fill='#f5c575')
        d.text((30, 776), 'Pas une mesure de distance à un objet.', font=font(17), fill='#f5c575')
        d.text((30, 831), 'AfTS-Db v3 · CC BY 4.0', font=font(16), fill='#8198af')
        d.text((30, 855), 'Young / Colombo et al. / © OSM', font=font(15), fill='#8198af')

    def pixel(self, point):
        return (240 + (point[0]-self.cx)*self.scale, 440 - (point[1]-self.cy)*self.scale)

    def position(self, t):
        if not self.times[0] <= t <= self.times[-1]:
            return None
        i = min(len(self.times)-2, max(0, bisect_right(self.times, t)-1))
        a, b = self.observations[i], self.observations[i+1]
        points = self.by_interval.get(i, [])
        if not points:
            return self.transform(a['snapped_longitude_deg'], a['snapped_latitude_deg'])
        return along_path(points, (t-self.times[i])/(self.times[i+1]-self.times[i]))

    def image(self, row):
        im = self.background.copy()
        d = ImageDraw.Draw(im)
        distance = row.get('route_cumulative_estimate_m', '')
        if distance != '':
            value = float(distance)
            total = self.route['route_distance_estimate_m']
            d.text((30, 103), f'{value:,.0f} m'.replace(',', ' '), font=font(48, True), fill='#64dbc3')
            d.text((30, 164), f'sur {total:,.0f} m estimés'.replace(',', ' '), font=font(23), fill='#e8f1fb')
            pos = self.position(float(row['video_time_s']))
            if pos:
                x, y = self.pixel(pos)
                d.ellipse((x-14, y-14, x+14, y+14), fill='#64dbc3', outline='white', width=3)
        else:
            d.text((30, 113), 'Position non résolue', font=font(25), fill='#f5c575')
        return im


def render(localisation, gabarit, destination, preview_indices=(0, 1770, 3540, 5077, 5307)):
    localisation, gabarit, destination = Path(localisation), Path(gabarit), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    video = gabarit / 'VIDEO_GABARIT_2D.mp4'
    rows = read_csv(localisation / 'GABARIT/RAILS_GAUCHE_DROIT_GPS.csv')
    object_frames = {int(r['frame_index']): r for r in read_csv(gabarit / 'FRAMES.csv')}
    panel = RoutePanel(localisation)
    started, count = time.perf_counter(), 0
    temporary = destination.with_name(destination.stem + '.partial.mp4')
    if temporary.exists():
        raise FileExistsError(temporary)
    mapping = []
    with av.open(str(video)) as source, av.open(str(temporary), 'w', options={'movflags': '+faststart'}) as out:
        stream = out.add_stream('libx264', rate=30)
        stream.width, stream.height, stream.pix_fmt = 1920, 960, 'yuv420p'
        stream.time_base = Fraction(1, 1000000)
        stream.codec_context.time_base = Fraction(1, 1000000)
        stream.codec_context.thread_count = 4
        stream.options = {'crf': '23', 'preset': 'veryfast'}
        for i, frame in enumerate(source.decode(video=0)):
            if i >= len(rows):
                raise ValueError('Too many rendered source frames')
            row = rows[i]
            t = float(row['video_time_s'])
            if int(row['frame_index']) != i or abs(float(frame.pts * frame.time_base) - t) > 2e-6:
                raise ValueError('Rendered 2D video is not aligned to native rail timestamps')
            if (frame.width, frame.height) != (int(row['image_width_px']), int(row['image_height_px'])):
                raise ValueError('Unexpected 2D video dimensions')
            canvas = Image.new('RGB', (1920, 960), '#0a1320')
            d = ImageDraw.Draw(canvas)
            d.text((25, 17), f"{row['video_id']}  ·  Rails, objets dans le couloir et localisation", font=font(29, True), fill='#eff6ff')
            pane_w, pane_h, pane_x, pane_y = pane_transform(frame.width, frame.height)
            canvas.paste(frame.to_image().resize((pane_w, pane_h), Image.Resampling.BILINEAR), (pane_x, pane_y))
            canvas.paste(panel.image(row), (1440, 70))
            obj = object_frames[i]
            n, q = int(obj['overlap_candidates']), int(obj.get('quarantined_candidates') or 0)
            d.text((25, 897), f"Temps source : {t:.3f} s  |  Image : {i}  |  Observations retenues : {n}  |  À vérifier : {q}", font=font(24), fill='#eff6ff')
            d.text((25, 933), 'Recouvrement dans l’image : danger physique non confirmé. CSV dans les pixels d’origine.', font=font(19), fill='#b8c9dc')
            if i in preview_indices:
                canvas.save(destination.parent / f'APERCU_{i:06d}.jpg', quality=90)
            encoded = av.VideoFrame.from_image(canvas)
            encoded.pts = round(t * 1000000)
            encoded.time_base = Fraction(1, 1000000)
            for packet in stream.encode(encoded):
                out.mux(packet)
            mapping.append({'frame_index': i, 'source_time_s': t, 'presentation_pts_us': encoded.pts,
                            'pane_x': pane_x, 'pane_y': pane_y, 'scale_x': pane_w / frame.width, 'scale_y': pane_h / frame.height})
            count += 1
            if count % 600 == 0:
                print(f'Vidéo connectée : {count} / {len(rows)} images', flush=True)
        for packet in stream.encode():
            out.mux(packet)
    if count != len(rows):
        raise ValueError('Incomplete connected video')
    checked, max_error = 0, 0.0
    with av.open(str(temporary)) as inp:
        for i, frame in enumerate(inp.decode(video=0)):
            if i >= len(rows):
                raise ValueError('Output contains extra frames')
            error = abs(float(frame.pts * frame.time_base) - float(rows[i]['video_time_s']))
            if error > 2e-6:
                raise ValueError('Output timestamps changed')
            max_error = max(max_error, error)
            checked += 1
    if checked != len(rows):
        raise ValueError('Output frame count mismatch')
    temporary.replace(destination)
    with (destination.parent / 'CORRESPONDANCE_VIDEO.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    report = {'status': 'passed', 'frames_decoded_and_checked': checked, 'max_native_time_error_s': max_error,
              'source_gabarit_video_sha256': sha256(video), 'output_video_sha256': sha256(destination),
              'runtime_seconds': time.perf_counter()-started, 'no_new_inference': True,
              'coordinate_notice': 'CSV coordinates are original pixels. Use CORRESPONDANCE_VIDEO.csv: x_out=pane_x+scale_x*x; y_out=pane_y+scale_y*y. Aspect ratio preserved.',
              'map_position_meaning': 'Along-route interpolation, not observed GPS and not object geolocation'}
    write_json(destination.parent / 'CONTROLE_VIDEO_CONNECTEE.json', report)
    return report
