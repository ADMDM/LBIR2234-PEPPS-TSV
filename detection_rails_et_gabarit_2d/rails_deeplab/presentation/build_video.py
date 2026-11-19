"""Clean rail/2D video and a minimal collaborator ZIP (MP4, CSV, README only).

All joins use source SHA-256, video ID, native frame index and timestamp.
Scientific audit files and TXT caches stay outside the collaborator package.
The original predictions are preserved. GPS and gauge proxies are conditional.
"""
from pathlib import Path
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'localisation'))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]/'gabarit_2d'))
import av
import cv2
import numpy as np
from PIL import Image, ImageDraw
from handoff_bundle import sha256, read_csv, write_json, check_frame, timeline, validate_geometry
from build_handoff import build_handoff
from render_connected_video import font
from railtsv.adaptive_smoothing import AdaptiveProfileSmoother
from lire_contours import load_frame, object_masks
from small_object_refinement import train_guard
from rail_io import FrameGroups
from corridor_geometry import corridor_from_rows
from map_view import MapView

BG='#10191f'; CARD='#18222a'; TEXT='#e7edeb'; MUTED='#9fadb1'; TEAL='#75a99e'; ORANGE='#d69b61'
FR={'dechet':'Déchet','person':'Personne','car':'Voiture','truck':'Camion','motorcycle':'Moto','train':'Train','bicycle':'Vélo','cow':'Bovin','dog':'Chien','bird':'Oiseau'}


def csv_write(path, rows, fields=None):
    with Path(path).open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]),extrasaction='raise')
        writer.writeheader();writer.writerows(rows)


def pairs(row):
    left,right=json.loads(row['left_segments_json']),json.loads(row['right_segments_json'])
    if len(left)!=len(right):raise ValueError('Different numbers of rail segments')
    for a,b in zip(left,right):
        if len(a)!=len(b):raise ValueError('Unpaired points')
        for (xl,y),(xr,yr) in zip(a,b):
            if y!=yr or not 0<=xl<xr<int(row['image_width_px']):raise ValueError('Invalid rail section')
    return left,right


def smooth_existing(rows, parameters):
    """Optional display smoothing; raw coordinates stay in separate CSV columns."""
    smoother=AdaptiveProfileSmoother(int(rows[0]['image_width_px']), **parameters)
    states=Counter(); changes=[]
    for row in rows:
        left,right=pairs(row)
        row['left_raw_segments_json']=row['left_segments_json'];row['right_raw_segments_json']=row['right_segments_json']
        profile={'segments':[[dict(y=y,left_inner_x=xl,right_inner_x=xr,center_x=(xl+xr)/2) for (xl,y),(xr,_) in zip(ls,rs)] for ls,rs in zip(left,right)]}
        result=smoother.update(profile,float(row['video_time_s']))
        states.update(result['smoothing']['states'])
        newleft=[];newright=[]
        for seg in result['segments']:
            newleft.append([[p['left_inner_x'],p['y']] for p in seg]);newright.append([[p['right_inner_x'],p['y']] for p in seg])
            changes.extend(abs(p['center_x']-p['raw_center_x']) for p in seg)
        row['left_segments_json']=json.dumps(newleft,separators=(',',':'));row['right_segments_json']=json.dumps(newright,separators=(',',':'))
        row['geometry_variant']='causal_display_smoothing_raw_preserved'
    return dict(parameters=parameters,states=dict(states),mean_absolute_display_shift_px=float(np.mean(changes)) if changes else None,scope='Display shift, not ground-truth error; no missing-frame extrapolation')


def geometry_transform(width,height,portrait=False,crop=None):
    crop=crop or (0,0,width,height)
    cx,cy,cw,ch=crop
    area=(32,116,1000,808) if portrait else (32,132,1408,792)
    x,y,w,h=area;scale=min(w/cw,h/ch)
    dw,dh=round(cw*scale),round(ch*scale)
    px,py=x+(w-dw)//2,y+(h-dh)//2
    return dict(crop_x=cx,crop_y=cy,crop_width=cw,crop_height=ch,pane_x=px,pane_y=py,scale_x=dw/cw,scale_y=dh/ch,display_width=dw,display_height=dh)


def detect_black_bars(source):
    """Display-only trim, conservative union over sampled frames; never changes CSV pixels."""
    bounds=[]
    with av.open(str(source)) as v:
        for i,frame in enumerate(v.decode(video=0)):
            if i%600:continue
            a=frame.to_ndarray(format='rgb24');active=np.flatnonzero((a.max(2)>30).mean(0)>.04)
            if len(active):bounds.append((max(0,int(active.min())-4),min(frame.width,int(active.max())+5)))
    if not bounds:return None
    lo,hi=min(a for a,b in bounds),max(b for a,b in bounds)
    return (lo,0,hi-lo,frame.height) if hi-lo<frame.width*.7 else None


def compact_rail_row(row, transform, object_count=None, review_count=0, map_status=''):
    left,right=pairs(row)
    axis=[[[ (a[0]+b[0])/2,a[1]] for a,b in zip(ls,rs)] for ls,rs in zip(left,right)]
    speed=row.get('gps_speed_provided_m_s') or row.get('train_speed_m_s','')
    lat=row.get('gps_latitude_deg') or row.get('latitude_deg','')
    lon=row.get('gps_longitude_deg') or row.get('longitude_deg','')
    return dict(video_id=row['video_id'],frame_index=row['frame_index'],video_time_s=row['video_time_s'],source_pts_s=row.get('source_pts_s',row['video_time_s']),image_width_px=row['image_width_px'],image_height_px=row['image_height_px'],
        left_segments_json=row['left_segments_json'],right_segments_json=row['right_segments_json'],axis_segments_json=json.dumps(axis,separators=(',',':')),
        left_raw_segments_json=row.get('left_raw_segments_json',''),right_raw_segments_json=row.get('right_raw_segments_json',''),
        rails_status=row.get('rails_status',''),geometry_variant=row.get('geometry_variant',''),temporal_status=row.get('temporal_status',''),
        train_speed_m_s=speed,train_speed_km_h=float(speed)*3.6 if speed!='' else '',speed_status=row.get('gps_speed_status') or row.get('speed_status','not_available'),utc_time=row.get('utc_time',''),real_elapsed_s=row.get('real_elapsed_s',''),
        latitude_deg=lat,longitude_deg=lon,gps_status=row.get('gps_observation_status','not_available'),map_display_status=map_status,
        route_progress_estimate_m=row.get('route_cumulative_estimate_m',''),route_status=row.get('route_distance_status',''),
        nominal_gauge_m=row.get('mapped_gauge_nominal_m',''),design_gauge_m=row.get('gauge_design_default_m',''),
        left_actual_error_px=row.get('left_actual_error_px',''),right_actual_error_px=row.get('right_actual_error_px',''),error_status=row.get('error_status','no_per_frame_ground_truth'),
        metric_status='conditional_transverse_proxy_not_camera_calibration',object_processing_status='analysed' if object_count is not None else 'not_analysed',objects_in_corridor=object_count if object_count is not None else '',objects_to_review=review_count if object_count is not None else '',
        display_crop_x_px=transform['crop_x'],display_crop_y_px=transform['crop_y'],display_pane_x_px=transform['pane_x'],display_pane_y_px=transform['pane_y'],display_scale_x=transform['scale_x'],display_scale_y=transform['scale_y'],source_video_sha256=row['source_video_sha256'])


def paint_frame(frame,row,objects,reviews,mapview,layout,title,region,total_time):
    native=frame.to_image(); tr=layout['transform']
    x,y,cw,ch=tr['crop_x'],tr['crop_y'],tr['crop_width'],tr['crop_height']
    pane=native.crop((x,y,x+cw,y+ch)).resize((tr['display_width'],tr['display_height']),Image.Resampling.BILINEAR).convert('RGBA')
    def pt(p):return((p[0]-x)*tr['scale_x'],(p[1]-y)*tr['scale_y'])
    left,right=pairs(row)
    layer=Image.new('RGBA',pane.size);ld=ImageDraw.Draw(layer)
    for ls,rs in zip(left,right):
        if len(ls)>1:ld.polygon([pt(p) for p in ls]+[pt(p) for p in rs[::-1]],fill=(95,151,138,24))
    pane=Image.alpha_composite(pane,layer);pd=ImageDraw.Draw(pane)
    for ls,rs in zip(left,right):
        if len(ls)>1:
            pd.line([pt(p) for p in ls],fill='#549c91',width=2)
            pd.line([pt(p) for p in rs],fill='#549c91',width=2)
            pd.line([pt(((a[0]+b[0])/2,a[1])) for a,b in zip(ls,rs)],fill='#080a0b',width=2)
    for oi,obj in enumerate(objects or []):
        for c in obj['full_predicted_mask_contours']:
            points=[pt(p) for p in c['points']]
            if len(points)>1:pd.line(points+[points[0]],fill=ORANGE,width=2)
        if oi<6:
            bx,by=pt(obj['mask_bbox'][:2]);text=FR.get(obj['class_name'],obj['class_name'])
            tx=min(max(4,bx),max(4,pane.width-120));ty=min(max(4,by-24),pane.height-25)
            tw=pd.textlength(text,font=font(17));pd.rounded_rectangle((tx-3,ty-1,tx+tw+7,ty+24),radius=4,fill=(20,29,34,220))
            pd.text((tx+2,ty+1),text,font=font(17),fill='#f0cba3')
    canvas=Image.new('RGB',(1920,1080),BG);d=ImageDraw.Draw(canvas)
    d.text((32,23),'DÉTECTION DES RAILS  /  GABARIT 2D' if objects is not None else 'DÉTECTION DES RAILS  /  TRAJET GPS',font=font(29,True),fill=TEXT)
    d.text((32,66),title+'  ·  '+region,font=font(22),fill=MUTED)
    canvas.paste(pane.convert('RGB'),(tr['pane_x'],tr['pane_y']))
    mx,my,mw,mh=layout['map_box'];mapimage,mapstatus=mapview.image(float(row['video_time_s']))
    d.rounded_rectangle((mx-14,my-53,mx+mw+14,my+mh+213),radius=16,fill=CARD)
    d.text((mx+4,my-41),'REPÈRE GÉOGRAPHIQUE' if mapview.mode=='metadata' else 'TRAJET FERROVIAIRE' if mapview.mode!='none' else 'INFORMATIONS DU CLIP',font=font(20,True),fill=TEXT)
    canvas.paste(mapimage,(mx,my));d=ImageDraw.Draw(canvas)
    d.text((mx+4,my+mh+13),mapview.subtitle,font=font(16),fill=MUTED)
    distance=row.get('route_cumulative_estimate_m','')
    if distance!='' and mapview.total:
        d.text((mx+4,my+mh+45),f'{float(distance):,.0f} m / {mapview.total:,.0f} m'.replace(',',' '),font=font(29,True),fill=TEXT)
        d.text((mx+4,my+mh+82),'Avancement cartographique estimé',font=font(16),fill=MUTED)
    else:
        speed=row.get('gps_speed_provided_m_s') or row.get('train_speed_m_s','')
        d.text((mx+4,my+mh+45),f'{float(speed)*3.6:.1f} km/h' if speed!='' else 'Vitesse non fournie',font=font(28,True),fill=TEXT)
        d.text((mx+4,my+mh+82),'Vitesse du fichier GPS' if speed!='' else 'Pas de distance physique à un objet',font=font(16),fill=MUTED)
    if mapview.gauge:
        d.text((mx+4,my+mh+116),f'Écartement cartographié : {mapview.gauge:.3f} m',font=font(18),fill=MUTED)
    elif mapview.mode=='belgium':
        d.text((mx+4,my+mh+116),'GPS suspect : curseur masqué' if 'withheld' in mapstatus else 'GPS observé · tracé interrompu aux lacunes',font=font(17),fill=MUTED)
    else:d.text((mx+4,my+mh+116),'Écartement réel non confirmé',font=font(18),fill=MUTED)
    if objects is not None:
        count=len(objects);label=f'{count} observation'+('s' if count!=1 else '')+' dans le couloir'
        if reviews:label+=f' · {len(reviews)} à vérifier'
    else:label='Objets non analysés sur cette vidéo'
    d.text((32,955),label,font=font(24,True),fill=TEXT)
    legend_y=1004
    for px,color,text in [(32,'#549c91','Limites intérieures'),(300,'#080a0b','Axe médian lissé'),(575,ORANGE,'Contours des objets')]:
        d.rounded_rectangle((px,legend_y+4,px+38,legend_y+20),radius=3,fill='#c8d0ca' if color=='#080a0b' else BG)
        d.line([(px+3,legend_y+12),(px+35,legend_y+12)],fill=color,width=2)
        d.text((px+48,legend_y),text,font=font(18),fill=MUTED)
    t=float(row['video_time_s']);time_text=f'{int(t)//60:02d}:{t%60:04.1f} / {int(total_time)//60:02d}:{total_time%60:04.1f}'
    d.text((1472,989),time_text,font=font(22,True),fill=TEXT)
    d.text((1472,1022),f"Image source {row['frame_index']}",font=font(17),fill=MUTED)
    d.text((32,1049),'Prototype : recouvrement 2D, pas une alerte de collision. Mètres sous hypothèses.',font=font(16),fill='#7e9096')
    d.text((mx+4,my+mh+173),mapview.attribution,font=font(14),fill='#7e9096')
    return canvas,mapstatus


def build(config,video_id,preview=False):
    config=Path(config).resolve();cfg=json.loads(config.read_text());job=next(j for j in cfg['videos'] if j['video_id']==video_id)
    def resolve(v):
        if v is None:return None
        p=Path(v).expanduser();return p if p.is_absolute() else (config.parent/p).resolve()
    source=resolve(job['source']);destination=resolve(job['output']);proof=destination.parent.parent/'preuves'/video_id
    proof.mkdir(exist_ok=True,parents=True)
    if sha256(source)!=job['source_sha256']:raise ValueError('Source changed')
    gabarit=resolve(job.get('objects_result'));loc=resolve(job.get('gps_result'))
    if job.get('new_predictions'):
        rails=resolve(job['rails_result']);meta=json.loads((rails/'SOURCE_ET_PARAMETRES.json').read_text())
        if meta['status']!='complete' or meta['source_sha256']!=job['source_sha256']:raise ValueError('Incomplete rails')
        if loc:
            route=json.loads((loc/'RESULTATS_LOCALISATION.json').read_text())
        else:route=dict(video_id=video_id,source_video_sha256=job['source_sha256'],matches=[],route_status='gps_not_available',gauge_selected_nominal_m=None)
        enriched=proof/'DONNEES_ENRICHIES'
        if not enriched.exists():build_handoff(rails/'RAILS_SOURCE.csv',route,enriched)
        rows=read_csv(enriched/'RAILS_GAUCHE_DROIT_GPS.csv');points=enriched/'POINTS_RAILS_METRES.csv'
    else:
        rows=read_csv(loc/'GABARIT/RAILS_GAUCHE_DROIT_GPS.csv');points=loc/'GABARIT/POINTS_RAILS_METRES.csv'
    index=timeline(rows)
    if len(index)!=len(rows) or rows[0]['source_video_sha256']!=job['source_sha256']:raise ValueError('Bad rail identity')
    smoothing_report=None
    if job.get('apply_display_smoothing'):
        smoothing_report=smooth_existing(rows,job['apply_display_smoothing'])
        # Recompute conditional scales for the displayed geometry, retain original
        # columns in RAILS.csv. Never call these estimated scales measured gauges.
        enriched=proof/'DONNEES_LISSEES'
        if not enriched.exists():
            excluded={'gps_latitude_deg','gps_longitude_deg','gps_observation_source','gps_observation_status','gps_ocr_confidence','gps_speed_provided_m_s','gps_speed_status','gps_map_status','distance_gps_to_map_m','matched_edge_id','mapped_line','mapped_gauge_nominal_m','gauge_design_default_m','gauge_selection_status','route_cumulative_estimate_m','route_distance_status','route_interval_start_frame','route_interval_end_frame','route_progress_m_per_video_second','route_progress_status','map_dataset_doi','physical_uncertainty_m'}
            basic=[{k:v for k,v in r.items() if k not in excluded} for r in rows]
            csv_write(proof/'RAILS_LISSES_SOURCE.csv',basic)
            route=json.loads((loc/'RESULTATS_LOCALISATION.json').read_text())
            build_handoff(proof/'RAILS_LISSES_SOURCE.csv',route,enriched)
        points=enriched/'POINTS_RAILS_METRES.csv'
    if gabarit:
        meta=json.loads((gabarit/'MANIFESTE.json').read_text())
        if meta['status']!='complete' or meta['source_sha256']!=job['source_sha256'] or meta['video_id']!=video_id:raise ValueError('Object source mismatch')
        if meta['schema_version']==3:
            if json.loads((gabarit/'VERIFICATION.json').read_text())['status']!='passed':raise ValueError('Unverified historical 2D result')
        frames=read_csv(gabarit/'FRAMES.csv')
        if len(frames)!=len(rows):raise ValueError('Objects and rails have different lengths')
        for f in frames:check_frame(f,index)
        validate_geometry(gabarit/'RAILS.csv',points)
    width,height=int(rows[0]['image_width_px']),int(rows[0]['image_height_px'])
    crop=detect_black_bars(source) if video_id=='TSV_record' else None
    portrait=height>width or crop is not None
    tr=geometry_transform(width,height,portrait,crop)
    mapbox=(1080,156,790,570) if portrait else (1480,156,400,570)
    layout=dict(transform=tr,map_box=mapbox)
    mapview=MapView(job['map_mode'],loc,resolve(cfg['africa_network'] if job['map_mode'] in ('africa','metadata') else cfg['belgium_network']) if job['map_mode']!='none' else None,mapbox[2],mapbox[3],proof,job.get('clip_metadata'))
    previews={0,len(rows)//4,len(rows)//2,3*len(rows)//4,len(rows)-1}
    if video_id=='TSV_03383':previews|={5077,5097}
    if not preview:
        destination.mkdir(exist_ok=False,parents=True)
        shutil.copy2(points,destination/'POINTS_RAILS.csv')
    counts=Counter();review_ids=set();all_ids=set();summary_rows=[];clock=[]
    objpath=gabarit/'TXT' if gabarit else None
    started=time.perf_counter();temporary=proof/'VIDEO.partial.mp4';encoder=None;out=None
    if not preview and temporary.exists():raise FileExistsError(temporary)
    rail_handle=(gabarit/'RAILS.csv').open(newline='') if gabarit else None
    rail_groups=FrameGroups(rail_handle,'RAILS.csv') if rail_handle else None
    try:
        with av.open(str(source)) as video:
            if not preview:
                out=av.open(str(temporary),'w',options={'movflags':'+faststart'})
                encoder=out.add_stream('libx264',rate=video.streams.video[0].average_rate or Fraction(30))
                encoder.width,encoder.height,encoder.pix_fmt=1920,1080,'yuv420p'
                encoder.time_base=encoder.codec_context.time_base=Fraction(1,1000000)
                encoder.codec_context.thread_count=4;encoder.options={'crf':'23','preset':'veryfast'}
            for i,frame in enumerate(video.decode(video=0)):
                if i>=len(rows):raise ValueError('More source frames than rail rows')
                row=rows[i];t=float(frame.pts*frame.time_base)
                if int(row['frame_index'])!=i or abs(t-float(row['video_time_s']))>1e-6:raise ValueError('Native frame/time mismatch')
                clock.append(t);rr=rail_groups.take(i) if rail_groups else []
                objects=[] if gabarit else None;reviews=[]
                if gabarit:
                    payload=load_frame(objpath/f'frame_{i:06d}.txt');check_frame(payload,index)
                    reviews=list(payload.get('quarantined_predictions',[]))
                    for obj in payload['objects']:
                        if obj['class_name'].casefold()=='train' and meta['schema_version']!=3:
                            mask,_=object_masks(obj,width,height);corridor,_,_=corridor_from_rows(rr,width,height)
                            guard=train_guard(dict(obj,mask=mask.astype(bool)),corridor.astype(bool))
                            if guard['quarantine']:
                                reviews.append(dict(obj,review_reason=guard['reason']));continue
                        objects.append(obj)
                    for obj in objects:all_ids.add(obj['observation_id']);counts[obj['class_name']]+=1
                    for obj in reviews:review_ids.add(obj['observation_id'])
                mapstatus=mapview.position(t)[1]
                if not preview or i in previews:
                    image,mapstatus=paint_frame(frame,row,objects,reviews,mapview,layout,job['title'],job['region'],float(rows[-1]['video_time_s']))
                    if i in previews:image.save(proof/f'APERCU_{i:06d}.jpg',quality=91)
                    if not preview:
                        vf=av.VideoFrame.from_image(image);vf.pts=round(t*1000000);vf.time_base=Fraction(1,1000000)
                        for packet in encoder.encode(vf):out.mux(packet)
                if not preview:summary_rows.append(compact_rail_row(row,tr,len(objects) if objects is not None else None,len(reviews),mapstatus))
                if not preview and (i+1)%600==0:print(f'{video_id} : rendu {i+1}/{len(rows)}',flush=True)
            if len(clock)!=len(rows):raise ValueError('Incomplete source decode')
            if not preview:
                for packet in encoder.encode():out.mux(packet)
                out.close();out=None
    finally:
        if rail_handle:rail_handle.close()
        if out:out.close()
    report=dict(video_id=video_id,source_sha256=job['source_sha256'],frames=len(clock),first_pts_s=clock[0],last_pts_s=clock[-1],source_dimensions=[width,height],display_transform=tr,objects_by_class=dict(counts),object_observations=sum(counts.values()),quarantined_observations=len(review_ids),axis_style='thin_black_2px',smoothing=smoothing_report or {'mode':'existing_paired_causal_smoothing_preserved','max_gap_seconds':.15,'min_cutoff_hz':.8,'beta':15},map=mapview.audit,preview_only=preview,accuracy_not_remeasured=True)
    write_json(proof/'CONTROLE_PREPARATION.json',report)
    if preview:return report
    csv_write(destination/'RAILS.csv',summary_rows)
    if job.get('clip_metadata'):
        csv_write(destination/'CONTEXTE_VIDEO.csv',[dict(video_id=video_id,**job['clip_metadata'],video_time_s='',scope='single_file_metadata_position_not_synchronised_to_frames')])
    if gabarit:
        retained,reviewed=[],[]
        fields=None
        for name in ['OBJETS.csv','A_VERIFIER.csv']:
            if not (gabarit/name).exists():continue
            with (gabarit/name).open(newline='') as f:
                reader=csv.DictReader(f)
                if fields is None:fields=[x for x in reader.fieldnames if x!='txt_file']
                for obj in reader:
                    record={k:obj.get(k,'') for k in fields}
                    if obj['observation_id'] in all_ids:retained.append(record)
                    elif obj['observation_id'] in review_ids:reviewed.append(dict(record,review_reason='ambiguous_train_tracklike'))
                    else:raise ValueError('Unmatched object observation')
        csv_write(destination/'OBJETS.csv',retained,fields)
        if reviewed:csv_write(destination/'A_VERIFIER.csv',reviewed,fields+['review_reason'])
        # Contours include every exported observation, with status to distinguish review.
        with (gabarit/'POINTS_OBJETS.csv').open(newline='') as fi,(destination/'POINTS_OBJETS.csv').open('w',newline='') as fo:
            reader=csv.DictReader(fi);writer=csv.DictWriter(fo,fieldnames=reader.fieldnames+['observation_status']);writer.writeheader()
            for p in reader:
                oid=p['observation_id']
                if oid not in all_ids and oid not in review_ids:raise ValueError('Unknown contour observation')
                writer.writerow(dict(p,observation_status='to_review' if oid in review_ids else 'retained'))
    # Full decode verifies native timestamps, not just the first and last frames.
    decoded=0;max_error=0
    with av.open(str(temporary)) as v:
        for i,frame in enumerate(v.decode(video=0)):
            if i>=len(clock):raise ValueError('Extra encoded frames')
            err=abs(float(frame.pts*frame.time_base)-clock[i]);max_error=max(max_error,err)
            if err>2e-6:raise ValueError('Encoded PTS changed')
            decoded+=1
    if decoded!=len(clock):raise ValueError('Incomplete output video')
    temporary.replace(destination/'VIDEO_ANNOTEE.mp4')
    note=readme(job,rows,counts,review_ids,tr,mapview)
    if job.get('clip_metadata'):
        note+='\nREPÈRE FIXE DU FICHIER\nCONTEXTE_VIDEO.csv contient une seule position issue des métadonnées du clip.\nSon instant de mesure est inconnu : elle ne doit pas être répétée comme une\nposition GPS mesurée sur chaque frame. La carte affiche ce repère fixe, pas un\ntrajet reconstruit. Les champs GPS par frame et la vitesse restent vides.\n'
    (destination/'LIRE_MOI.txt').write_text(note,encoding='utf-8')
    files=sorted(destination.iterdir())
    if any(not p.is_file() or p.name!='LIRE_MOI.txt' and p.suffix not in {'.mp4','.csv'} for p in files):raise ValueError('Unexpected file in minimal delivery')
    hashes={p.name:sha256(p) for p in files}
    archive=destination.with_suffix('.zip')
    if archive.exists():raise FileExistsError(archive)
    with zipfile.ZipFile(archive,'x') as z:
        for p in files:z.write(p,p.name,compress_type=zipfile.ZIP_STORED if p.suffix=='.mp4' else zipfile.ZIP_DEFLATED,compresslevel=5)
    with zipfile.ZipFile(archive) as z:
        for name,digest in hashes.items():
            h=hashlib.sha256()
            with z.open(name) as f:
                for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
            if h.hexdigest()!=digest:raise ValueError('ZIP file altered')
    report.update(status='passed',frames_decoded_and_checked=decoded,max_timestamp_error_s=max_error,output_files_sha256=hashes,archive_sha256=sha256(archive),archive_bytes=archive.stat().st_size,runtime_s=time.perf_counter()-started)
    write_json(proof/'VERIFICATION_FINALE.json',report)
    print(json.dumps({'video_id':video_id,'zip':str(archive),'frames':decoded,'status':'passed'},ensure_ascii=False),flush=True)
    return report


def readme(job,rows,counts,review_ids,tr,mapview):
    return f'''{job['title']} — Rails, gabarit 2D et GPS

OUVRIR EN PREMIER
VIDEO_ANNOTEE.mp4 : limites intérieures des rails en vert discret, axe médian
en fine ligne noire, contours d'objets en orange lorsqu'ils ont été analysés.
RAILS.csv : une ligne par image source ; rails gauche et droit, milieu,
horodatage, GPS/vitesse disponibles, statut et transformation d'affichage.
POINTS_RAILS.csv : chaque paire de points des deux limites, en pixels,
et conversions transversales en mètres sous les hypothèses indiquées.
OBJETS.csv et POINTS_OBJETS.csv, s'ils sont présents : observations retenues
et points de leurs contours. A_VERIFIER.csv, s'il est présent : hypothèses
ambiguës conservées pour relecture. « full_predicted_mask » désigne tout le
contour prédit ; « corridor_intersection » uniquement sa partie dans le couloir.

VIDÉO ET SYNCHRONISATION
Identifiant : {job['video_id']}
Source : {Path(job['source']).name}
Empreinte SHA-256 : {job['source_sha256']}
Images décodées : {len(rows)} ; temps natifs : {float(rows[0]['video_time_s']):.6f}
à {float(rows[-1]['video_time_s']):.6f} secondes.
Joindre les CSV avec video_id + frame_index ; vérifier aussi le temps vidéo.
Le temps s'appelle video_time_s dans les CSV rails, et time_video_s dans les
CSV objets. Les deux désignent le même instant de la même vidéo, en secondes.
frame_index commence à 0 et compte les images réellement décodées. Le temps
est celui du fichier, jamais recalculé par frame_index / une cadence supposée.
La vidéo conserve tous ces instants et les éventuelles lacunes de décodage.
Le son n'est pas inclus.

COORDONNÉES
Les pixels désignent l'image source de {rows[0]['image_width_px']} × {rows[0]['image_height_px']},
origine en haut à gauche, x vers la droite, y vers le bas. Les champs JSON
contiennent des segments de points [x,y] : ne pas relier deux segments distincts.
Les CSV restent dans ce repère, même si la vidéo comporte une carte latérale.
Pour retrouver un point dans la vidéo annotée :
x_affiché = display_pane_x_px + (x - display_crop_x_px) * display_scale_x
y_affiché = display_pane_y_px + (y - display_crop_y_px) * display_scale_y
{('Attention : le clip filmé de côté a été réorienté de 90 degrés dans le sens horaire. Les CSV utilisent cette source redressée, 1080 × 1920. Depuis le raster original : x_redressé=1079-y_original, y_redressé=x_original.' if job.get('orientation_original') else '')}

LISSAGE
Les deux limites sont filtrées ensemble par le filtre causal adaptatif déjà
utilisé. L'axe reste exactement leur milieu horizontal. Le filtre réagit plus
vite aux mouvements importants et se réinitialise aux sauts/interruption ; il
ne prolonge pas un rail absent. Pour la vidéo à 5 images/s, la limite de rupture
est adaptée à 0,35 s ; les limites brutes sont aussi conservées dans RAILS.csv.
Sur les autres vidéos : coupure minimale 0,8 Hz, beta 15, rupture à 0,15 s.
L'affichage plus stable ne prouve pas à lui seul une meilleure précision.

GPS, VITESSE ET MÈTRES
{('Carte AfTS-Db v3 : parcours estimé en raccordant les GPS lus sur la vidéo au réseau. Le curseur est interpolé entre lectures acceptées, sans combler une lacune supérieure à 5 secondes.' if job['map_mode']=='africa' else 'Carte Infrabel : positions du CSV GPS sur le réseau belge. La voie exacte et la longueur totale parcourue ne sont pas certifiées. Les positions suspectes sont masquées sur la carte, mais les GPS originaux restent disponibles.' if job['map_mode']=='belgium' else 'Une position de métadonnées est affichée comme repère fixe. Aucun instant ni trajet n’est connu.' if job['map_mode']=='metadata' else 'Aucun GPS exploitable fourni pour ce clip : pas de carte, de vitesse ou de trajet inventés.')}
train_speed_m_s : vitesse fournie dans le fichier GPS si elle existe. Pour une
vidéo accélérée, cette vitesse utilise le temps réel du fichier GPS, pas le
temps accéléré de lecture. Une cellule vide signifie inconnu, jamais zéro.
Écartement nominal cartographié : {f'{mapview.gauge:.3f} m' if mapview.gauge else 'non confirmé pour cette vidéo'}.
Défaut de dimensionnement : 1,676 m. Ce défaut n'est PAS une mesure de la voie.
À une hauteur d'image y : échelle locale ≈ écartement supposé / largeur en pixels.
Les décalages transversaux en mètres sont relatifs au milieu de l'image ; ils
restent conditionnels à la perspective et à l'écartement. Les sections trop
étroites sont exclues. Ce n'est ni une calibration 3D, ni la distance à un objet,
ni une mesure indépendante de l'écartement. L'erreur réelle en mètres reste vide.

CE QUI A CHANGÉ
Présentation plus lisible, axe noir discret, mêmes principes de lissage.
Ajout du contexte GPS et de la carte lorsque disponibles ; mètres conditionnels
avec hypothèse explicite. Aucun nouvel entraînement et aucun gain de précision
terrain revendiqué pour cette livraison. La détection a été exécutée sur les
nouveaux clips avec les modèles sélectionnés précédemment.

RÉSULTATS ET LIMITES
{f'Objets : {sum(counts.values())} observations retenues ; {len(review_ids)} à relire.' if job.get('objects_result') else 'Les objets ne sont pas analysés sur cette vidéo ; seuls les rails et le GPS sont livrés.'}
Une observation correspond à un objet prédit sur une image : le même objet peut
revenir dans plusieurs images. Un recouvrement 2D ne confirme pas un danger
physique ; aucun objet détecté ne certifie pas une voie libre.
Les scores internes de confiance ne sont pas des pourcentages de précision.
Référence disponible pour les rails : Dice moyen 97,27 % et erreurs moyennes
gauche 3,74 px / droite 5,43 px sur 12 images de développement, une seule capture.
Ces valeurs ne sont PAS les erreurs connues de chaque frame des vidéos livrées.
Le modèle de déchets reste faible pour les petits objets ; aucun gain nouveau
sur ce point n'est démontré ici. Cartographie : {mapview.attribution or 'non disponible'}.
'''


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--video-id',required=True);p.add_argument('--preview',action='store_true')
    a=p.parse_args();build(a.config,a.video_id,a.preview)
