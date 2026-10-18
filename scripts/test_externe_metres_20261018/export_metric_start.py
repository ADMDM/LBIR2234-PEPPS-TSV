"""V3 adds gap-aware travel distance and explicitly conditional lateral metre proxies."""
import sys,csv,json,math,collections,shutil,os
from pathlib import Path
import numpy as np
from PIL import Image
R=Path(sys.argv[1]);BASE=R/'03_RESULTATS/GABARIT_2_VIDEOS_V2_20261018';OUT=R/'03_RESULTATS/GABARIT_V3_METRES_ESTIMES_20261018';OUT.mkdir(exist_ok=False)
REPORT=R/'03_RESULTATS/TESTS_ET_METRES_20261018/03_PREMIERS_METRES';REPORT.mkdir(parents=True,exist_ok=False)
def read(p):
 with p.open(newline='') as f:yield from csv.DictReader(f)
def number(v):return float(v) if v not in ['',None] else None
def write(p,rows,fields=None):
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def link_tree(src,dst):
 dst.mkdir(exist_ok=False)
 for p in src.iterdir():
  target=dst/p.name
  if p.is_dir():link_tree(p,target)
  else:os.link(p,target)

reference_width={};maskdir=BASE/'02_TSV_03676_180S/REFERENCES_12'
for p in maskdir.glob('*_mask.png'):
 idx=int(p.stem.removesuffix('_mask').split('_')[-1]);mask=np.asarray(Image.open(p))
 for y in range(0,mask.shape[0],8):
  xs=np.flatnonzero(mask[y]==1)
  if len(xs)>1 and np.all(np.diff(xs)==1):reference_width[idx,y]=int(xs[-1]-xs[0])

err=[]
for r in read(BASE/'ERREURS_SUR_FRAMES_REFERENCE.csv'):
 width=reference_width[int(r['frame_index']),int(r['y_px'])];allowed=width>=.02*1920
 e=float(r['error_selected_px'])
 err.append(dict(frame_index=int(r['frame_index']),y_px=int(r['y_px']),reference_width_px=width,error_signed_px=e,metric_status='conditional_local_scale_from_human_reference' if allowed else 'rejected_narrow_reference',error_equivalent_m_g1067=e*1.067/width if allowed else None,error_equivalent_m_g1435=e*1.435/width if allowed else None,note='Development reference; approximate local horizontal scale; not independently measured physical error'))
write(REPORT/'ERREURS_EQUIVALENTES_METRES.csv',err);write(OUT/'ERREURS_EQUIVALENTES_METRES.csv',err)
byframe=[]
for i in sorted({r['frame_index'] for r in err}):
 a=[r for r in err if r['frame_index']==i and r['error_equivalent_m_g1067'] is not None]
 byframe.append(dict(frame_index=i,valid_sections=len(a),mae_equiv_m_g1067=float(np.mean([abs(r['error_equivalent_m_g1067']) for r in a])) if a else None,mae_equiv_m_g1435=float(np.mean([abs(r['error_equivalent_m_g1435']) for r in a])) if a else None))
write(REPORT/'ERREURS_METRES_PAR_IMAGE.csv',byframe)
values=[abs(r['error_equivalent_m_g1067']) for r in err if r['error_equivalent_m_g1067'] is not None]
stats=dict(reference_frames=len(byframe),all_reference_sections=len(err),convertible_sections=len(values),narrow_sections_rejected=len(err)-len(values),macro_image_mae_equiv_m_g1067=float(np.mean([r['mae_equiv_m_g1067'] for r in byframe if r['mae_equiv_m_g1067'] is not None])),macro_image_mae_equiv_m_g1435=float(np.mean([r['mae_equiv_m_g1435'] for r in byframe if r['mae_equiv_m_g1435'] is not None])),section_p95_abs_equiv_m_g1067=float(np.percentile(values,95)),scope='Conditional conversion of actual pixel residuals using HUMAN reference width, not predicted width. One development capture. Neither field accuracy nor calibrated uncertainty.',gauge_sensitivity_ratio=1.435/1.067)
(REPORT/'ERREUR_METRES_CONDITIONNELLE.json').write_text(json.dumps(stats,indent=2));(OUT/'ERREUR_METRES_CONDITIONNELLE.json').write_text(json.dumps(stats,indent=2))

allcounts=[]
for folder,gauge in [('01_TSV_RECORD',1.435),('02_TSV_03676_180S',1.067)]:
 src=BASE/folder;dst=OUT/folder;dst.mkdir()
 for sub in ['IMAGES_COMMUNES','REFERENCES_12','VITESSE_ESTIMEE_GPS']:
  if (src/sub).exists():link_tree(src/sub,dst/sub)
 for name in ['SOURCE_ET_DECOUPAGE.json','GPS_SOURCE.csv']:
  if (src/name).exists():shutil.copy2(src/name,dst/name)
 frames=list(read(src/'FRAMES_VITESSE_VECTEUR.csv'));lookup={};segments=[];prev=None;seg=-1;inside=0.;known=0.;continuous=True;valid_steps=0;invalid_steps=0;covered=0.;reason_counts=collections.Counter()
 for r in frames:
  t=number(r['real_elapsed_s']);speed=number(r['train_speed_m_s']);is_overlay=r['speed_status'].startswith('estimated_');basic=t is not None and speed is not None and speed>=0
  # Delivered source flags mark suspect telemetry; overlay flags document its conditional nature.
  good=basic and (is_overlay or not r['telemetry_flags'])
  interval=None;why='start'
  if prev is not None:
   pt,pv,pg=prev;dt=t-pt if t is not None and pt is not None else None
   if good and pg and dt is not None and 0<dt<=5:
    interval=dt*pv if is_overlay else .5*(pv+speed)*dt;inside+=interval;known+=interval;covered+=dt;valid_steps+=1;why='valid_estimated_interval' if is_overlay else 'valid_trapezoid_from_telemetry'
   else:
    invalid_steps+=1;continuous=False;why='telemetry_missing_or_flagged' if not (good and pg) else 'time_gap_over_5s_or_nonpositive';reason_counts[why]+=1
  if interval is None and good:
   seg+=1;inside=0.
  r.update(assumed_gauge_m=gauge,gauge_status='plausible_nominal_assumption_not_confirmed_for_exact_track',metric_status='conditional_lateral_scale_no_camera_calibration',travel_segment_id=seg if good else None,travel_segment_distance_m_estimate=inside if good else None,travel_from_video_start_m_estimate=known if good and continuous else None,travel_cumulative_valid_parts_m_estimate=known,travel_interval_distance_m_estimate=interval,travel_interval_status=why,travel_interval_duration_s=(t-prev[0]) if prev is not None and t is not None and prev[0] is not None else None,travel_distance_uncertainty_m=None,ground_depth_z_m=None)
  lookup[int(r['frame_index'])]=r;prev=(t,speed,good)
 write(dst/'FRAMES_VITESSE_VECTEUR.csv',frames)
 counts=dict(video_id=frames[0]['video_id'],frames=len(frames),assumed_gauge_m=gauge,valid_motion_intervals=valid_steps,rejected_motion_intervals=invalid_steps,integrated_time_s=covered,cumulative_distance_of_valid_parts_m=known,continuous_distance_from_video_start_available_to_end=continuous,motion_segments=seg+1,rejection_reasons=dict(reason_counts),notes='Distances are speed/time integrals, not visual rail depth. Gaps start new segments; cumulative valid parts excludes missing stretches. Unknown instrumental uncertainty stays blank.')
 allcounts.append(counts)
 # Stream points; no false measured error assigned to unannotated video frames.
 extra=['assumed_gauge_m','metric_status','scale_m_per_px_assumed','axis_offset_from_image_midline_m_assumed','left_offset_from_image_midline_m_assumed','right_offset_from_image_midline_m_assumed','axis_offset_m_scenario_1067','axis_offset_m_scenario_1435','measured_gauge_m','ground_x_m','ground_z_m','physical_uncertainty_m','reference_width_px','reference_error_equiv_m_assumed','reference_error_metric_status','travel_segment_id','travel_segment_distance_m_estimate']
 with (src/'AXE_MEDIAN.csv').open(newline='') as f,(dst/'AXE_MEDIAN.csv').open('x',newline='') as g:
  rd=csv.DictReader(f);wr=csv.DictWriter(g,fieldnames=rd.fieldnames+extra);wr.writeheader()
  for row in rd:
   fr=lookup[int(row['frame_index'])];row.update({k:None for k in extra});row.update(assumed_gauge_m=gauge,metric_status='no_valid_axis',reference_error_metric_status='no_reference',travel_segment_id=fr['travel_segment_id'],travel_segment_distance_m_estimate=fr['travel_segment_distance_m_estimate'])
   if row['x_px']:
    x,l,rr,w=map(float,[row['x_px'],row['left_x_px'],row['right_x_px'],row['image_width_px']]);width=rr-l;origin=(w-1)/2
    if width>=.02*w:
     scale=gauge/width;row.update(metric_status='nominal_gauge_local_horizontal_proxy_not_calibrated',scale_m_per_px_assumed=scale,axis_offset_from_image_midline_m_assumed=(x-origin)*scale,left_offset_from_image_midline_m_assumed=(l-origin)*scale,right_offset_from_image_midline_m_assumed=(rr-origin)*scale,axis_offset_m_scenario_1067=(x-origin)*1.067/width,axis_offset_m_scenario_1435=(x-origin)*1.435/width)
    else:row['metric_status']='rejected_narrow_predicted_corridor'
    if row['error_signed_px']:
     rw=reference_width[int(row['frame_index']),int(row['y_px'])];row['reference_width_px']=rw
     if rw>=.02*w:row.update(reference_error_equiv_m_assumed=float(row['error_signed_px'])*gauge/rw,reference_error_metric_status='conditional_residual_using_human_reference_scale')
     else:row['reference_error_metric_status']='rejected_narrow_reference'
   wr.writerow(row)
 # Small exchange examples share exactly the original example frame IDs.
 ids={int(r['frame_index']) for r in read(src/'EXEMPLE_SEQUENCE_FRAMES.csv')};write(dst/'EXEMPLE_SEQUENCE_FRAMES.csv',[lookup[i] for i in sorted(ids)])
 with (dst/'AXE_MEDIAN.csv').open(newline='') as f,(dst/'EXEMPLE_SEQUENCE_POINTS.csv').open('x',newline='') as g:
  rd=csv.DictReader(f);wr=csv.DictWriter(g,fieldnames=rd.fieldnames);wr.writeheader();wr.writerows(r for r in rd if int(r['frame_index']) in ids)
 print(folder,json.dumps(counts),flush=True)

(REPORT/'AVANCEMENT_PAR_VIDEO.json').write_text(json.dumps(allcounts,indent=2));(OUT/'AVANCEMENT_PAR_VIDEO.json').write_text(json.dumps(allcounts,indent=2))
for f in ['ERREURS_SUR_FRAMES_REFERENCE.csv','RESUME_ERREURS_PAR_FRAME.csv','ERREUR_REFERENCE_RESUME.json']:
 shutil.copy2(BASE/f,OUT/f)
link_tree(BASE/'outils',OUT/'outils')
provenance=json.loads((BASE/'PROVENANCE.json').read_text());provenance.update(version='V3 conditional metres, same raw model predictions as V2',metres='First-order lateral scale from assumed nominal gauge, flat local horizontal cross-section assumption. Pixel midpoint is not certified 3D midline. Forward travel integrates speed/physical time; it is not depth from camera. Gauge and camera geometry unvalidated.',gauge_sources=[{'video':'TSV_03676','assumed_gauge_m':1.067,'source':'https://nrc.gov.ng/','basis':'NRC describes historical cape gauge; visible video GPS in northern Nigeria. Plausible assumption, not confirmation of exact track.'},{'video':'TSV_record','assumed_gauge_m':1.435,'basis':'Engineering scenario for the Belgian GPS location; must be confirmed for the precise filmed line.'}],frozen_external_test='../TESTS_ET_METRES_20261018/01_TEST_EXTERNE_FIGE/RESULTAT.json')
(OUT/'PROVENANCE.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
print(json.dumps(stats,indent=2),flush=True)
