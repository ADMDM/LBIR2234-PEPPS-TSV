"""Add explicitly experimental interval-mean GPS speed to the unreleased V2.

19 displayed DMS/time anchors were visually read from original decoded crops.
OCR is kept as evidence only: it produced false digits and is not trusted blindly.
No instantaneous speed, UTC, metric rail geometry or instrument uncertainty is claimed.
"""
import csv,json,math,hashlib,shutil,sys
from pathlib import Path
import numpy as np

# Anchors are local measurements. They are not stored in the repository.
ANCHORS_CSV=Path(__file__).resolve().parents[2]/'data/telemetry/overlay_anchors.csv'

def distance_m(a,b):
 p,q=map(math.radians,[a['latitude_deg'],b['latitude_deg']]);dl=q-p;dn=math.radians(b['longitude_deg']-a['longitude_deg'])
 z=math.sin(dl/2)**2+math.cos(p)*math.cos(q)*math.sin(dn/2)**2
 return 2*6371008.8*math.asin(min(1,math.sqrt(z)))

def build():
 with ANCHORS_CSV.open(newline='',encoding='utf-8') as f:
  points=list(csv.DictReader(f))
 if len(points)<2:raise ValueError('Au moins deux ancrages GPS relus sont nécessaires.')
 for p in points:
  for key in ['video_time_s','overlay_elapsed_s','latitude_deg','longitude_deg']:p[key]=float(p[key])
  p['frame_index']=int(p['frame_index'])
  if not all(math.isfinite(p[k]) for k in ['video_time_s','overlay_elapsed_s','latitude_deg','longitude_deg']):raise ValueError('Ancrage non fini')
  if not -90<=p['latitude_deg']<=90 or not -180<=p['longitude_deg']<=180:raise ValueError('Coordonnées GPS invalides')
  p.setdefault('clock_display','');p['latitude_dms']='';p['longitude_dms']=''

 intervals=[]
 for a,b in zip(points,points[1:]):
  dt=b['overlay_elapsed_s']-a['overlay_elapsed_s'];distance=distance_m(a,b)
  assert dt>0 and b['frame_index']>a['frame_index']
  intervals.append(dict(start_frame=a['frame_index'],end_frame=b['frame_index'],start_video_s=a['video_time_s'],end_video_s=b['video_time_s'],start_clock_elapsed_s=a['overlay_elapsed_s'],end_clock_elapsed_s=b['overlay_elapsed_s'],duration_clock_s=dt,chord_distance_m=distance,speed_m_s=distance/dt,speed_km_h=distance/dt*3.6))
 return points,intervals

def at_frame(i,points,intervals):
 t=i/30
 # Never extrapolate beyond the last visually checked clock/position.
 if t<points[0]['video_time_s'] or t>points[-1]['video_time_s']:return None
 j=min(max(0,int(np.searchsorted([a['end_frame'] for a in intervals],i,side='right'))),len(intervals)-1)
 a=intervals[j];q=(t-a['start_video_s'])/(a['end_video_s']-a['start_video_s'])
 return a,a['start_clock_elapsed_s']+q*a['duration_clock_s']

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def main(root):
 R=Path(root);O=R/'03_RESULTATS/GABARIT_2_VIDEOS_V2_20261018';D=O/'02_TSV_03676_180S';E=D/'VITESSE_ESTIMEE_GPS';E.mkdir(exist_ok=False)
 B=R/'runs/fiabilisation_20261018/avant_ajout_vitesse_incrustee';B.mkdir(exist_ok=False)
 assert not (O/'RELEASED.json').exists()
 points,intervals=build()
 for name,data in [('ANCRAGES_RELUS.csv',points),('INTERVALLES_VITESSE.csv',intervals)]:
  with (E/name).open('x',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 shutil.copy2('/tmp/tsv_gps_contact.jpg',E/'ANCRAGES_VISIBLES.jpg')
 shutil.copy2('/tmp/tsv_gps_overlay_180/ocr.jsonl',B/'ocr_180_non_valide.jsonl')
 shutil.copy2('/tmp/tsv_gps_audit_v2/AUDIT.json',B/'AUDIT_OCR_AUTOMATIQUE.json')
 checked=E/'extraits_relus';checked.mkdir()
 for p in points:shutil.copy2(Path('/tmp/tsv_gps_overlay_180')/f"frame_{p['frame_index']:06d}.png",checked/f"frame_{p['frame_index']:06d}.png")
 def rewrite(path):
  shutil.copy2(path,B/path.name)
  tmp=path.with_suffix('.tmp.csv');count=covered=0
  with path.open(newline='') as src,tmp.open('x',newline='') as dst:
   rd=csv.DictReader(src);fields=rd.fieldnames+['speed_interval_start_frame','speed_interval_end_frame','speed_interval_duration_s','speed_is_estimated']
   wr=csv.DictWriter(dst,fieldnames=fields);wr.writeheader()
   for row in rd:
    match=at_frame(int(row['frame_index']),points,intervals);count+=1
    row.update(speed_interval_start_frame='',speed_interval_end_frame='',speed_interval_duration_s='',speed_is_estimated=True)
    if match:
     a,clock=match;covered+=1
     row.update(train_speed_km_h=a['speed_km_h'],train_speed_m_s=a['speed_m_s'],speed_status='estimated_interval_mean_from_burned_gps_unvalidated',real_elapsed_s=clock,telemetry_flags='interval_average_not_instantaneous;display_clock_interpolated;instrument_accuracy_unknown;not_independently_validated',speed_interval_start_frame=a['start_frame'],speed_interval_end_frame=a['end_frame'],speed_interval_duration_s=a['duration_clock_s'])
     if 'time_basis' in row:row['time_basis']='interpolated_burned_clock_unvalidated'
    else:
     row.update(train_speed_km_h='',train_speed_m_s='',real_elapsed_s='',speed_status='outside_visually_checked_gps_interval',telemetry_flags='no_extrapolation_past_last_anchor',speed_is_estimated=False)
     if 'time_basis' in row:row['time_basis']='video_playback_only'
    wr.writerow(row)
  tmp.replace(path);return dict(rows=count,rows_with_estimated_speed=covered)
 counts={n:rewrite(D/n) for n in ['FRAMES_VITESSE_VECTEUR.csv','AXE_MEDIAN.csv','EXEMPLE_SEQUENCE_FRAMES.csv','EXEMPLE_SEQUENCE_POINTS.csv']}
 specpath=D/'SOURCE_ET_DECOUPAGE.json';shutil.copy2(specpath,B/specpath.name);spec=json.loads(specpath.read_text());spec.update(source_time_is_physical_time=False,physical_clock_note='Relative clock interpolated between 19 visually transcribed burned-in GPS/time anchors. Experimental, quantized displayed seconds, no UTC or synchronization accuracy validated.',speed_source='VITESSE_ESTIMEE_GPS/ANCRAGES_RELUS.csv',speed_kind='interval_mean_estimate_9_to_11_s_not_instantaneous');specpath.write_text(json.dumps(spec,indent=2))
 p=O/'PROVENANCE.json';shutil.copy2(p,B/p.name);prov=json.loads(p.read_text())
 for v in prov['videos']:
  if v['video_id']=='TSV_03676':v.update(speed_available=True,speed_kind='experimental_interval_mean_from_burned_gps',speed_observed_in_instrument_log=False);v['counts']['speed_missing']=28;v['counts']['speed_estimated']=5371
 p.write_text(json.dumps(prov,indent=2))
 p=O/'COLONNES.json';col=json.loads(p.read_text());col.update(speed='TSV_record: supplied telemetry. TSV_03676: experimental interval-mean estimate from displayed GPS, explicitly flagged. Blank when unavailable. Not all values are instantaneous measured speed.',real_elapsed_s='TSV_record: telemetry UTC elapsed. TSV_03676: linear interpolation of visually read displayed clock; experimental, no absolute UTC.',speed_is_estimated='True for estimated 03676 values; TSV_record uses speed_status provided/provided_with_quality_flags; absent column there means use speed_status.',speed_interval_start_frame='03676 only: first normalized anchor frame of the averaging interval',speed_interval_end_frame='03676 only: last normalized anchor frame of the averaging interval',speed_interval_duration_s='03676 only: elapsed displayed-clock seconds, 9 to 11 seconds');p.write_text(json.dumps(col,indent=2))
 speeds=[a['speed_km_h'] for a in intervals]
 meta=dict(status='experimental_not_independently_validated',method='Spherical great-circle chord distance (mean Earth radius 6371008.8 m) between visually checked DMS coordinates divided by displayed elapsed seconds. Constant interval-mean speed assigned to frames within each interval; relative clock linearly interpolated.',anchor_count=len(points),interval_count=len(intervals),covered_frames=5371,missing_frames=28,covered_frame_range=[0,5370],video_frames=5399,speed_km_h_min=min(speeds),speed_km_h_max=max(speeds),distance_chord_sum_m=sum(a['chord_distance_m'] for a in intervals),interval_duration_range_s=[9,11],absolute_date_text='01/09/2017',timezone='unknown; UTC intentionally blank',accuracy='No independent GNSS log, speedometer, synchronization or instrument accuracy available. OCR had false digits despite high confidence; only visually checked anchors used. Original overlay itself is not ground truth.',limitations=['Interval mean, not instantaneous speed at every frame','Piecewise constant values do not demonstrate abrupt real acceleration','Chord distances can underestimate curved travel and GPS noise can inflate them','Displayed clock quantized to seconds; varying GPS refresh/delay unknown','Offline future endpoint is used within each interval, not a real-time estimate','No extrapolation for final 28 frames','Rail positions remain pixels; train GPS does not calibrate camera or gauge'],counts=counts)
 (E/'METHODE_ET_LIMITES.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
 (E/'LIRE_MOI.txt').write_text('VITESSE EXPÉRIMENTALE — 03676\n\nLes valeurs sont calculées sur 18 intervalles de 9 à 11 secondes à partir de 19 positions/heures GPS visibles et relues dans la vidéo. Ce sont des vitesses MOYENNES, pas des vitesses instantanées mesurées.\nLes colonnes train_speed_m_s et train_speed_km_h portent ces estimations avec speed_status=estimated_interval_mean_from_burned_gps_unvalidated. Les frames 0 à 5370 sont couvertes ; les 28 dernières restent vides, sans extrapolation.\nreal_elapsed_s est interpolé entre les heures affichées ; utc_time reste vide car le fuseau n’est pas connu. Pour intégrer une distance approchée, utiliser ce temps relatif, et traiter les positions répétées comme telles. Ne pas présenter cette trajectoire comme une mesure réelle validée.\nL’incertitude instrumentale et l’erreur réelle de vitesse ne sont pas connues ; speed_uncertainty_m_s reste vide. ANCRAGES_RELUS.csv et les extraits PNG permettent une vérification par une autre personne.\n')
 shutil.copy2(Path(__file__),R/'scripts/fiabilisation_20261018'/Path(__file__).name)
 print(json.dumps(meta,ensure_ascii=False,indent=2))

if __name__=='__main__':main(sys.argv[1])
