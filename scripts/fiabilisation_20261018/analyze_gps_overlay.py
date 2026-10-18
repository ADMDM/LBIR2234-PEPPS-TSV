"""Audit OCR GPS; experimental finite-window speed, never measured instantaneous speed."""
import json,re,csv,math,sys
from pathlib import Path
import numpy as np
source=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
raw=[json.loads(l) for l in source.read_text().splitlines()]
rows=[]
for obj in raw:
 idx=int(Path(obj['path']).stem.split('_')[-1]);texts=[r['text'] for r in obj.get('lines',[])];txt=' | '.join(texts)
 row=dict(frame_index=idx,video_time_s=idx/30,ocr_text=txt,status='parse_failed',clock_display='',clock_elapsed_s='',latitude_deg='',longitude_deg='')
 tm=re.search(r'(\d{2}):(\d{2}):(\d{2})',txt)
 # Integer latitude degrees, separator can look like a decimal point in OCR.
 gps=re.search(r'N\s*(\d{1,2})[.\s]*(\d{2})\D+(\d+(?:[.,]\d+)?)\D+E\s*(\d{1,3}(?:\.0)?)\D+(\d{1,2})\D+(\d+(?:[.,]\d+)?)',txt)
 if tm and gps:
  h,m,s=map(int,tm.groups());a,b,c,d,e,f=[float(x.replace(',','.')) for x in gps.groups()]
  if 0<=h<24 and 0<=m<60 and 0<=s<60 and 0<=a<90 and 0<=d<180 and 0<=b<60 and 0<=e<60 and 0<=c<60 and 0<=f<60:
   row.update(status='parsed_not_independently_validated',clock_display=tm.group(),clock_seconds_of_day=h*3600+m*60+s,latitude_deg=a+b/60+c/3600,longitude_deg=d+e/60+f/3600)
 rows.append(row)
valid=[r for r in rows if r['status'].startswith('parsed')]
base=valid[0]['clock_seconds_of_day']
for r in valid:r['clock_elapsed_s']=r['clock_seconds_of_day']-base
def distance(a,b):
 lat1,lat2=map(math.radians,[a['latitude_deg'],b['latitude_deg']]);dl=lat2-lat1;dn=math.radians(b['longitude_deg']-a['longitude_deg'])
 z=math.sin(dl/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dn/2)**2
 return 2*6371008.8*math.asin(min(1,math.sqrt(z)))
pairs=[]
for a,b in zip(valid,valid[1:]):
 dt=b['clock_elapsed_s']-a['clock_elapsed_s'];dist=distance(a,b)
 pairs.append(dict(frame_a=a['frame_index'],frame_b=b['frame_index'],dt_clock_s=dt,distance_m=dist,speed_m_s=dist/dt if dt>0 else None))
keys=list(dict.fromkeys(k for r in rows for k in r))
with (out/'GPS_OVERLAY_OCR_SAMPLES.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
stats=dict(total_samples=len(rows),parsed_samples=len(valid),failed_frames=[r['frame_index'] for r in rows if not r['status'].startswith('parsed')],first=valid[0],last=valid[-1],clock_nonpositive_pairs=sum(p['dt_clock_s']<=0 for p in pairs),clock_backward_pairs=sum(p['dt_clock_s']<0 for p in pairs),max_step_distance_m=max(p['distance_m'] for p in pairs),max_positive_step_speed_m_s=max(p['speed_m_s'] for p in pairs if p['speed_m_s'] is not None),clock_minus_video_s_min_max=[min(r['clock_elapsed_s']-r['video_time_s'] for r in valid),max(r['clock_elapsed_s']-r['video_time_s'] for r in valid)])
(out/'AUDIT.json').write_text(json.dumps(stats,indent=2));(out/'PAIRS.json').write_text(json.dumps(pairs,indent=2))
print(json.dumps(stats,indent=2))
