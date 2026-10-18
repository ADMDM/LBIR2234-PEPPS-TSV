"""Optional causal display smoothing; raw predictions stay immutable."""
from copy import deepcopy
import csv,json,math
from pathlib import Path

class ProfileSmoother:
    def __init__(self,width,tau_seconds=.08,max_gap_seconds=.15,jump_fraction=.04):
        if width<2 or not all(math.isfinite(v) and v>0 for v in [tau_seconds,max_gap_seconds,jump_fraction]):
            raise ValueError('Paramètres de lissage positifs requis')
        self.width=width;self.tau=tau_seconds;self.max_gap=max_gap_seconds
        self.jump=max(12,width*jump_fraction);self.previous={};self.time=None;self.last=None
    def update(self,profile,seconds):
        if not math.isfinite(seconds):raise ValueError('Temps fini requis')
        if self.time is not None and seconds<=self.time:raise ValueError('Observations strictement croissantes requises ; exclure les répétitions')
        result=deepcopy(profile);current={};counts={}
        gap=self.time is not None and seconds-self.time>self.max_gap
        alpha=1-math.exp(-(seconds-self.time)/self.tau) if self.time is not None else 1.
        for segment in result['segments']:
            for p in segment:
                y=p['y'];left=float(p['left_inner_x']);right=float(p['right_inner_x'])
                if y in current or not all(math.isfinite(v) for v in [left,right]) or not 0<=left<right<self.width:
                    raise ValueError('Profil invalide pour le lissage')
                p.update(raw_left_inner_x=left,raw_right_inner_x=right,raw_center_x=(left+right)/2,raw_apparent_width_px=right-left)
                prev=self.previous.get(y)
                state='initial' if self.time is None else 'new_row'
                if gap:state='reset_gap'
                elif prev is not None:
                    if max(abs(left-prev[0]),abs(right-prev[1]))>self.jump:state='reset_jump'
                    else:
                        left=alpha*left+(1-alpha)*prev[0];right=alpha*right+(1-alpha)*prev[1];state='filtered'
                p.update(left_inner_x=left,right_inner_x=right,center_x=(left+right)/2,apparent_width_px=right-left,smoothing_status=state)
                current[y]=(left,right);counts[state]=counts.get(state,0)+1
        result.update(definition='Causal exponential smoothing of paired observed boundaries; centre is their exact midpoint',
            units='original_image_pixels',raw_predictions_modified=False,inferred_hidden_parts=False,
            smoothing=dict(tau_seconds=self.tau,alpha=alpha,states=counts))
        self.previous=current;self.time=seconds;self.last=deepcopy(result)
        return result

def with_measurement_row(profile,section):
    """Add the exact observed section row within an already continuous segment.

    This is a measured raw row, never interpolation across missing portions.
    The raw exported profile remains unchanged.
    """
    result=deepcopy(profile)
    if section['status']!='candidate':return result
    y=section['row_y']
    for segment in result['segments']:
        if segment[0]['y']<=y<=segment[-1]['y']:
            if not any(p['y']==y for p in segment):
                segment.append(dict(y=y,left_inner_x=section['left_inner_x'],right_inner_x=section['right_inner_x'],
                    center_x=section['center_x_px'],apparent_width_px=section['apparent_width_px']))
                segment.sort(key=lambda p:p['y']);result['points']+=1
            break
    return result

SMOOTH_FIELDS=['frame_index','seconds','source_frame_index','source_seconds','repeated_source_frame',
 'raw_section_status','raw_center_x_px','raw_width_px','smoothed_center_x_px','smoothed_width_px',
 'row_y','smoothing_status','axis_points_json','width_m','independent_accuracy_pct']
POINT_FIELDS=['second','frame_index','seconds','source_frame_index','source_seconds','segment','point','y_px',
 'center_x_raw_px','center_x_smoothed_px','left_raw_px','right_raw_px','left_smoothed_px','right_smoothed_px',
 'width_raw_px','width_smoothed_px','smoothing_status','units','width_m','independent_accuracy_pct']

def export_smoothed(folder,duration):
    folder=Path(folder);groups={i:[] for i in range(math.ceil(duration-1e-9))}
    all_count=0;second_count=0
    with (folder/'mesures_lissees.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    observed=[r for r in rows if r['repeated_source_frame'].lower() not in ['true','1']]
    for r in observed:groups[int(float(r['seconds']))].append(r)
    def points(row):
        for segment,values in enumerate(json.loads(row['axis_points_json'])):
            for number,p in enumerate(values):
                yield dict(second=int(float(row['seconds'])),frame_index=row['frame_index'],seconds=row['seconds'],
                  source_frame_index=row['source_frame_index'],source_seconds=row['source_seconds'],segment=segment,point=number,y_px=p['y'],
                  center_x_raw_px=p['raw_center_x'],center_x_smoothed_px=p['center_x'],left_raw_px=p['raw_left_inner_x'],right_raw_px=p['raw_right_inner_x'],
                  left_smoothed_px=p['left_inner_x'],right_smoothed_px=p['right_inner_x'],width_raw_px=p['raw_apparent_width_px'],width_smoothed_px=p['apparent_width_px'],
                  smoothing_status=p['smoothing_status'],units='original_image_pixels',width_m=None,independent_accuracy_pct=None)
    with (folder/'axe_brut_et_lisse_toutes_images.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=POINT_FIELDS);w.writeheader()
        for r in observed:
            for p in points(r):w.writerow(p);all_count+=1
    with (folder/'axe_brut_et_lisse_par_seconde.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=POINT_FIELDS);w.writeheader()
        for second,items in groups.items():
            valid=[r for r in items if r['axis_points_json']!='[]']
            if valid:
                row=min(valid,key=lambda r:abs(float(r['seconds'])-(second+.5)))
                for p in points(row):w.writerow(p);second_count+=1
    return dict(frames_csv='mesures_lissees.csv',points_all='axe_brut_et_lisse_toutes_images.csv',
      points_per_second='axe_brut_et_lisse_par_seconde.csv',all_points=all_count,per_second_points=second_count,
      sampling='One actual observed frame nearest second midpoint; same y for raw and filtered coordinates',
      note='Display stabilisation only, no guarantee of greater accuracy or fixed physical gauge; unsupported sections stay gaps')
