import csv,json
from pathlib import Path
import numpy as np
import pytest
from railtsv.geometry import axis_profile,TemporalGuard
from railtsv.second_export import export_seconds
from railtsv.telemetry import measure_section,CSV_FIELDS

def rails():
    mask=np.zeros((100,200),np.uint8)
    for y in range(100):
        x=20+y//5;mask[y,x:x+5]=1;mask[y,x+80:x+86]=2
    p=np.eye(3,dtype=np.float32)[mask].transpose(2,0,1)
    return mask,p

def test_axis_is_exact_midpoint_in_curves_and_does_not_bridge_missing_rail():
    mask,p=rails()
    mask[40:56][mask[40:56]==2]=0
    profile=axis_profile(mask,p)
    assert len(profile['segments'])==2
    for segment in profile['segments']:
        for r in segment:assert r['center_x']==(r['left_inner_x']+r['right_inner_x'])/2
    assert not any(40<=r['y']<56 for s in profile['segments'] for r in s)
    mask[:,170:176]=2
    assert axis_profile(mask,p)['points']==0

def test_temporal_guard_rejects_jump_without_altering_measurement_and_reacquires():
    mask,p=rails();s=measure_section(mask,p);guard=TemporalGuard(200)
    assert guard.check(s,0)=='initial'
    jumped=dict(s,left_inner_x=s['left_inner_x']+50,right_inner_x=s['right_inner_x']+50)
    assert guard.check(jumped,.03)=='jump_rejected'
    assert guard.check(jumped,.06)=='jump_rejected'
    assert guard.check(jumped,.09)=='reacquired'
    assert s['center_x_px']==(s['left_inner_x']+s['right_inner_x'])/2

def test_second_export_retains_rejections_excludes_repeats_and_uses_real_axis_snapshot(tmp_path):
    point=dict(y=80,left_inner_x=30,right_inner_x=130,left_outer_x=25,right_outer_x=135,center_x=80,apparent_width_px=100)
    values=[]
    for i,(time,width,status,repeated) in enumerate([(0.,100,'candidate',False),(.4,110,'candidate',False),(.5,999,'repeated',True),(1.1,None,'rejected',False)]):
        row={k:'' for k in CSV_FIELDS}
        row.update(frame_index=i,seconds=time,source_frame_index=i,source_seconds=time,
                   repeated_source_frame=repeated,status=status,reason='missing' if status=='rejected' else '',
                   apparent_width_px=width,center_x_px=80,center_y_px=80,row_y=80,
                   model_score_left=.8,model_score_right=.9,axis_points_json=json.dumps([[point]]) if status=='candidate' else '[]')
        values.append(row)
    with (tmp_path/'mesures.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=CSV_FIELDS);w.writeheader();w.writerows(values)
    report=export_seconds(tmp_path,1.2)
    with (tmp_path/'mesures_par_seconde.csv').open() as f:rows=list(csv.DictReader(f))
    assert report['seconds']==2 and len(rows)==2
    assert float(rows[0]['apparent_width_px_median'])==105
    assert rows[0]['representative_frame_index']=='1'
    assert rows[0]['source_observations']=='2' and rows[0]['repeated_frames']=='1'
    assert rows[1]['apparent_width_px_median']=='' and rows[1]['width_m']==''
    with (tmp_path/'axe_par_seconde.csv').open() as f:points=list(csv.DictReader(f))
    assert len(points)==1 and float(points[0]['center_x_px'])==80
    with pytest.raises(FileExistsError):export_seconds(tmp_path,1.2)
