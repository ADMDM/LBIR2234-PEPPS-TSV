from copy import deepcopy
import math,csv,json
import pytest
from railtsv.smoothing import ProfileSmoother,export_smoothed,SMOOTH_FIELDS

def profile(left=20,right=100,ys=(16,24,32)):
 return dict(segments=[[dict(y=y,left_inner_x=left+y/8,right_inner_x=right+y/8,center_x=(left+right)/2+y/8,apparent_width_px=right-left) for y in ys]],points=len(ys))

def test_curve_pair_midpoint_and_original_remain_intact():
 s=ProfileSmoother(200);first=profile();original=deepcopy(first);assert s.update(first,0)['segments']==s.last['segments'];assert first==original
 raw=profile(24,108);copy=deepcopy(raw);filtered=s.update(raw,1/30)
 assert raw==copy
 for p in filtered['segments'][0]:
  assert p['left_inner_x']<p['raw_left_inner_x'] and p['right_inner_x']<p['raw_right_inner_x']
  assert p['center_x']==(p['left_inner_x']+p['right_inner_x'])/2
  assert p['apparent_width_px']==p['right_inner_x']-p['left_inner_x']
 assert filtered['segments'][0][1]['center_x']-filtered['segments'][0][0]['center_x']==pytest.approx(1)

def test_gaps_missing_rows_and_large_changes_do_not_create_hidden_axis():
 s=ProfileSmoother(200);s.update(profile(),0)
 p=s.update(profile(24,104,ys=(16,32)),.03)
 assert [p['y'] for p in p['segments'][0]]==[16,32]
 p=s.update(profile(26,106),.06)
 assert p['segments'][0][1]['smoothing_status']=='new_row'
 s.update(dict(segments=[],points=0),.09)
 p=s.update(profile(27,107),.12)
 assert all(x['smoothing_status']=='new_row' for x in p['segments'][0])
 p=s.update(profile(70,150),.15)
 assert all(x['smoothing_status']=='reset_jump' and x['left_inner_x']==x['raw_left_inner_x'] for x in p['segments'][0])
 p=s.update(profile(71,151),.5)
 assert all(x['smoothing_status']=='reset_gap' for x in p['segments'][0])
 with pytest.raises(ValueError):s.update(profile(),.5)

def test_elapsed_time_filtering_is_independent_of_frame_rate():
 a=ProfileSmoother(200);b=ProfileSmoother(200);a.update(profile(),0);b.update(profile(),0)
 target=profile(24,104)
 pa=a.update(target,.12)
 for t in [.03,.06,.09,.12]:pb=b.update(target,t)
 for x,y in zip(pa['segments'][0],pb['segments'][0]):assert x['center_x']==pytest.approx(y['center_x'])

def test_csv_export_excludes_repeats_and_retains_raw_and_smoothed(tmp_path):
 s=ProfileSmoother(200);values=[]
 for i,t in enumerate([0.,.1]):
  p=s.update(profile(20+i*3,100+i*3),t)
  values.append(dict(frame_index=i,seconds=t,source_frame_index=i,source_seconds=t,repeated_source_frame=False,axis_points_json=json.dumps(p['segments'])))
 values.append(dict(values[-1],frame_index=2,seconds=.2,repeated_source_frame=True,axis_points_json='[]'))
 with (tmp_path/'mesures_lissees.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=SMOOTH_FIELDS);w.writeheader();w.writerows(values)
 result=export_smoothed(tmp_path,.3)
 assert result['all_points']==6 and result['per_second_points']==3
 with (tmp_path/result['points_per_second']).open() as f:rows=list(csv.DictReader(f))
 assert all(r['frame_index']=='1' and r['width_m']=='' for r in rows)
 assert float(rows[0]['center_x_raw_px'])>float(rows[0]['center_x_smoothed_px'])

@pytest.mark.parametrize('method',['ema','adaptive'])
def test_video_smoothing_preserves_the_raw_csv(tmp_path,monkeypatch,method):
 import cv2,numpy as np,torch
 from types import SimpleNamespace
 from railtsv import video
 source=tmp_path/'input.avi';writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),10,(96,64))
 assert writer.isOpened()
 for i in range(10):writer.write(np.full((64,96,3),60+i*10,np.uint8))
 writer.release();checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'test')
 monkeypatch.setattr(video,'load_checkpoint',lambda _:dict(task='corridor',classes=['background','couloir'],state_dict={},height=64,width=96,synthetic_demo=True))
 monkeypatch.setattr(video,'build_model',lambda _:torch.nn.Identity())
 def known_prediction(model,rgb,dev,height,width):
  dx=4 if rgb.mean()>100 else 0;mask=np.zeros((64,96),np.uint8);mask[:,20+dx:76+dx]=1
  return mask,np.eye(2,dtype=np.float32)[mask].transpose(2,0,1)
 monkeypatch.setattr(video,'predict_image',known_prediction)
 rawargs=SimpleNamespace(input=str(source),checkpoint=str(checkpoint),out=str(tmp_path/'raw'),max_frames=0,cpu_threads=1,device='cpu')
 video.infer_video(rawargs)
 smoothargs=SimpleNamespace(**vars(rawargs));smoothargs.out=str(tmp_path/'smooth');smoothargs.smooth_display=True
 smoothargs.smoothing_method=method
 result=video.infer_video(smoothargs)
 assert (tmp_path/'raw/mesures.csv').read_bytes()==(tmp_path/'smooth/mesures.csv').read_bytes()
 assert result['display_smoothing']['enabled'] and result['frames']==10
 with (tmp_path/'smooth/mesures_lissees.csv').open() as f:rows=list(csv.DictReader(f))
 assert any(r['smoothed_center_x_px'] and abs(float(r['raw_center_x_px'])-float(r['smoothed_center_x_px']))>.01 for r in rows)
 assert (tmp_path/'smooth/axe_brut_et_lisse_par_seconde.csv').exists()
