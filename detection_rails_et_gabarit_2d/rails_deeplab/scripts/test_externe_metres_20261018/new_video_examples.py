"""Two untouched institutional clips, frozen detector, supplied metric pose reference."""
import sys,csv,json,zipfile,subprocess,math
from pathlib import Path
import numpy as np,cv2,torch,imageio_ffmpeg
from PIL import Image
sys.path.insert(0,'/tmp/tsv_parquet_runtime');import pyarrow as pa,pyarrow.parquet as pq
R=Path(sys.argv[1]);sys.path.insert(0,str(R))
from railtsv.model import build_model,load_checkpoint
from railtsv.engine import predict_image
from railtsv.geometry import corridor_profile
from railtsv.video import verify_encoded
RUN=R/'runs/test_externe_metres_20261018';DATA=R/'data/external/railbench_locked_20261018';OUT=R/'03_RESULTATS/TESTS_ET_METRES_20261018/04_NOUVELLES_SEQUENCES';OUT.mkdir(exist_ok=False)
p=json.loads((RUN/'PROTOCOLE_FIGE.json').read_text());c=load_checkpoint(R/p['checkpoint']);model=build_model(2);model.load_state_dict(c['state_dict']);dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model.eval().to(dev);torch.set_num_threads(4)
ff=imageio_ffmpeg.get_ffmpeg_exe();summary=[]
for scene in p['video_clips']['scenes']:
 dest=OUT/scene;dest.mkdir()
 with zipfile.ZipFile(DATA/'gt_poses.zip') as z:poses=pq.read_table(pa.BufferReader(z.read(f'gt_poses/train/{scene}.parquet'))).to_pylist()
 with zipfile.ZipFile(DATA/'calibration.zip') as z:cal=json.loads(z.read('calibration/intrinsics_rgb_12mp_middle.json'))
 (dest/'CAMERA_SOURCE.json').write_text(json.dumps(cal,indent=2))
 paths=sorted((DATA/f'scenes/train/{scene}').glob('*.png'),key=lambda p:int(p.name.split('_')[0]));assert len(paths)==50
 lookup={round(q['timestamp'],5):q for q in poses};rows=[];allpoints=[];distance=0.;previous=None;first=None;outw,outh=1280,780
 cmd=[ff,'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s:v',f'{outw}x{outh}','-r','5','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/'DETECTION_RALENTIE_0_5X.mp4')]
 with (dest/'render.log').open('wb') as log:
  enc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
  for i,path in enumerate(paths):
   t=float(path.stem.split('_',1)[1]);q=lookup[round(t,5)];xyz=np.array([q[k] for k in ['t_x','t_y','t_z']]);first=t if first is None else first
   step=float(np.linalg.norm(xyz-previous[1])) if previous else 0.;speed=step/(t-previous[0]) if previous else None;distance+=step;previous=(t,xyz)
   im=np.asarray(Image.open(path).convert('RGB'));mask,prob=predict_image(model,im,dev,c['height'],c['width']);profile=corridor_profile(mask,prob);segments=[[[z['center_x'],z['y']] for z in s] for s in profile['segments']]
   row=dict(video_id=scene,frame_index=i,source_filename=path.name,source_timestamp_s=t,physical_elapsed_s=t-first,preview_playback_s=i/5,preview_speed_factor=.5,reference_pose_x_m=q['t_x'],reference_pose_y_m=q['t_y'],reference_pose_z_m=q['t_z'],reference_path_distance_m=distance,speed_from_reference_pose_m_s=speed,axis_segments_json=json.dumps(segments,separators=(',',':')),axis_points=profile['points'],axis_error_px=None,axis_error_status='no_axis_annotation_for_this_clip',pose_source='RAIL-BENCH supplied gt_poses; not estimated by our detector',width_px=im.shape[1],height_px=im.shape[0]);rows.append(row)
   for j,s in enumerate(profile['segments']):
    for k,z in enumerate(s):allpoints.append(dict(video_id=scene,frame_index=i,source_timestamp_s=t,segment_id=j,point_index=k,x_px=z['center_x'],y_px=z['y'],left_x_px=z['left_inner_x'],right_x_px=z['right_inner_x'],source_width_px=im.shape[1],source_height_px=im.shape[0],axis_error_px=None))
   view=cv2.resize(im,(outw,outh));sx=outw/im.shape[1];sy=outh/im.shape[0]
   for s in profile['segments']:
    for key,color in [('center_x',(55,240,90)),('left_inner_x',(255,195,65)),('right_inner_x',(255,195,65))]:cv2.polylines(view,[np.array([[round(z[key]*sx),round(z['y']*sy)] for z in s],np.int32)],False,color,2,cv2.LINE_AA)
   cv2.rectangle(view,(0,0),(outw,116),(15,30,40),-1)
   texts=[f'{scene} | NOUVELLE CAPTURE | frame {i} | lecture ralentie x0.5',f'Temps physique {t-first:.1f} s | Avancement REFERENCE FOURNIE {distance:.1f} m',f'Vitesse derivee des poses : {speed*3.6:.1f} km/h' if speed is not None else 'Vitesse : premier point, pas de difference temporelle', 'Axe vert = notre prediction ; erreur de cet axe non annotee sur cet extrait']
   for j,txt in enumerate(texts):cv2.putText(view,txt,(10,25+j*27),cv2.FONT_HERSHEY_SIMPLEX,.61,(240,245,245),1,cv2.LINE_AA)
   enc.stdin.write(view.tobytes())
   if i in [0,25,49]:Image.fromarray(view).save(dest/f'apercu_{i:03d}.jpg',quality=90)
   del im,mask,prob
  enc.stdin.close();assert enc.wait(timeout=60)==0
 verify_encoded(dest/'DETECTION_RALENTIE_0_5X.mp4',50,5,[outw,outh])
 for name,data in [('FRAMES_AXE_ET_POSE_REFERENCE.csv',rows),('AXE_POINTS.csv',allpoints)]:
  with (dest/name).open('x',newline='') as f:
   wr=csv.DictWriter(f,fieldnames=list(data[0]) if data else ['video_id','frame_index','x_px','y_px']);wr.writeheader();wr.writerows(data)
 rec=dict(scene=scene,frames=50,physical_duration_s=rows[-1]['physical_elapsed_s'],preview_duration_s=10.,reference_travel_m=distance,frames_without_axis=sum(r['axis_points']==0 for r in rows),no_training=True,metric_pose_from_publisher_not_our_model=True,axis_metric_accuracy_unknown=True);summary.append(rec);print(json.dumps(rec),flush=True)
(OUT/'RESUME.json').write_text(json.dumps(summary,indent=2))
