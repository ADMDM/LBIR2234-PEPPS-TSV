"""Extract the native bottom-right GPS overlay once per normalized video second."""
import sys,json,subprocess,hashlib
from pathlib import Path
import imageio_ffmpeg
import numpy as np
from PIL import Image
r=Path(sys.argv[1]);o=Path(sys.argv[2]);o.mkdir(exist_ok=False,parents=True)
source=r/'data/raw/tsv_videos/03676_Track insp 9_SOS_0123_20170109ok.AVI'
ff=imageio_ffmpeg.get_ffmpeg_exe()
# fps first: exactly the same normalized frames as the gabarit CSV.
cmd=[ff,'-hide_banner','-loglevel','error','-nostdin','-i',str(source),'-an','-vf',"fps=fps=30:start_time=0:round=near,select='not(mod(n,30))',crop=350:120:1570:920",'-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
rows=[]
with (o/'decode.log').open('wb') as log:
 p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=log)
 for i in range(180):
  b=p.stdout.read(350*120*3)
  if not b:break
  if len(b)!=350*120*3:raise RuntimeError('partial crop')
  im=Image.fromarray(np.frombuffer(b,np.uint8).reshape(120,350,3))
  name=f'frame_{i*30:06d}.png';im.resize((1050,360),Image.Resampling.LANCZOS).save(o/name)
  rows.append(dict(frame_index=i*30,video_time_s=i,image=name,crop=[1570,920,350,120],native_crop_rgb_sha256=hashlib.sha256(b).hexdigest()))
 p.stdout.close();assert p.wait()==0
(o/'MANIFESTE.json').write_text(json.dumps(rows,indent=2))
print(len(rows),'GPS crops extracted')
