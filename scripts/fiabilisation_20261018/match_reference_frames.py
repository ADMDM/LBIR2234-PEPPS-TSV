"""Locate existing reviewed stills in the exact normalized video decoding."""
from pathlib import Path
import sys,json,subprocess,csv,hashlib
import numpy as np,cv2,imageio_ffmpeg
R=Path(sys.argv[1]);O=R/'runs/fiabilisation_20261018/reference_video_alignment';O.mkdir(exist_ok=False)
V=R/'data/raw/tsv_videos/03676_Track insp 9_SOS_0123_20170109ok.AVI'
P=R/'03_RESULTATS/FUSION_AXE_ET_VITESSE_20261018/references'
files=sorted(p for p in P.glob('*.png') if not p.stem.endswith('_mask'))
refs=[cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB) for p in files]
small=np.stack([cv2.resize(im,(96,54),interpolation=cv2.INTER_AREA).astype(np.float32) for im in refs])
best=[dict(distance=float('inf')) for _ in files]
ff=imageio_ffmpeg.get_ffmpeg_exe();cmd=[ff,'-hide_banner','-loglevel','error','-nostdin','-i',str(V),'-map','0:v:0','-an','-vf','fps=fps=30:start_time=0:round=near','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
with (O/'decode.log').open('wb') as log:
 p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=log);i=0
 while True:
    raw=p.stdout.read(1920*1080*3)
    if not raw:break
    if len(raw)!=1920*1080*3:raise RuntimeError('Truncated decoded frame')
    frame=np.frombuffer(raw,np.uint8).reshape(1080,1920,3)
    sm=cv2.resize(frame,(96,54),interpolation=cv2.INTER_AREA).astype(np.float32)
    dif=np.mean(np.abs(small-sm),axis=(1,2,3))
    for j,d in enumerate(dif):
        if d<best[j]['distance']:
            exact=np.array_equal(refs[j],frame)
            full=float(np.abs(refs[j].astype(np.int16)-frame.astype(np.int16)).mean())
            best[j]=dict(image_id=files[j].stem,frame_index=i,video_time_s=i/30,distance=float(d),full_rgb_mean_abs_difference=full,exact_decoded_rgb_match=exact,decoded_rgb_sha256=hashlib.sha256(raw).hexdigest(),reference_rgb_sha256=hashlib.sha256(refs[j].tobytes()).hexdigest())
    i+=1
    if i%1000==0:print(i,'frames checked',flush=True)
 if p.wait()!=0:raise RuntimeError('Decoder error')
with (O/'CORRESPONDANCE_REFERENCES.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(best[0]));w.writeheader();w.writerows(best)
(O/'RESULTAT.json').write_text(json.dumps(dict(frames=i,references=best,method='Exhaustive nearest thumbnail search followed by full-resolution decoded RGB equality check; only exact equality accepted automatically.',exact_matches=sum(r['exact_decoded_rgb_match'] for r in best),ffmpeg_command=cmd),indent=2))
print(json.dumps(best,indent=2),flush=True)
