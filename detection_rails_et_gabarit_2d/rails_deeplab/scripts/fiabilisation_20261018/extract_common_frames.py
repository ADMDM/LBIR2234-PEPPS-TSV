"""Extract exact shared inputs: normalize timeline BEFORE selecting frames.

Requires numpy, Pillow, imageio-ffmpeg. No model or training dependencies.
The supplied source SHA256 must match; annotated output videos are not inputs.
"""
import argparse,hashlib,json,subprocess,csv
from pathlib import Path
import numpy as np
from PIL import Image
import imageio_ffmpeg

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def extract(video,spec,out,indices):
    video,out=Path(video),Path(out)
    if digest(video)!=spec['source_sha256']:raise ValueError('Le fichier source ne correspond pas au SHA256 fourni. Utiliser la vidéo originale, pas la vidéo annotée.')
    indices=sorted(set(indices))
    if not indices or min(indices)<0 or max(indices)>=spec['normalized_frames']:raise ValueError('Indices hors vidéo')
    out.mkdir(parents=True,exist_ok=False)
    w,h=spec['dimensions'];fps=spec['fps'];ff=imageio_ffmpeg.get_ffmpeg_exe()
    cmd=[ff,'-hide_banner','-loglevel','error','-nostdin','-i',str(video),'-map','0:v:0','-an','-vf',f'fps=fps={fps:.12g}:start_time=0:round=near','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    wanted=set(indices);rows=[]
    with (out/'decode.log').open('wb') as log:
        process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=log)
        try:
            for i in range(max(indices)+1):
                raw=process.stdout.read(w*h*3)
                if len(raw)!=w*h*3:raise RuntimeError(f'Décodage interrompu à la frame {i}')
                if i in wanted:
                    name=f"{spec['video_id']}_frame_{i:06d}.png";image=Image.fromarray(np.frombuffer(raw,np.uint8).reshape(h,w,3));image.save(out/name)
                    expected=spec.get('shared_rgb_sha256',{}).get(str(i));actual=hashlib.sha256(raw).hexdigest()
                    if expected and expected!=actual:raise ValueError(f'Pixels décodés différents pour {i}. Utiliser les PNG communs fournis ou le même décodeur.')
                    rows.append(dict(video_id=spec['video_id'],frame_index=i,video_time_s=i/fps,image=name,width_px=w,height_px=h,rgb_sha256=actual,png_sha256=digest(out/name)))
        finally:
            process.stdout.close();process.terminate();process.wait(timeout=15)
    with (out/'MANIFESTE_IMAGES.csv').open('x',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    (out/'DECODEUR.json').write_text(json.dumps(dict(command=cmd,ffmpeg_sha256=digest(ff),normalization='fps before frame selection; no crop or resize; RGB24',indices=indices),indent=2))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',required=True);p.add_argument('--spec',required=True);p.add_argument('--out',required=True);p.add_argument('--start',type=int);p.add_argument('--count',type=int);a=p.parse_args()
    spec=json.loads(Path(a.spec).read_text())
    if (a.start is None)!=(a.count is None):p.error('--start et --count se donnent ensemble')
    ids=list(range(a.start,a.start+a.count)) if a.start is not None else spec['shared_frame_indices']
    result=extract(a.video,spec,a.out,ids);print(len(result),'images communes extraites. Comparer les équipes sur video_id + frame_index + y_px.')
