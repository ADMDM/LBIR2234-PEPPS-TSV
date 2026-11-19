"""Side-by-side replay of raw, EMA and adaptive observed corridor geometry."""
import argparse,csv,gzip,json,subprocess,sys
from pathlib import Path
import cv2,numpy as np,imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from railtsv.video import verify_encoded


def render(video_folder,stability_folder,output):
    video_folder,stability_folder,output=map(Path,[video_folder,stability_folder,output])
    if output.exists():raise FileExistsError(output)
    meta=json.loads((video_folder/'video.json').read_text());fps=meta['fps'];w,h=meta['dimensions']
    intervals=[(55.,65.),(135.,145.)]
    wanted={i for start,end in intervals for i in range(round(start*fps),round(end*fps))}
    profiles={}
    with gzip.open(stability_folder/'profils_adaptatifs.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['frame_index'] in wanted:profiles[r['frame_index']]={'adaptive':r['segments']}
    with (video_folder/'mesures_lissees.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            i=int(row['frame_index'])
            if i in profiles:profiles[i]['ema']=json.loads(row['axis_points_json'])
    ff=imageio_ffmpeg.get_ffmpeg_exe();total=0;last=None
    with (output.with_suffix('.log')).open('wb') as log:
        enc=subprocess.Popen([ff,'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1920x440','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE,stderr=log)
        try:
            for start,end in intervals:
                # Same timestamp normalization as the original inference.
                vf=f'fps=fps={fps}:start_time=0:round=near,trim=start_frame={round(start*fps)}:end_frame={round(end*fps)},setpts=PTS-STARTPTS'
                dec=subprocess.Popen([ff,'-hide_banner','-loglevel','error','-nostdin','-i',meta['source'],'-an','-vf',vf,'-fps_mode','passthrough','-f','rawvideo','-pix_fmt','bgr24','pipe:1'],stdout=subprocess.PIPE,stderr=log)
                try:
                    for i in range(round(start*fps),round(end*fps)):
                        data=dec.stdout.read(w*h*3)
                        if len(data)!=w*h*3:raise RuntimeError(f'Décodage incomplet à la frame {i} : {len(data)} octets')
                        frame=np.frombuffer(data,np.uint8).reshape(h,w,3)
                        observed=i in profiles
                        if observed:last=profiles[i]
                        geometry=last if last else {'adaptive':[],'ema':[]}
                        canvas=np.zeros((440,1920,3),np.uint8)
                        for panel,(method,label) in enumerate([('raw','BRUT'),('ema','LISSAGE EXISTANT'),('adaptive','LISSAGE ADAPTATIF')]):
                            image=frame.copy();segments=geometry['ema'] if method=='ema' else geometry['adaptive']
                            for seg in segments:
                                left='raw_left_inner_x' if method=='raw' else 'left_inner_x'
                                right='raw_right_inner_x' if method=='raw' else 'right_inner_x'
                                polygon=np.array([[p[left],p['y']] for p in seg]+[[p[right],p['y']] for p in reversed(seg)],np.int32)
                                overlay=image.copy();cv2.fillPoly(overlay,[polygon],(220,185,30));image=cv2.addWeighted(overlay,.2,image,.8,0)
                                for key in [left,right]:cv2.polylines(image,[np.array([[p[key],p['y']] for p in seg],np.int32)],False,(250,210,50),2)
                                center=np.array([[(p[left]+p[right])/2,p['y']] for p in seg],np.int32)
                                cv2.polylines(image,[center],False,(70,245,70),3)
                            canvas[40:400,panel*640:(panel+1)*640]=cv2.resize(image,(640,360))
                            cv2.putText(canvas,label,(panel*640+15,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
                        text=f'Temps source {i/fps:.2f} s | Memes predictions | Lissage != precision | Metres non calibres'
                        if not observed:text+=' | IMAGE REPETEE'
                        cv2.putText(canvas,text,(12,425),cv2.FONT_HERSHEY_SIMPLEX,.6,(245,245,245),1)
                        enc.stdin.write(canvas.tobytes());total+=1
                        if total==90:cv2.imwrite(str(output.with_suffix('.jpg')),canvas)
                    print('Extrait',start,end,'terminé',flush=True)
                finally:
                    dec.stdout.close()
                    if dec.poll() is None:dec.terminate()
                    dec.wait(timeout=15)
            enc.stdin.close()
            if enc.wait(timeout=60):raise RuntimeError('Encodage échoué')
        finally:
            if enc.poll() is None:
                enc.stdin.close();enc.terminate()
                try:enc.wait(timeout=15)
                except subprocess.TimeoutExpired:enc.kill();enc.wait(timeout=5)
    verify_encoded(output,total,fps,[1920,440])
    output.with_suffix('.json').write_text(json.dumps(dict(intervals_source_seconds=intervals,frames=total,fps=fps,
        source=meta['source'],geometry='Same model raw predictions, different causal display filters',
        missing_geometry='No new measurements on repeats or absent segments',metric_calibration=None),indent=2))
    print(output,total,'frames verified')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video-folder',required=True);p.add_argument('--stability-folder',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    render(a.video_folder,a.stability_folder,a.output)
