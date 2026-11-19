"""Render the selected RAW CSV coordinates on both original videos; no model."""
import sys,json,csv,subprocess,time,itertools
from pathlib import Path
import cv2,numpy as np,imageio_ffmpeg
R=Path(sys.argv[1]);O=R/'03_RESULTATS/GABARIT_2_VIDEOS_V2_20261018';ff=imageio_ffmpeg.get_ffmpeg_exe()
sys.path.insert(0,str(R));from railtsv.video import verify_encoded
def rows(p):
    with p.open(newline='') as f:yield from csv.DictReader(f)
result=[]
for folder,source in [('01_TSV_RECORD',R/'data/raw/drive_20261014/tsv_record_video.mp4'),('02_TSV_03676_180S',R/'data/raw/tsv_videos/03676_Track insp 9_SOS_0123_20170109ok.AVI')]:
    if len(sys.argv)>2 and folder!=sys.argv[2]:
        result.append(json.loads((O/folder/'RENDU_VERIFIE.json').read_text()));continue
    D=O/folder;meta=json.loads((D/'SOURCE_ET_DECOUPAGE.json').read_text());w,h=meta['dimensions'];fps=meta['fps'];dest=D/'VIDEO_ANNOTEE.mp4';partial=D/'VIDEO_ANNOTEE.partial.mp4'
    if dest.exists() or partial.exists():raise FileExistsError(dest)
    decoder_cmd=[ff,'-hide_banner','-loglevel','error','-nostdin','-i',str(source),'-map','0:v:0','-an','-vf',meta['extraction_filter'],'-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    encoder_cmd=[ff,'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s:v',f'{w}x{h}','-r',f'{fps:g}','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart','-threads','2',str(partial)]
    by_frame=itertools.groupby(rows(D/'AXE_MEDIAN.csv'),key=lambda r:int(r['frame_index']));begin=time.monotonic()
    with (D/'RENDU.log').open('wb') as log:
        dec=subprocess.Popen(decoder_cmd,stdout=subprocess.PIPE,stderr=log);enc=subprocess.Popen(encoder_cmd,stdin=subprocess.PIPE,stderr=log)
        n=0
        try:
            for frame,(idx,points) in zip(rows(D/'FRAMES_VITESSE_VECTEUR.csv'),by_frame):
                assert int(frame['frame_index'])==idx==n
                raw=dec.stdout.read(w*h*3)
                if len(raw)!=w*h*3:raise RuntimeError('Truncated source')
                rgb=np.frombuffer(raw,np.uint8).reshape(h,w,3).copy();segments={}
                for p in points:
                    if p['x_px']:segments.setdefault(int(p['segment_id']),[]).append(p)
                for segment in segments.values():
                    for key,color,thickness in [('left_x_px',(255,195,65),1),('right_x_px',(255,195,65),1),('x_px',(45,240,100),2)]:
                        xy=np.array([[round(float(p[key])),round(float(p['y_px']))] for p in segment],np.int32)
                        cv2.polylines(rgb,[xy],False,color,thickness,cv2.LINE_AA)
                scale=.65 if w<1500 else .85;line=round(30*scale/.65)
                cv2.rectangle(rgb,(0,0),(w,3*line+7),(15,27,35),-1)
                speed=f"{float(frame['train_speed_km_h']):.1f} km/h" if frame['train_speed_km_h'] else 'inconnue (aucun releve pour cette frame)'
                if frame['speed_status'].startswith('estimated_'):speed+=' ESTIMEE (moyenne GPS ~10 s, non validee)'
                physical=f" | t reel {float(frame['real_elapsed_s']):.2f} s" if frame['real_elapsed_s'] else ''
                error=f"MAE reference : {float(frame['axis_mae_px']):.2f} px (developpement)" if frame['axis_mae_px'] else 'Erreur reelle non mesuree sur cette frame'
                text=[f"{meta['video_id']} | frame {idx} | t video {float(frame['video_time_s']):.3f} s{physical}",f"AXE BRUT RETENU POUR LE CSV | Vitesse : {speed}",f"{frame['axis_status']} | {frame['observation_status']} | {error}"]
                for j,t in enumerate(text):cv2.putText(rgb,t,(12,(j+1)*line),cv2.FONT_HERSHEY_SIMPLEX,scale,(235,242,244),1,cv2.LINE_AA)
                enc.stdin.write(rgb.tobytes());n+=1
                if n%1000==0:print(meta['video_id'],n,'/',meta['normalized_frames'],'rendered',flush=True)
            assert n==meta['normalized_frames']
            if dec.stdout.read(1) or dec.wait(timeout=60)!=0:raise RuntimeError('Decoder did not finish exactly')
            enc.stdin.close()
            if enc.wait(timeout=60)!=0:raise RuntimeError('Encoder failed')
        finally:
            for p in [dec,enc]:
                if p.poll() is None:p.terminate()
        verify_encoded(partial,n,fps,[w,h]);partial.rename(dest)
    record=dict(video_id=meta['video_id'],frames=n,fps=fps,dimensions=[w,h],variant='selected_raw',source_images='same original RGB decode as shared PNG extractor',overlay_coordinates='AXE_MEDIAN.csv, same video/frame/segment keys',render_wall_seconds=time.monotonic()-begin,no_new_inference=True)
    (D/'RENDU_VERIFIE.json').write_text(json.dumps(record,indent=2));result.append(record)
    print(json.dumps(record),flush=True)
(O/'RENDU_DEUX_VIDEOS.json').write_text(json.dumps(result,indent=2))
