"""Measure the actual frozen TSV pipeline on this Mac, with synchronized MPS timings."""
import csv,json,sys,time,subprocess,platform,statistics
from pathlib import Path
import cv2,numpy as np,torch,imageio_ffmpeg
ROOT=Path(sys.argv[1]);OUT=Path(sys.argv[2]);sys.path.insert(0,str(ROOT));OUT.mkdir(parents=True,exist_ok=False)
from railtsv.model import build_model,load_checkpoint
from railtsv.engine import predict_image
from railtsv.inference_region import predict_region
from railtsv.geometry import corridor_profile,draw_axis,draw_corridor_limits
from railtsv.adaptive_smoothing import AdaptiveProfileSmoother
from railtsv.common import sha256
CK=ROOT/'runs/experience_egopath_20261016_141426/modele/best.pt'
VIDEO=ROOT/'data/raw/drive_20261014/tsv_record_video.mp4'
GPS=ROOT/'03_RESULTATS/FUSION_AXE_ET_VITESSE_20261018/vitesse_GPS_TSV_RECORD/VITESSE_TRAJET_COMPLET_7608_FRAMES.csv'
gps=list(csv.DictReader(GPS.open()));torch.set_num_threads(4)
dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
start=time.perf_counter();ck=load_checkpoint(CK);model=build_model(2);model.load_state_dict(ck['state_dict']);model.to(dev).eval()
def sync():
    if dev.type=='mps':torch.mps.synchronize()
sync();load_seconds=time.perf_counter()-start
cap=cv2.VideoCapture(str(VIDEO));ok,bgr=cap.read();assert ok;cap.release()
rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
for _ in range(10):predict_region(model,rgb,dev,480,640,[370,0,910,720],predict_image)
sync();all_rows=[];trials=[]
for trial in range(3):
    # Same 60 consecutive source observations for repeatable system timing. No accuracy claim.
    cap=cv2.VideoCapture(str(VIDEO));smoother=AdaptiveProfileSmoother(1280,max_gap_seconds=5)
    video=OUT/f'trial_{trial+1}.mp4';log=(OUT/f'trial_{trial+1}.log').open('wb')
    encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-nostdin',
      '-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','5','-i','pipe:0','-an','-c:v','libx264',
      '-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE,stderr=log)
    csvfile=(OUT/f'trial_{trial+1}_points.csv').open('x',newline='');writer=csv.writer(csvfile);writer.writerow(['frame','y','left','right','axis'])
    wall=time.perf_counter();timings=[]
    for i in range(60):
        t=time.perf_counter();ok,bgr=cap.read();assert ok
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);decoded=time.perf_counter()
        sync();infer_start=time.perf_counter()
        mask,prob=predict_region(model,rgb,dev,480,640,[370,0,910,720],predict_image)
        sync();inferred=time.perf_counter()
        p=corridor_profile(mask,prob);p=smoother.update(p,float(gps[i]['real_elapsed_s']))
        for s in p['segments']:
            for q in s:writer.writerow([i,q['y'],q['left_inner_x'],q['right_inner_x'],q['center_x']])
        view=draw_axis(draw_corridor_limits(rgb.copy(),p),p);post=time.perf_counter()
        encoder.stdin.write(view.tobytes());end=time.perf_counter()
        row=dict(trial=trial+1,frame=i,decode_ms=(decoded-t)*1000,inference_ms=(inferred-infer_start)*1000,
          geometry_csv_overlay_ms=(post-inferred)*1000,encode_queue_ms=(end-post)*1000,pipeline_submission_ms=(end-t)*1000)
        all_rows.append(row);timings.append(row)
    csvfile.close();cap.release();encoder.stdin.close();assert encoder.wait()==0;log.close();elapsed=time.perf_counter()-wall
    verify=cv2.VideoCapture(str(video));assert int(verify.get(cv2.CAP_PROP_FRAME_COUNT))==60;verify.release()
    trials.append(dict(trial=trial+1,frames=60,total_wall_seconds=elapsed,end_to_end_fps=60/elapsed,
        median_submission_ms=float(np.median([x['pipeline_submission_ms'] for x in timings])),
        p95_submission_ms=float(np.percentile([x['pipeline_submission_ms'] for x in timings],95))))
    print('BENCH',trials[-1],flush=True)
with (OUT/'LATENCES_PAR_FRAME.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
machine=subprocess.run(['sysctl','-n','machdep.cpu.brand_string'],capture_output=True,text=True).stdout.strip()
result=dict(device=str(dev),hardware=machine,platform=platform.platform(),torch=torch.__version__,source_sha256=sha256(VIDEO),
 checkpoint_sha256=sha256(CK),network_input=[640,480],source_dimensions=[1280,720],roi=[370,0,910,720],
 model_loading_seconds=load_seconds,warmup_frames=10,trials=trials,source_fps=5,
 scope='Decode, preprocessing, inference, original-coordinate restoration, corridor profile, physical-clock smoothing, CSV, overlay, H264 submission and encoder drain. No camera capture, transport, GUI display or instrument latency.',
 latency_definition='Per-frame synchronized pipeline submission latency, not camera-to-display latency; total throughput includes encoder drain.',
 conclusion='Offline throughput benchmark; not certification of hard real-time or camera-to-display latency.',
 references=['https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html'])
(OUT/'RESULTATS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('BENCHMARK_COMPLETE',flush=True)
