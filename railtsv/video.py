"""Predict every video frame and encode a playable MP4 without frame sampling."""
import csv
from contextlib import ExitStack
import json
import math
import re
from pathlib import Path
import subprocess
import time

import cv2
import imageio_ffmpeg
import numpy as np
import torch

from .common import device, dump, new_dir, sha256
from .engine import predict_image
from .model import build_model, load_checkpoint
from .telemetry import CSV_FIELDS, draw_telemetry, measure_section, measure_corridor
from .geometry import axis_profile, corridor_profile, draw_corridor_limits, draw_axis, TemporalGuard
from .second_export import export_seconds
from .smoothing import ProfileSmoother,SMOOTH_FIELDS,export_smoothed,with_measurement_row
from .adaptive_smoothing import AdaptiveProfileSmoother
from .inference_region import predict_region
from .frame_clock import load_frame_telemetry, TELEMETRY_FIELDS


def overlay_frame(rgb, mask, label, task='tsv'):
    colors = np.array([[0, 0, 0], [0, 220, 255], [255, 150, 0]], np.uint8)
    result = rgb.copy()
    selected = mask > 0
    alpha = .18 if task == 'corridor' else .45
    result[selected] = ((1-alpha) * rgb[selected] + alpha * colors[mask[selected]]).astype(np.uint8)
    cv2.rectangle(result, (0, 0), (result.shape[1] - 1, 28), (0, 0, 0), -1)
    cv2.putText(result, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def probe_timing(source, fps):
    """Count decoded images and normalized frames using presentation timestamps."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-i", str(source),
        "-map", "0:v:0", "-an", "-vf", f"showinfo,fps=fps={fps:.12g}:start_time=0:round=near",
        "-progress", "pipe:1", "-f", "null", "-"], capture_output=True, text=True)
    timestamps = [float(x) for x in re.findall(r"\bn:\s*\d+\s+pts:\s*\S+\s+pts_time:([\d.eE+-]+)", result.stderr)]
    counts = [int(x) for x in re.findall(r"^frame=(\d+)$", result.stdout, flags=re.MULTILINE)]
    if result.returncode or not timestamps or not counts or counts[-1] < 1:
        raise ValueError("Impossible de décoder la chronologie vidéo : " + result.stderr[-1500:])
    if not np.isfinite(timestamps).all() or any(b <= a for a,b in zip(timestamps,timestamps[1:])):
        raise ValueError("Horodatages vidéo non monotones")
    return dict(decoded_frames=len(timestamps), normalized_frames=counts[-1], timestamps_seconds=timestamps,
                timestamp_gaps=sum(b-a > 1.5/fps for a,b in zip(timestamps,timestamps[1:])), fps=fps)


def timeline_indices(timing):
    """Hold the previous known image through a gap; never invent a measurement."""
    slots = np.rint(np.asarray(timing['timestamps_seconds']) * timing['fps']).astype(np.int64)
    return np.clip(np.searchsorted(slots, np.arange(timing['normalized_frames']), side='right')-1,
                   0, timing['decoded_frames']-1)


def encoding_command(ffmpeg, source, target, width, height, fps, keep_audio=False):
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
            "-r", format(fps, ".12g"), "-i", "pipe:0"]
    if keep_audio:
        command += ["-i", str(source)]
    command += ["-map", "0:v:0"]
    command += ["-map", "1:a?", "-c:a", "aac"] if keep_audio else ["-an"]
    return command + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                      "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]


def verify_encoded(path, count, fps, dimensions):
    capture = cv2.VideoCapture(str(path))
    try:
        result_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        result_fps = float(capture.get(cv2.CAP_PROP_FPS))
        result_size = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        first_ok, _ = capture.read()
        capture.set(cv2.CAP_PROP_POS_FRAMES, count-1)
        last_ok, _ = capture.read()
    finally:
        capture.release()
    if not first_ok or not last_ok or result_frames != count or result_size != dimensions or abs(result_fps-fps) > .02:
        raise RuntimeError("Le contrôle de lecture du MP4 a échoué")


def recover_video(source, annotated, output, keep_audio=False):
    """Recover already inferred coded images using the original presentation times."""
    source, annotated = Path(source).resolve(), Path(annotated).resolve()
    capture = cv2.VideoCapture(str(annotated))
    if not capture.isOpened():
        capture.release()
        raise ValueError('La vidéo calculée est illisible')
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    w,h = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        timing = probe_timing(source, fps)
        if timing['decoded_frames'] != count:
            raise ValueError('La vidéo calculée ne contient pas une image par image source décodée')
        mapping = timeline_indices(timing)
        if mapping[-1] != count-1:
            raise ValueError('La dernière image source manque dans la chronologie récupérée')
        out = new_dir(output)
        dump(out/'timing.json', timing)
        target = out/'rails_annotes.mp4'
        partial = out/'rails_annotes.en_cours.mp4'
        process = None
        start = time.perf_counter()
        current, rgb, last_progress = -1, None, 0.
        try:
            with (out/'encodage.log').open('wb') as log:
                process = subprocess.Popen(encoding_command(imageio_ffmpeg.get_ffmpeg_exe(),source,partial,w,h,fps,keep_audio),
                    stdin=subprocess.PIPE,stderr=log,stdout=subprocess.DEVNULL)
                for i,k in enumerate(mapping):
                    while current < k:
                        ok,bgr = capture.read()
                        if not ok:
                            raise ValueError('Une image de la vidéo calculée ne se décode pas')
                        rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                        current += 1
                    process.stdin.write(rgb.tobytes())
                    now=time.perf_counter()
                    if i==0 or now-last_progress >= 10:
                        print(f'Récupération vidéo : {i+1} / {len(mapping)} images',flush=True)
                        last_progress=now
                process.stdin.close()
                if process.wait(timeout=60):
                    raise RuntimeError('Échec de récupération ; consulter encodage.log')
            verify_encoded(partial,len(mapping),fps,[w,h])
            partial.rename(target)
            report=dict(video=target.name,source=str(source),source_sha256=sha256(source),
                recovered_from=str(annotated),recovered_from_sha256=sha256(annotated),
                source_decoded_frames=count,frames=len(mapping),fps=fps,duration_seconds=len(mapping)/fps,
                dimensions=[w,h],timestamp_gaps=timing['timestamp_gaps'],
                repeated_frames=int((np.diff(mapping)==0).sum()),source_frames_used=len(np.unique(mapping)),
                audio=('Original source audio retained when present; AAC' if keep_audio else 'Disabled'),complete_source_video=True,
                predictions_recomputed=False,technical_test_only=True,
                timing_note='Repeated frames fill missing capture intervals; they are not new observations',
                seconds=time.perf_counter()-start)
            dump(out/'video.json',report)
            print(f'Vidéo récupérée : {target}',flush=True)
            return report
        finally:
            if process is not None and process.poll() is None:
                try:
                    if not process.stdin.closed:process.stdin.close()
                except OSError:
                    pass
                process.terminate();process.wait(timeout=10)
    finally:
        capture.release()


def infer_video(args):
    if args.max_frames < 0:
        raise ValueError("max-frames doit être >= 0 ; 0 traite toute la vidéo")
    row_fraction = getattr(args, 'measurement_row', .8)
    min_score = getattr(args, 'min_model_score', .55)
    if not 0 < row_fraction < 1 or not 0 <= min_score <= 1:
        raise ValueError('Ligne dans ]0,1[ et score minimal dans [0,1] requis')
    source = Path(args.input).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    torch.set_num_threads(args.cpu_threads)
    dev = device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    task = checkpoint.get('task')
    classes = {'tsv': ['background', 'rail_left', 'rail_right'],
               'corridor': ['background', 'couloir']}
    if task not in classes or checkpoint['classes'] != classes[task]:
        raise ValueError('Un modèle TSV (deux rails) ou corridor (background/couloir) est requis')
    model = build_model(len(classes[task]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(dev).eval()
    label = ("TEST TECHNIQUE - MODELE SYNTHETIQUE" if checkpoint["synthetic_demo"] else
             "ESSAI - MODELE EXPERIMENTAL" if checkpoint.get("training_only_demo", False) else
             "PREDICTION - A VALIDER")
    if task == 'corridor':
        label = 'COULOIR - '+label
    smooth_display = getattr(args, "smooth_display", False)
    if smooth_display and task != "corridor":
        raise ValueError("Le lissage d’affichage nécessite un modèle couloir")
    if smooth_display:
        label += " | AXE LISSE"
    height, width = checkpoint["height"], checkpoint["width"]
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Vidéo illisible : {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or min(w, h) < 2 or w % 2 or h % 2:
        capture.release()
        raise ValueError("Une cadence valide et des dimensions paires sont requises pour le MP4 H.264")
    capture.release()
    roi = getattr(args, 'roi', None)
    if roi is not None and (len(roi)!=4 or not 0<=roi[0]<roi[2]<=w or not 0<=roi[1]<roi[3]<=h):
        raise ValueError('ROI invalide pour les dimensions de cette vidéo')
    print("Vérification des images et des horodatages de la vidéo…", flush=True)
    timing = probe_timing(source, fps)
    mapping = timeline_indices(timing)
    telemetry_path = getattr(args, 'frame_telemetry', None)
    physical_clock = (load_frame_telemetry(telemetry_path, sha256(source), timing['timestamps_seconds'])
                      if telemetry_path else None)
    expected = min(timing['normalized_frames'], args.max_frames) if args.max_frames else timing['normalized_frames']
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out = new_dir(args.out)
    partial = out / "rails_annotes.en_cours.mp4"
    target = out / "rails_annotes.mp4"
    dump(out / 'timing.json', timing)
    keep_audio = getattr(args, 'keep_audio', False)
    command = encoding_command(ffmpeg, source, partial, w, h, fps, keep_audio)
    count, start, last_progress = 0, time.perf_counter(), 0.
    process = None
    reader = None
    preview = None
    candidates, rejected, repeated_count = 0, 0, 0
    guard=TemporalGuard(w)
    smoothing_method = getattr(args, 'smoothing_method', 'ema')
    if smoothing_method not in ('ema', 'adaptive'):
        raise ValueError('Méthode de lissage inconnue')
    smoother = None
    if smooth_display:
        # A 5 fps capture has 0.2 s between valid observations: that is not a gap.
        smoothing_gap = physical_clock['reset_gap_s'] if physical_clock else max(.15, 2.5/fps)
        smoother = (AdaptiveProfileSmoother(w, getattr(args, 'adaptive_min_cutoff', .8),
                                           getattr(args, 'adaptive_beta', 15.), max_gap_seconds=smoothing_gap)
                    if smoothing_method == 'adaptive' else
                    ProfileSmoother(w, getattr(args, 'smoothing_tau', .08), max_gap_seconds=smoothing_gap))
    if getattr(args,'preview',False):
        from .live_preview import LivePreview
        preview=LivePreview()
        print('Aperçu du traitement : '+preview.url,flush=True)
        dump(out/'apercu_direct.json',dict(url=preview.url,note='Local processing preview; throughput is not guaranteed real-time'))
        if __import__('sys').platform=='darwin':subprocess.run(['open',preview.url],check=False)
    last_preview=0.
    try:
        with (out / "encodage.log").open("wb") as log, (out / "decodage.log").open("wb") as decoder_log, \
                (out / 'mesures.csv').open('x', newline='', encoding='utf-8') as csv_file, ExitStack() as extra:
            measures = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS + (TELEMETRY_FIELDS if physical_clock else []))
            measures.writeheader()
            smooth_writer=None
            if smoother:
                smooth_file=extra.enter_context((out/"mesures_lissees.csv").open("x",newline="",encoding="utf-8"))
                smooth_writer=csv.DictWriter(smooth_file,fieldnames=SMOOTH_FIELDS + (TELEMETRY_FIELDS if physical_clock else []));smooth_writer.writeheader()
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log, stdout=subprocess.DEVNULL)
            reader = subprocess.Popen([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-i', str(source),
                '-map', '0:v:0', '-an', '-vf', f'fps=fps={fps:.12g}:start_time=0:round=near',
                '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                stdout=subprocess.PIPE, stderr=decoder_log)
            while count < expected:
                raw = reader.stdout.read(w*h*3)
                if not raw:
                    break
                if len(raw) != w*h*3:
                    raise ValueError('Image vidéo incomplète pendant le décodage')
                rgb = np.frombuffer(raw, np.uint8).reshape(h,w,3)
                source_index = int(mapping[count])
                repeated = count > 0 and source_index == int(mapping[count-1])
                if not repeated:
                    if roi is None:
                        mask, probabilities = predict_image(model, rgb, dev, height, width)
                    else:
                        mask, probabilities = predict_region(model,rgb,dev,height,width,roi,predict_image)
                section = (measure_corridor if task == 'corridor' else measure_section)(
                    mask, probabilities, row_fraction, min_score)
                profile = (corridor_profile if task == 'corridor' else axis_profile)(mask, probabilities, min_score)
                clock_fields = physical_clock['samples'][source_index] if physical_clock else {}
                filter_time = (clock_fields['physical_seconds'] if physical_clock else timing['timestamps_seconds'][source_index])
                if physical_clock and not repeated and source_index > 0 and filter_time-physical_clock['samples'][source_index-1]['physical_seconds'] > physical_clock['reset_gap_s']:
                    guard = TemporalGuard(w)
                temporal='repeated' if repeated else guard.check(section,filter_time if physical_clock else count/fps)
                if temporal=='jump_rejected':
                    section.update(status='rejected',reason='temporal_jump_not_validated',
                        left_inner_x=None,right_inner_x=None,apparent_width_px=None,
                        center_x_px=None,center_y_px=None,left_outer_x=None,right_outer_x=None,
                        rail_left_thickness_px=None,rail_right_thickness_px=None)
                    profile['segments']=[];profile['points']=0
                if repeated:
                    repeated_count += 1
                    section.update(status='repeated',reason='no_new_source_observation',
                                   left_inner_x=None,right_inner_x=None,apparent_width_px=None,
                                   center_x_px=None,center_y_px=None,left_outer_x=None,right_outer_x=None,
                                   rail_left_thickness_px=None,rail_right_thickness_px=None)
                elif section['status']=='candidate':
                    candidates += 1
                else:
                    rejected += 1
                measures.writerow(dict(frame_index=count, seconds=count/fps, source_frame_index=source_index,
                                       source_seconds=timing['timestamps_seconds'][source_index],
                                       repeated_source_frame=repeated,temporal_status=temporal,
                                       axis_points_json=json.dumps(profile['segments'] if not repeated else [],separators=(',',':')),**section,**clock_fields))
                display_profile=profile
                display_section=dict(section)
                if smoother:
                    display_profile=(smoother.last or profile) if repeated else smoother.update(with_measurement_row(profile,section),filter_time)
                    point=next((p for seg in display_profile['segments'] for p in seg if p['y']==section['row_y']),None)
                    smooth_writer.writerow(dict(frame_index=count,seconds=count/fps,source_frame_index=source_index,
                        source_seconds=timing['timestamps_seconds'][source_index],repeated_source_frame=repeated,
                        raw_section_status=section['status'],raw_center_x_px=section['center_x_px'],raw_width_px=section['apparent_width_px'],
                        smoothed_center_x_px=point['center_x'] if point and not repeated else None,
                        smoothed_width_px=point['apparent_width_px'] if point and not repeated else None,
                        row_y=section['row_y'],smoothing_status='repeated' if repeated else (point['smoothing_status'] if point else 'no_observed_point'),
                        axis_points_json=json.dumps(display_profile['segments'] if not repeated else [],separators=(',',':')),
                        width_m=None,independent_accuracy_pct=None,**clock_fields))
                    if section['status']=='candidate':
                        display_section.update(left_inner_x=point['left_inner_x'] if point else None,
                            right_inner_x=point['right_inner_x'] if point else None,
                            center_x_px=point['center_x'] if point else None,
                            apparent_width_px=point['apparent_width_px'] if point else None)
                view = overlay_frame(rgb, mask, label, task)
                if task == 'corridor':
                    view = draw_corridor_limits(view, display_profile)
                frame_info = None
                if physical_clock:
                    speed = clock_fields['train_speed_km_h']
                    text_speed = f'{speed:.1f} km/h' if speed is not None else 'indisponible'
                    frame_info = f'Temps reel : {filter_time:.1f} s | Vitesse fournie : {text_speed}'
                view = draw_telemetry(draw_axis(view, display_profile), display_section,
                    count/fps, repeated, smoothed=bool(smoother), frame_info=frame_info)
                if count == 0:
                    if not cv2.imwrite(str(out/'apercu.png'), cv2.cvtColor(view, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError("Impossible d'enregistrer l'aperçu vidéo")
                process.stdin.write(view.tobytes())
                count += 1
                now = time.perf_counter()
                if preview and (count==1 or now-last_preview>=1):
                    preview.update(view,images=count,total=expected,video_seconds=count/fps,
                                   processing_fps=round(count/(now-start),2),source_fps=fps,
                                   state=section['status'],axis_points=profile['points'])
                    last_preview=now
                if count == 1 or now - last_progress >= 10:
                    suffix = f" / {expected}" if expected else ""
                    print(f"Vidéo : {count}{suffix} images traitées — {count/fps:.1f} s de vidéo", flush=True)
                    last_progress = now
            if args.max_frames and expected < timing['normalized_frames']:
                reader.stdout.close()
                reader.terminate()
                reader.wait(timeout=10)
            else:
                if reader.stdout.read(1) or reader.wait(timeout=60) != 0:
                    raise RuntimeError('Décodage FFmpeg incomplet ; consulter decodage.log')
            process.stdin.close()
            if process.wait(timeout=60) != 0:
                raise RuntimeError("Échec d'encodage ; consulter encodage.log")
        if count == 0 or (expected and count != expected):
            raise ValueError(f"Décodage incomplet : {count} images, {expected} attendues")
        verify_encoded(partial, count, fps, [w,h])
        partial.rename(target)
        per_second=export_seconds(out,count/fps)
        report = dict(source=str(source), source_sha256=sha256(source), checkpoint_sha256=sha256(args.checkpoint),
                      video=target.name, representation=task, codec="H.264 / yuv420p", fps=fps, frames=count,
                      duration_seconds=count/fps, dimensions=[w,h], source_frames=source_frames,
                      source_decoded_frames=timing['decoded_frames'], timestamp_gaps=timing['timestamp_gaps'],
                      frames_added_net=timing['normalized_frames']-timing['decoded_frames'],
                      timing_note='Missing source intervals repeated to preserve timing; repeats are not new observations',
                      complete_source_video=count == timing['normalized_frames'],
                      audio=("Source audio retained when present, re-encoded as AAC" if keep_audio else "Disabled"),
                      device=str(dev), seconds=time.perf_counter()-start,
                      technical_test_only=bool(checkpoint["synthetic_demo"] or checkpoint.get("training_only_demo", False)),
                      prediction_note="Automatic predictions, no manual ground truth or calibrated reliability")
        report['telemetry'] = dict(csv='mesures.csv',measurement_row_fraction=row_fraction,
            min_uncalibrated_model_score=min_score,candidate_frames=candidates,
            rejected_frames=rejected,repeated_frames=repeated_count,
            quantity=('horizontal apparent separation of predicted corridor-boundary pixel centres' if task == 'corridor' else
                      'horizontal apparent separation of predicted inner-edge pixel centres'),
            units='pixels at original image resolution',metric_calibration=None,
            independent_accuracy=None,score=('Mean uncalibrated softmax on local predicted corridor pixels; not accuracy' if task == 'corridor' else
                                            'Mean uncalibrated softmax on local predicted rail pixels; not accuracy'),
            warning='Geometry checks can accept false positives; candidates are not validated physical gauge')
        report['telemetry']['per_second']=per_second
        report['telemetry']['axis'] = ('Exact horizontal image-space midpoint of observed predicted corridor boundaries; '
            'no rail-metal thickness inferred; unsupported gaps are not connected' if task == 'corridor' else
            'Exact horizontal image-space midpoint of observed predicted inner edges; unsupported gaps are not connected')
        report['telemetry']['representation'] = task
        if task == 'corridor':
            report['telemetry']['width_definition'] = 'Horizontal separation of corridor boundary pixel centres; not a validated physical rail gauge'
            report['telemetry']['rail_metal_geometry_available'] = False
        report['telemetry']['temporal_guard']='Heuristic jump rejection, no smoothing or forced gauge; not calibrated reliability'
        if smoother:
            report['display_smoothing']=dict(enabled=True,tau_seconds=smoother.tau,max_gap_seconds=smoother.max_gap,
                jump_threshold_px=smoother.jump,method=smoothing_method,
                raw_measurements='mesures.csv',exports=export_smoothed(out,count/fps),
                warning='May lag on curves; does not correct false detections or prove metric accuracy. Raw measurements remain unchanged.')
            if smoothing_method == 'adaptive':
                report['display_smoothing'].update(min_cutoff_hz=smoother.min_cutoff,
                    beta_normalized=smoother.beta, derivative_cutoff_hz=smoother.derivative_cutoff)
        report['clock'] = dict(filter_clock='physical_utc' if physical_clock else 'source_presentation',
            telemetry_source=str(telemetry_path) if telemetry_path else None,
            telemetry_sha256=sha256(telemetry_path) if telemetry_path else None,
            physical_time_available=bool(physical_clock),
            speed_uncertainty='not supplied',
            note='Playback seconds locate frames; UTC elapsed seconds drive filtering when supplied.')
        report['processing_fps']=count/report['seconds']
        report['inference_roi_xyxy_exclusive']=roi
        report['throughput_meets_source_fps']=report['processing_fps']>=fps
        dump(out / "video.json", report)
        print(f"Vidéo prête : {target}", flush=True)
        return report
    except BaseException as error:
        dump(out / "echec.json", {"frames_processed": count, "error": str(error)})
        raise
    finally:
        capture.release()
        if preview:preview.close()
        if reader is not None:
            if reader.stdout is not None and not reader.stdout.closed:
                reader.stdout.close()
            if reader.poll() is None:
                reader.terminate()
                reader.wait(timeout=10)
        if process is not None and process.poll() is None:
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            process.terminate()
            process.wait(timeout=10)
