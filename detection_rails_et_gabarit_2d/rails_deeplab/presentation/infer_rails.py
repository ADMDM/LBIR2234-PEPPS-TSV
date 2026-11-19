"""Export native-time corridor boundaries for any source video.

The model and the causal filter are the established TSV version. This entry
point changes packaging, not trained weights. No frame is duplicated at gaps.
Coordinates are those of the specified source, which must be upright.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from railtsv.model import build_model, load_checkpoint
from railtsv.geometry import corridor_profile, TemporalGuard
from railtsv.telemetry import measure_corridor
from railtsv.adaptive_smoothing import AdaptiveProfileSmoother


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def predict(model, rgb, device, height, width):
    h, w = rgb.shape[:2]
    scale = min(height / h, width / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = (height - nh) // 2, (width - nw) // 2
    prepared = np.full((height, width, 3), (124, 116, 104), np.uint8)
    prepared[top:top+nh, left:left+nw] = cv2.resize(rgb, (nw, nh))
    x = (prepared.astype(np.float32)/255 - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    with torch.inference_mode():
        logits = model(torch.from_numpy(x.transpose(2, 0, 1).copy())[None].to(device)).cpu()
        logits = logits[:, :, top:top+nh, left:left+nw]
        probs = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False).softmax(1)[0].numpy()
    return probs.argmax(0).astype(np.uint8), probs


def run(source, checkpoint_path, destination, video_id, device='auto', max_frames=0):
    source, destination = Path(source).resolve(), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint['task'] != 'corridor' or checkpoint['classes'] != ['background', 'couloir']:
        raise ValueError('Expected the background/corridor checkpoint')
    model = build_model(2)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device).eval()
    identity = sha(source)
    common = ['video_id', 'frame_index', 'time_video_s', 'image_width', 'image_height']
    af = common + ['segment_index', 'y_px', 'left_raw_px', 'right_raw_px', 'axis_raw_px', 'left_display_px', 'right_display_px', 'axis_display_px', 'display_valid', 'smoothing_status']
    ff = common + ['rail_points_raw', 'rail_points_displayed', 'rail_temporal_status']
    rf = ['video_id', 'frame_index', 'source_frame_index', 'video_time_s', 'source_pts_s', 'image_width_px', 'image_height_px', 'left_segments_json', 'right_segments_json', 'rail_points_per_side', 'rails_status', 'geometry_variant', 'temporal_status', 'observation_status', 'held_from_frame', 'utc_time', 'real_elapsed_s', 'time_basis', 'train_speed_m_s', 'train_speed_km_h', 'speed_status', 'speed_uncertainty_m_s', 'speed_is_estimated', 'latitude_deg', 'longitude_deg', 'left_actual_error_px', 'right_actual_error_px', 'error_status', 'metric_status', 'source_video_sha256']
    start = time.perf_counter()
    count = supported = 0
    meta = dict(status='running', video_id=video_id, source=str(source), source_sha256=identity,
                checkpoint_sha256=sha(checkpoint_path), network_width=checkpoint['width'], network_height=checkpoint['height'],
                device=device, clock='native PTS; no duplication; no extrapolation at gaps',
                smoothing=dict(min_cutoff_hz=.8, beta=15, max_gap_seconds=.15),
                metrics_scope='No new ground truth or accuracy measurement', partial_source=bool(max_frames))
    try:
        with av.open(str(source)) as video, (destination/'AXE_ET_LIMITES.csv').open('x', newline='') as ap, (destination/'FRAMES.csv').open('x', newline='') as fp, (destination/'RAILS_SOURCE.csv').open('x', newline='') as rp:
            stream = video.streams.video[0]
            w, h = stream.width, stream.height
            meta.update(width=w, height=h, header_frame_count=stream.frames)
            aw, fw, rw = csv.DictWriter(ap, fieldnames=af), csv.DictWriter(fp, fieldnames=ff), csv.DictWriter(rp, fieldnames=rf)
            for writer in (aw, fw, rw): writer.writeheader()
            smooth = AdaptiveProfileSmoother(w, min_cutoff_hz=.8, beta=15, max_gap_seconds=.15)
            guard = TemporalGuard(w)
            previous = None
            for i, frame in enumerate(video.decode(video=0)):
                if max_frames and i >= max_frames: break
                if frame.pts is None: raise ValueError('Missing source PTS')
                t = float(frame.pts * frame.time_base)
                if previous is not None and t <= previous: raise ValueError('Non-increasing PTS')
                if (frame.width, frame.height) != (w, h): raise ValueError('Changing image dimensions')
                mask, probs = predict(model, frame.to_ndarray(format='rgb24'), device, checkpoint['height'], checkpoint['width'])
                raw = corridor_profile(mask, probs, min_score=.55, step=8)
                status = guard.check(measure_corridor(mask, probs, row_fraction=.8, min_score=.55), t)
                accepted = raw if status != 'jump_rejected' else dict(raw, segments=[], points=0)
                filtered = smooth.update(accepted, t)
                points = {p['y']: p for segment in filtered['segments'] for p in segment}
                base = dict(video_id=video_id, frame_index=i, time_video_s=t, image_width=w, image_height=h)
                for si, segment in enumerate(raw['segments']):
                    for p in segment:
                        q = points.get(p['y'], {})
                        aw.writerow(dict(base, segment_index=si, y_px=p['y'], left_raw_px=p['left_inner_x'], right_raw_px=p['right_inner_x'], axis_raw_px=p['center_x'], left_display_px=q.get('left_inner_x'), right_display_px=q.get('right_inner_x'), axis_display_px=q.get('center_x'), display_valid=bool(q), smoothing_status=q.get('smoothing_status', 'rejected_or_absent')))
                left = [[[p['left_inner_x'], p['y']] for p in s] for s in filtered['segments']]
                right = [[[p['right_inner_x'], p['y']] for p in s] for s in filtered['segments']]
                fw.writerow(dict(base, rail_points_raw=sum(map(len, raw['segments'])), rail_points_displayed=len(points), rail_temporal_status=status))
                row = dict.fromkeys(rf, '')
                row.update(video_id=video_id, frame_index=i, source_frame_index=i, video_time_s=t, source_pts_s=t, image_width_px=w, image_height_px=h, left_segments_json=json.dumps(left, separators=(',', ':')), right_segments_json=json.dumps(right, separators=(',', ':')), rail_points_per_side=len(points), rails_status='observed' if points else 'unavailable', geometry_variant='causal_display_smoothing', temporal_status=status, observation_status='observed_on_this_frame' if points else 'no_accepted_corridor', time_basis='native_video_pts', speed_status='not_available', error_status='no_per_frame_ground_truth', metric_status='uncalibrated', source_video_sha256=identity)
                rw.writerow(row)
                if count == 0: meta['first_source_time_s'] = t
                previous = t
                count += 1
                supported += bool(points)
                if count % 150 == 0:
                    for handle in (ap, fp, rp): handle.flush()
                    print(json.dumps(dict(video_id=video_id, frames=count, video_seconds=round(t, 2), runtime_s=round(time.perf_counter()-start, 1))), flush=True)
            if not count: raise ValueError('No decoded frames')
        meta.update(status='complete', frames=count, last_source_time_s=previous, frames_with_corridor=supported, runtime_s=time.perf_counter()-start, axis_export_sha256=sha(destination/'AXE_ET_LIMITES.csv'), frame_export_sha256=sha(destination/'FRAMES.csv'), native_rail_export_sha256=sha(destination/'RAILS_SOURCE.csv'))
    except Exception as exc:
        meta.update(status='failed', frames=count, error=str(exc))
        raise
    finally:
        (destination/'SOURCE_ET_PARAMETRES.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'checkpoint', 'out'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--video-id', required=True)
    p.add_argument('--device', default='auto', choices=['auto', 'mps', 'cpu'])
    p.add_argument('--max-frames', type=int, default=0)
    a = p.parse_args()
    print(json.dumps(run(a.source, a.checkpoint, a.out, a.video_id, a.device, a.max_frames), indent=2))
