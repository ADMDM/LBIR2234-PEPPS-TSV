"""Native video timeline with optional sparse OCR crops; no frame-rate reconstruction."""
from pathlib import Path
import hashlib
import json
import math
import av
from PIL import Image


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): digest.update(chunk)
    return digest.hexdigest()


def inspect_video(path, video_id, crop_dir=None, interval_s=1.0, roi=(.78125, .8333333333, 1., 1.), scale=3):
    if interval_s <= 0 or len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise ValueError("Invalid sampling interval or normalized crop")
    if not isinstance(scale, int) or not 1 <= scale <= 6:
        raise ValueError("Invalid OCR enlargement")
    if crop_dir:
        crop_dir = Path(crop_dir); crop_dir.mkdir(parents=True, exist_ok=False)
    frames, samples = [], []
    previous = None
    next_target = None
    previous_time = -math.inf
    def save(record, frame):
        if samples and samples[-1]["frame_index"] == record["frame_index"]:
            return
        image = frame.to_image()
        w, h = image.size
        box = (round(roi[0]*w),round(roi[1]*h),round(roi[2]*w),round(roi[3]*h))
        image = image.crop(box)
        image = image.resize((image.width*scale,image.height*scale),Image.Resampling.LANCZOS)
        name = f"frame_{record['frame_index']:08d}.png"
        image.save(crop_dir/name)
        samples.append(dict(record, crop_file=name, crop_native_box=box))
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        declared = stream.frames
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or not frame.time_base:
                raise ValueError("Video contains a frame without a native timestamp")
            timestamp = float(frame.pts*frame.time_base)
            if timestamp <= previous_time:
                raise ValueError("Native video timestamps are not strictly increasing")
            previous_time = timestamp
            record = {"video_id": video_id, "frame_index": index, "source_frame_index": index,
                      "video_time_s": timestamp, "source_pts_s": timestamp,
                      "source_pts": frame.pts, "time_base": str(frame.time_base),
                      "image_width_px": frame.width, "image_height_px": frame.height}
            frames.append(record)
            if crop_dir:
                if next_target is None:
                    save(record,frame)
                    next_target = (math.floor(timestamp/interval_s)+1)*interval_s
                while timestamp >= next_target:
                    options = [(record,frame)] + ([previous] if previous else [])
                    chosen = min(options,key=lambda item:abs(item[0]["video_time_s"]-next_target))
                    save(*chosen); next_target += interval_s
                previous = (record,frame)
            if index and index % 1500 == 0:
                print(f"Décodage : {index+1} images",flush=True)
        if crop_dir and previous: save(*previous)
    if not frames: raise ValueError("No decoded video frames")
    digest = sha256_file(path)
    for frame in frames: frame["source_video_sha256"] = digest
    metadata = {"video_id":video_id,"source_video":str(Path(path).resolve()),"source_video_sha256":digest,
                "decoded_frames":len(frames),"declared_frames":declared,
                "first_pts_s":frames[0]["video_time_s"],"last_pts_s":frames[-1]["video_time_s"],
                "pyav_version":av.__version__,"native_pts_preserved":True,
                "metadata_frame_count_mismatch":declared not in (0,len(frames)),
                "ocr_interval_s":interval_s if crop_dir else None,"ocr_roi_normalized":roi if crop_dir else None,
                "ocr_samples":len(samples)}
    return frames,samples,metadata
