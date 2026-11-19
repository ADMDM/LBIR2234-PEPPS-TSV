"""Inspect a video and optionally materialize its display orientation.

The original is never modified. A single MP4 location tag is clip context,
not a timed GPS trace. Use the resulting SHA-256 in the pipeline configuration.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import av


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def location(value):
    match=re.fullmatch(r'([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)(?:[+-]\d+(?:\.\d+)?)?/?',str(value or ''))
    if not match:return None
    lat,lon=map(float,match.groups())
    if (lat,lon)==(0,0) or not -90<=lat<=90 or not -180<=lon<=180:return None
    return dict(latitude_deg=lat,longitude_deg=lon,position_source='MP4 location tag',coordinate_time_known=False)


def inspect(path):
    probe=shutil.which('ffprobe')
    if not probe:raise RuntimeError('Install ffmpeg/ffprobe first')
    data=json.loads(subprocess.check_output([probe,'-v','error','-show_entries','format_tags:stream_side_data','-of','json',str(path)]))
    tags=data.get('format',{}).get('tags',{})
    times=[]
    with av.open(str(path)) as video:
        for frame in video.decode(video=0):
            if frame.pts is None:raise ValueError('Missing timestamp')
            times.append(float(frame.pts*frame.time_base))
        if not times:raise ValueError('No decoded video')
    if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Non-increasing video times')
    return dict(source=str(Path(path).resolve()),source_sha256=sha(path),frames=len(times),first_pts_s=times[0],last_pts_s=times[-1],width=frame.width,height=frame.height,
                display_rotations=[s['rotation'] for stream in data.get('streams',[]) for s in stream.get('side_data_list',[]) if 'rotation' in s],clip_metadata=location(tags.get('location') or tags.get('location-eng')),times=times)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--normalize-output',type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    report=inspect(a.source)
    if a.normalize_output:
        if a.normalize_output.exists():raise FileExistsError(a.normalize_output)
        ffmpeg=shutil.which('ffmpeg')
        if not ffmpeg:raise RuntimeError('Install ffmpeg first')
        subprocess.run([ffmpeg,'-v','error','-n','-i',str(a.source),'-map','0:v:0','-an','-c:v','libx264','-crf','18','-preset','veryfast','-fps_mode','passthrough','-enc_time_base','demux','-video_track_timescale','90000','-movflags','+faststart',str(a.normalize_output)],check=True)
        normalized=inspect(a.normalize_output)
        if normalized['frames']!=report['frames']:raise ValueError('Orientation normalization changed frame count')
        error=max(abs(a-b) for a,b in zip(report['times'],normalized['times']))
        if error>2e-5:raise ValueError('Orientation normalization changed timestamps')
        normalized.pop('times');report['normalized_source']=normalized;report['max_normalization_timestamp_error_s']=error
    report.pop('times');a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False))
