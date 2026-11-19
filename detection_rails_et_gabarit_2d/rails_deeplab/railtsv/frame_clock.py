"""Validate a supplied per-source-frame physical clock before video fusion.

Playback seconds locate a displayed frame. UTC elapsed seconds drive motion
and smoothing. A missing speed is never replaced with zero or extrapolated.
"""
import csv
import math
import statistics
from datetime import datetime
from pathlib import Path

TELEMETRY_FIELDS = ['physical_seconds', 'utc_time', 'train_speed_km_h',
                    'train_speed_m_s', 'telemetry_quality_flags']


def load_frame_telemetry(path, source_sha256, source_timestamps):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows or len(rows) != len(source_timestamps):
        raise ValueError('Le relevé doit contenir exactement une ligne par image source décodée')
    parsed, origin, previous = [], None, None
    for i, (r, presentation_time) in enumerate(zip(rows, source_timestamps)):
        if int(r['frame_index']) != i or r['source_sha256'] != source_sha256:
            raise ValueError('Relevé associé à une autre vidéo ou indices de frames incorrects')
        playback = float(r['video_time_s'])
        if not math.isfinite(playback) or abs(playback-presentation_time) > .002:
            raise ValueError('Le temps de lecture du relevé ne correspond pas à la vidéo')
        instant = datetime.fromisoformat(r['utc_time'].replace('Z', '+00:00'))
        if instant.utcoffset() is None:
            raise ValueError('Le fuseau horaire UTC du relevé doit être explicite')
        if origin is None: origin = instant
        physical = (instant-origin).total_seconds()
        declared = float(r['real_elapsed_s'])
        if not math.isfinite(declared) or abs(physical-declared) > .002:
            raise ValueError('Le temps physique déclaré ne correspond pas aux horodatages UTC')
        if previous is not None and physical <= previous:
            raise ValueError('Les horodatages physiques doivent être strictement croissants')
        speed = float(r['train_speed_km_h']) if r['train_speed_km_h'] else None
        if speed is not None and (not math.isfinite(speed) or speed < 0):
            raise ValueError('Vitesse fournie non finie ou négative')
        parsed.append(dict(physical_seconds=physical, utc_time=r['utc_time'],
            train_speed_km_h=speed, train_speed_m_s=speed/3.6 if speed is not None else None,
            telemetry_quality_flags=r.get('quality_flags', '')))
        previous = physical
    intervals = [b['physical_seconds']-a['physical_seconds'] for a,b in zip(parsed, parsed[1:])]
    median = statistics.median(intervals) if intervals else None
    return dict(samples=parsed, source_sha256=source_sha256,
        median_physical_interval_s=median,
        reset_gap_s=max(.15, 2.5*median) if median else .15,
        clock='elapsed UTC seconds', speed_uncertainty='not supplied')
