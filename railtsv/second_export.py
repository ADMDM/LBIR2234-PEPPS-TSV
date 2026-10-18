"""One row per second, preserving complete frame records and an axis snapshot."""
import csv
import json
import math
from pathlib import Path
import numpy as np

SECOND_FIELDS=['second','interval_start_s','interval_end_s','frames','source_observations',
    'repeated_frames','candidate_frames','rejected_frames','coverage_pct',
    'apparent_width_px_median','apparent_width_px_mean','apparent_width_px_min',
    'apparent_width_px_max','apparent_width_px_std','center_x_px_median','center_y_px',
    'model_score_left_median','model_score_right_median','width_m','independent_accuracy_pct',
    'representative_frame_index','representative_seconds','axis_points_json','rejection_reasons_json',
    'representation','corridor_model_score_median']
AXIS_FIELDS=['second','frame_index','seconds','source_frame_index','source_seconds',
    'segment','point','y_px','center_x_px','left_inner_x_px','right_inner_x_px',
    'left_outer_x_px','right_outer_x_px','apparent_width_px','units','representation',
    'left_boundary_x_px','right_boundary_x_px','section_status','temporal_status',
    'model_score_left','model_score_right','corridor_model_score','width_m','independent_accuracy_pct']
FRAME_AXIS_FIELDS = AXIS_FIELDS[1:]


def _repeated(row):
    return str(row.get('repeated_source_frame', '')).lower() in ('true', '1', 'yes')


def _axis_records(row):
    """Actual point coordinates, no hidden interpolation or metric conversion."""
    if _repeated(row):
        return
    for segment, points in enumerate(json.loads(row.get('axis_points_json') or '[]')):
        for number, point in enumerate(points):
            yield dict(frame_index=row['frame_index'], seconds=row['seconds'],
                source_frame_index=row['source_frame_index'], source_seconds=row['source_seconds'],
                segment=segment, point=number, y_px=point['y'], center_x_px=point['center_x'],
                left_boundary_x_px=point['left_inner_x'], right_boundary_x_px=point['right_inner_x'],
                left_inner_x_px=point['left_inner_x'], right_inner_x_px=point['right_inner_x'],
                left_outer_x_px=point.get('left_outer_x'), right_outer_x_px=point.get('right_outer_x'),
                apparent_width_px=point['apparent_width_px'], units='original_image_pixels',
                representation=row.get('representation') or 'tsv', section_status=row.get('status'),
                temporal_status=row.get('temporal_status'), model_score_left=row.get('model_score_left'),
                model_score_right=row.get('model_score_right'), corridor_model_score=row.get('corridor_model_score'),
                width_m=None, independent_accuracy_pct=None)


def export_axis_frames(source_csv, target_csv):
    """Exclusive LONG export: every observed point with video/source timestamps.

    Supports old TSV CSVs (representation defaults to TSV) and new corridor CSVs.
    Repeated frames are excluded. Unknown metres and accuracy stay empty.
    """
    points, frames = 0, 0
    with Path(source_csv).open(newline='', encoding='utf-8') as source, \
         Path(target_csv).open('x', newline='', encoding='utf-8') as target:
        writer = csv.DictWriter(target, fieldnames=FRAME_AXIS_FIELDS)
        writer.writeheader()
        for row in csv.DictReader(source):
            frame_points = 0
            for point in _axis_records(row):
                writer.writerow(point)
                points += 1
                frame_points += 1
            frames += bool(frame_points)
    return dict(file=Path(target_csv).name, points=points, observed_frames_with_axis=frames,
                repeated_frames_included=False, units='original_image_pixels',
                sampling='Every observed axis point; exact video and source timestamps retained')


def export_seconds(folder,duration):
    folder=Path(folder)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Une durée strictement positive et finie est requise')
    for name in ['mesures_par_seconde.csv', 'axe_par_seconde.csv', 'axe_toutes_images.csv']:
        if (folder/name).exists():
            raise FileExistsError(folder/name)
    groups={i:[] for i in range(math.ceil(duration-1e-9))}
    with (folder/'mesures.csv').open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):groups[int(float(row['seconds']))].append(row)
    all_axes = export_axis_frames(folder/'mesures.csv', folder/'axe_toutes_images.csv')
    with (folder/'mesures_par_seconde.csv').open('x',newline='',encoding='utf-8') as f, \
         (folder/'axe_par_seconde.csv').open('x',newline='',encoding='utf-8') as af:
        writer=csv.DictWriter(f,fieldnames=SECOND_FIELDS);writer.writeheader()
        axis=csv.DictWriter(af,fieldnames=AXIS_FIELDS);axis.writeheader()
        for second,rows in groups.items():
            observed=[r for r in rows if not _repeated(r)]
            accepted=[r for r in observed if r['status']=='candidate']
            widths=np.array([float(r['apparent_width_px']) for r in accepted])
            reasons={}
            for r in observed:
                if r['status']!='candidate':reasons[r['reason']]=reasons.get(r['reason'],0)+1
            # The axis is one actual observed frame nearest the interval midpoint,
            # never an average of curves from different images or repeated frames.
            eligible=[r for r in observed if r.get('axis_points_json') not in ('','[]',None)]
            representative=min(eligible,key=lambda r:abs(float(r['seconds'])-(second+.5))) if eligible else None
            representations = {r.get('representation') or 'tsv' for r in rows}
            representation = next(iter(representations)) if len(representations) == 1 else ('mixed' if representations else None)
            row=dict(second=second,interval_start_s=second,interval_end_s=min(second+1,duration),
                frames=len(rows),source_observations=len(observed),repeated_frames=len(rows)-len(observed),
                candidate_frames=len(accepted),rejected_frames=len(observed)-len(accepted),
                coverage_pct=100*len(accepted)/len(observed) if observed else None,
                width_m=None,independent_accuracy_pct=None,representation=representation,
                center_y_px=accepted[0]['row_y'] if accepted else None,
                representative_frame_index=representative['frame_index'] if representative else None,
                representative_seconds=representative['seconds'] if representative else None,
                axis_points_json=representative['axis_points_json'] if representative else '[]',
                rejection_reasons_json=json.dumps(reasons,separators=(',',':')))
            for name,fn in [('median',np.median),('mean',np.mean),('min',np.min),('max',np.max),('std',np.std)]:
                row['apparent_width_px_'+name]=float(fn(widths)) if len(widths) else None
            for name,source in [('center_x_px_median','center_x_px'),('model_score_left_median','model_score_left'),('model_score_right_median','model_score_right'),('corridor_model_score_median','corridor_model_score')]:
                values=[float(r[source]) for r in accepted if r.get(source)]
                row[name]=float(np.median(values)) if values else None
            writer.writerow(row)
            if representative:
                for point in _axis_records(representative):
                    axis.writerow(dict(second=second, **point))
    return dict(seconds=len(groups),summary='mesures_par_seconde.csv',axis='axe_par_seconde.csv',
                full_frames='mesures.csv',axis_all_frames='axe_toutes_images.csv',axis_all_frames_details=all_axes,
                axis_sampling='One observed frame nearest each interval midpoint; exact timestamp included',
                aggregation='Candidate observations only; repeats excluded; rejected/empty seconds retained')
