import csv
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from railtsv.geometry import corridor_profile
from railtsv.second_export import export_axis_frames, export_seconds
from railtsv.telemetry import CSV_FIELDS, measure_corridor
from railtsv import video


def corridor_scene():
    mask = np.zeros((100, 200), np.uint8)
    for y in range(100):
        left = 30+y//10
        mask[y, left:left+71] = 1
    probabilities = np.eye(2, dtype=np.float32)[mask].transpose(2, 0, 1)
    return mask, probabilities


def test_corridor_midpoint_curves_and_no_inferred_metal_geometry():
    mask, probabilities = corridor_scene()
    profile = corridor_profile(mask, probabilities)
    assert profile['representation'] == 'corridor'
    assert profile['points'] == 13 and len(profile['segments']) == 1
    for segment in profile['segments']:
        for point in segment:
            assert point['center_x'] == (point['left_inner_x']+point['right_inner_x'])/2
            assert point['apparent_width_px'] == 70
            assert 'left_outer_x' not in point and 'right_outer_x' not in point
    result = measure_corridor(mask, probabilities)
    assert result['status'] == 'candidate' and result['corridor_model_score'] == 1
    assert result['center_x_px'] == 72 and result['apparent_width_px'] == 70
    assert all(result[key] is None for key in ['left_outer_x', 'right_outer_x',
        'rail_left_thickness_px', 'rail_right_thickness_px', 'model_score_left',
        'model_score_right', 'width_m', 'independent_accuracy_pct'])


def test_profile_does_not_connect_a_missing_unsampled_row_or_uncertain_rows():
    mask, probabilities = corridor_scene()
    # y=31 is not among exported samples: it must nevertheless split the line.
    mask[31] = 0
    profile = corridor_profile(mask, probabilities)
    assert len(profile['segments']) == 2
    assert max(point['y'] for point in profile['segments'][0]) < 31
    assert min(point['y'] for point in profile['segments'][1]) > 31
    probabilities[1, 40:56] = .3
    profile = corridor_profile(mask, probabilities)
    assert not any(40 <= point['y'] < 56 for segment in profile['segments'] for point in segment)


@pytest.mark.parametrize('problem,reason', [
    ('missing', 'corridor_absent_or_too_thin'),
    ('multiple', 'multiple_segments'),
    ('cut', 'corridor_cut_by_image_edge'),
    ('low_score', 'low_uncalibrated_model_score')])
def test_corridor_rejects_ambiguity_absence_clipped_edges_and_low_score(problem, reason):
    mask, probabilities = corridor_scene()
    if problem == 'missing':
        mask[:] = 0
    elif problem == 'multiple':
        mask[:, 150:155] = 1
    elif problem == 'cut':
        mask[:, :50] = 1
    else:
        probabilities[1] = .4
    result = measure_corridor(mask, probabilities)
    assert result['status'] == 'rejected' and result['reason'] == reason
    assert result['center_x_px'] is None and result['apparent_width_px'] is None
    assert corridor_profile(mask, probabilities)['points'] == 0


def test_corridor_invalid_probability_dimensions_and_parameters():
    mask, probabilities = corridor_scene()
    with pytest.raises(ValueError):
        corridor_profile(mask, probabilities, step=0)
    with pytest.raises(ValueError):
        measure_corridor(mask, probabilities, row_fraction=1)
    with pytest.raises(ValueError):
        measure_corridor(mask, np.ones((3, *mask.shape)))
    probabilities[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        measure_corridor(mask, probabilities)


def write_measurements(path, legacy=False):
    point = dict(y=80, left_inner_x=30, right_inner_x=100, center_x=65,
                 apparent_width_px=70)
    if legacy:
        point.update(left_outer_x=25, right_outer_x=105)
    rows = []
    fields = [field for field in CSV_FIELDS if not legacy or field not in ('representation', 'corridor_model_score')]
    for number, (seconds, repeated) in enumerate([(0., False), (.4, False), (.5, True)]):
        row = {field: '' for field in fields}
        row.update(frame_index=number, seconds=seconds, source_frame_index=number if not repeated else 1,
                   source_seconds=seconds if not repeated else .4, repeated_source_frame=repeated,
                   row_y=80, center_x_px=65, center_y_px=80, left_inner_x=30, right_inner_x=100,
                   apparent_width_px=70, status='repeated' if repeated else 'candidate',
                   axis_points_json=json.dumps([[point]]))
        if not legacy:
            row.update(representation='corridor', corridor_model_score=.9)
        rows.append(row)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize('legacy', [False, True])
def test_long_axis_csv_every_observed_frame_no_repeat_metal_fabrication_or_overwrite(tmp_path, legacy):
    source = tmp_path/'mesures.csv'
    write_measurements(source, legacy)
    original = source.read_bytes()
    report = export_seconds(tmp_path, 1)
    assert source.read_bytes() == original
    assert report['axis_all_frames_details']['points'] == 2
    with (tmp_path/'axe_toutes_images.csv').open() as handle:
        points = list(csv.DictReader(handle))
    assert [point['frame_index'] for point in points] == ['0', '1']
    assert [point['seconds'] for point in points] == ['0.0', '0.4']
    assert all(point['center_x_px'] == '65' and point['left_boundary_x_px'] == '30' for point in points)
    assert all(point['width_m'] == '' and point['independent_accuracy_pct'] == '' for point in points)
    if legacy:
        assert all(point['representation'] == 'tsv' and point['left_outer_x_px'] == '25' for point in points)
    else:
        assert all(point['representation'] == 'corridor' and point['left_outer_x_px'] == '' for point in points)
    with (tmp_path/'mesures_par_seconde.csv').open() as handle:
        second = next(csv.DictReader(handle))
    assert second['source_observations'] == '2' and second['repeated_frames'] == '1'
    assert second['corridor_model_score_median'] == ('' if legacy else '0.9')
    with (tmp_path/'axe_par_seconde.csv').open() as handle:
        assert next(csv.DictReader(handle))['frame_index'] == '1'
    before = (tmp_path/'axe_toutes_images.csv').read_bytes()
    with pytest.raises(FileExistsError):
        export_seconds(tmp_path, 1)
    with pytest.raises(FileExistsError):
        export_axis_frames(source, tmp_path/'axe_toutes_images.csv')
    assert (tmp_path/'axe_toutes_images.csv').read_bytes() == before


class KnownCorridor(torch.nn.Module):
    def forward(self, image):
        batch, _, height, width = image.shape
        logits = torch.zeros((batch, 2, height, width), device=image.device)
        logits[:, 1, :, 20:80] = 20
        return logits


def test_corridor_video_two_class_checkpoint_and_all_axis_frames_csv(tmp_path, monkeypatch):
    source = tmp_path/'source.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (96, 64))
    assert writer.isOpened()
    for number in range(4):
        writer.write(np.full((64, 96, 3), 80+number, np.uint8))
    writer.release()
    checkpoint = tmp_path/'model.pt'
    checkpoint.write_bytes(b'synthetic technical test only')
    monkeypatch.setattr(video, 'load_checkpoint', lambda _: dict(task='corridor',
        classes=['background', 'couloir'], state_dict={}, height=64, width=96, synthetic_demo=True))
    model_classes = []
    def build(classes):
        model_classes.append(classes)
        return KnownCorridor()
    monkeypatch.setattr(video, 'build_model', build)
    out = tmp_path/'result'
    report = video.infer_video(SimpleNamespace(input=str(source), checkpoint=str(checkpoint),
        out=str(out), max_frames=0, cpu_threads=1, device='cpu'))
    assert model_classes == [2] and report['frames'] == 4
    assert report['technical_test_only'] is True
    assert report['telemetry']['representation'] == 'corridor'
    assert report['telemetry']['rail_metal_geometry_available'] is False
    assert report['telemetry']['per_second']['axis_all_frames_details']['points'] == 32
    with (out/'mesures.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 and all(row['representation'] == 'corridor' for row in rows)
    assert all(row['left_outer_x'] == '' and row['model_score_left'] == '' for row in rows)
    assert all(float(row['center_x_px']) == 49.5 for row in rows)
