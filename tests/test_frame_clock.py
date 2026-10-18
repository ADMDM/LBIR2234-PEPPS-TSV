import csv
from copy import deepcopy
import pytest
from railtsv.frame_clock import load_frame_telemetry
from railtsv.adaptive_smoothing import AdaptiveProfileSmoother


def source():
    return [dict(frame_index=i, source_sha256='abc', video_time_s=i*.2,
        utc_time=f'2026-09-29T07:13:{s:02d}+00:00', real_elapsed_s=s-12,
        train_speed_km_h=v, quality_flags='')
        for i,(s,v) in enumerate([(12,'36'),(14,''),(16,'72')])]


def save(tmp_path, rows):
    p=tmp_path/'clock.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return p


def test_real_clock_preserves_speed_missingness(tmp_path):
    clock=load_frame_telemetry(save(tmp_path,source()),'abc',[0,.2,.4])
    assert clock['median_physical_interval_s']==2
    assert clock['reset_gap_s']==5
    assert clock['samples'][2]['physical_seconds']==4
    assert clock['samples'][1]['train_speed_m_s'] is None
    assert clock['samples'][2]['train_speed_m_s']==20


@pytest.mark.parametrize('field,value', [('source_sha256','other'),('frame_index',0),
    ('video_time_s',2),('real_elapsed_s',.2),('train_speed_km_h','nan'),
    ('utc_time','2026-09-29T07:13:14'),('utc_time','2026-09-29T07:13:12Z')])
def test_wrong_video_or_invalid_clocks_are_rejected(tmp_path,field,value):
    rows=source();rows[1][field]=value
    with pytest.raises(ValueError):load_frame_telemetry(save(tmp_path,rows),'abc',[0,.2,.4])


def test_no_interpolation_or_truncation(tmp_path):
    with pytest.raises(ValueError):load_frame_telemetry(save(tmp_path,source()[:2]),'abc',[0,.2,.4])


def test_physical_filter_keeps_exact_midpoint_and_resets_gap():
    def p(x):return {'segments':[[dict(y=8,left_inner_x=x,right_inner_x=x+100,center_x=x+50)]]}
    f=AdaptiveProfileSmoother(1280,max_gap_seconds=5)
    f.update(p(400),0)
    point=f.update(p(410),2)['segments'][0][0]
    assert 450<point['center_x']<460
    assert point['center_x']==(point['left_inner_x']+point['right_inner_x'])/2
    point=f.update(p(420),12)['segments'][0][0]
    assert point['smoothing_status']=='reset_gap' and point['center_x']==470
