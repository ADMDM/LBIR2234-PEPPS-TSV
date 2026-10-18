import numpy as np
import pytest
from railtsv.metrics import Metrics
from railtsv.axis_metrics import AxisMetrics


def test_false_positives_false_negatives_and_undefined_denominators():
    m=Metrics(['background','couloir'])
    m.update(np.array([1,1,1,0,0,1]),np.array([1,1,0,1,0,255]))
    r=m.result()['per_class']['couloir']
    assert (r['true_positive_pixels'],r['false_positive_pixels'],r['false_negative_pixels'])==(2,1,1)
    assert r['precision']==pytest.approx(2/3) and r['recall']==pytest.approx(2/3)
    absent=Metrics(['background','couloir']);absent.update(np.array([0,0]),np.array([0,1]))
    assert absent.result()['per_class']['couloir']['precision'] is None
    assert absent.result()['per_class']['couloir']['recall']==0
    empty=Metrics(['background','couloir']);empty.update(np.array([1]),np.array([255]))
    assert empty.result()['per_class']['couloir']['precision'] is None
    assert empty.result()['per_class']['couloir']['recall'] is None


def test_signed_errors_cancel_but_absolute_errors_do_not_and_widths_normalize_separately():
    m=AxisMetrics()
    for width,shift in [(300,6),(600,-6)]:
        reference=np.zeros((128,width),np.uint8);reference[:,50:180]=1
        prediction=np.roll(reference,shift,axis=1)
        prob=np.eye(2,dtype=np.float32)[prediction].transpose(2,0,1)
        m.update(prediction,prob,reference)
    r=m.result()
    assert r['signed_center_bias_px']==0
    assert r['mean_absolute_center_error_px']==6
    assert r['root_mean_square_center_error_px']==6
    assert r['mean_absolute_center_error_pct_width']==pytest.approx(1.5)
    assert r['coverage_pct']==100
    reference=np.zeros((128,300),np.uint8);reference[:,50:180]=1
    prediction=np.zeros_like(reference);prob=np.eye(2,dtype=np.float32)[prediction].transpose(2,0,1)
    m.update(prediction,prob,reference)
    r=m.result()
    assert r['coverage_pct']==pytest.approx(200/3)
    assert r['mean_absolute_center_error_px']==6
    assert r['per_image'][-1]['signed_center_bias_px'] is None


def test_empty_axis_does_not_report_zero_error():
    r=AxisMetrics().result()
    for k in ['signed_center_bias_px','root_mean_square_center_error_px','mean_absolute_center_error_pct_width','max_absolute_center_error_px']:
        assert r[k] is None
