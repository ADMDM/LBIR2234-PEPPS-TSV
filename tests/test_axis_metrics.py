import numpy as np
from railtsv.axis_metrics import AxisMetrics


def test_axis_error_and_missing_predictions_are_reported_separately():
    reference=np.zeros((128,300),np.uint8)
    for y in range(128):
        x=40+y//8;reference[y,x:x+130]=1
    prediction=np.roll(reference,7,axis=1)
    prob=np.eye(2,dtype=np.float32)[prediction].transpose(2,0,1)
    metrics=AxisMetrics();metrics.update(prediction,prob,reference,'curve')
    report=metrics.result()
    assert report['coverage_pct']==100 and report['mean_absolute_center_error_px']==7
    assert report['p95_absolute_center_error_px']==7
    absent=np.zeros_like(reference);p=np.eye(2,dtype=np.float32)[absent].transpose(2,0,1)
    metrics.update(absent,p,reference,'missing')
    report=metrics.result()
    assert report['coverage_pct']==50 and report['mean_absolute_center_error_px']==7
    assert report['per_image'][1]['mean_absolute_center_error_px'] is None


def test_ignore_only_reference_has_no_invented_accuracy():
    reference=np.full((128,300),255,np.uint8);prediction=np.zeros_like(reference)
    prob=np.eye(2,dtype=np.float32)[prediction].transpose(2,0,1)
    m=AxisMetrics();m.update(prediction,prob,reference)
    assert m.result()['mean_absolute_center_error_px'] is None and m.result()['coverage_pct'] is None
