from copy import deepcopy
import math
import numpy as np
import pytest
from railtsv.adaptive_smoothing import AdaptiveProfileSmoother
from railtsv.metric_geometry import gauge_scenario, project_plane, fit_plane_calibration, project_with_pixel_uncertainty
from railtsv.inference_region import predict_region


def profile(c=500., width=200., ys=(16,24,32)):
    return dict(segments=[[dict(y=y,left_inner_x=c-width/2,right_inner_x=c+width/2,
        center_x=c,apparent_width_px=width) for y in ys]],points=len(ys))


def test_adaptive_is_causal_preserves_raw_midpoint_and_gaps():
    smoother=AdaptiveProfileSmoother(1000)
    original=profile(); before=deepcopy(original)
    smoother.update(original,0); assert original == before
    result=smoother.update(profile(506),1/30)
    for p in result['segments'][0]:
        assert 500 < p['center_x'] < 506
        assert p['center_x'] == (p['left_inner_x']+p['right_inner_x'])/2
    smoother.update(dict(segments=[],points=0),2/30)
    p=smoother.update(profile(510),3/30)['segments'][0][0]
    assert p['smoothing_status']=='new_row' and p['center_x']==510
    p=smoother.update(profile(520),1.)['segments'][0][0]
    assert p['smoothing_status']=='reset_gap'
    p=smoother.update(profile(700),1.03)['segments'][0][0]
    assert p['smoothing_status']=='reset_jump'
    with pytest.raises(ValueError):smoother.update(profile(),1.03)


def test_adaptive_reduces_stationary_jitter_and_tracks_motion_faster():
    low=AdaptiveProfileSmoother(1000,beta=0)
    adaptive=AdaptiveProfileSmoother(1000,beta=15)
    raw=[]; filtered=[]; lag_low=[]; lag_adaptive=[]
    for i in range(300):
        t=i/30;truth=500 if i<150 else 500+80*math.sin((t-5)*.8)
        noisy=truth+2*math.sin(i*2.7)
        a=adaptive.update(profile(noisy),t)['segments'][0][0]['center_x']
        b=low.update(profile(noisy),t)['segments'][0][0]['center_x']
        if 30<i<150:raw.append(noisy-500);filtered.append(a-500)
        if i>160:lag_low.append(abs(b-truth));lag_adaptive.append(abs(a-truth))
    assert np.std(filtered)<np.std(raw)*.5
    assert np.mean(lag_adaptive)<np.mean(lag_low)


def test_scenario_cannot_claim_measured_gauge_or_calibration():
    s=gauge_scenario(700,1000,1920,1.435)
    assert s['scale_m_per_px_assumed']==pytest.approx(1.435/300)
    assert s['axis_offset_from_image_midline_m_assumed']==pytest.approx((850-959.5)*1.435/300)
    assert s['measured_gauge_m'] is None and s['ground_x_m'] is None and s['physical_uncertainty_m'] is None
    assert gauge_scenario(950,955,1920)['metric_status']=='rejected_narrow_corridor'
    for args in [(100,90,1920,1.435),(0,100,1920,-1),(0,100,1920,float('nan'))]:
        with pytest.raises(ValueError):gauge_scenario(*args)


def test_known_projective_geometry_independent_control_and_covariance():
    h=np.array([[.01,0,-1],[0,.02,-2],[0,.001,1.]])
    fit=np.array([[0,0],[200,0],[200,200],[0,200],[50,40],[150,160]],float)
    check=np.array([[30,30],[160,30],[50,150]],float)
    result=fit_plane_calibration(fit,project_plane(fit,h),check,project_plane(check,h))
    assert result['control_rmse_m'] < 1e-6
    assert result['field_validated'] is False
    with pytest.raises(ValueError):fit_plane_calibration(fit,project_plane(fit,h),fit[:3],project_plane(fit[:3],h))
    _,cov=project_with_pixel_uncertainty([10,20],np.diag([.01,.02,1]),np.diag([4,9]))
    np.testing.assert_allclose(cov,np.diag([.0004,.0036]))
    with pytest.raises(ValueError):project_with_pixel_uncertainty([10,20],h,[[1,2],[2,1]])
    with pytest.raises(ValueError):project_plane([[0,0]],np.zeros((3,3)))


def test_roi_restores_original_coordinates_without_hallucinating_outside():
    image=np.zeros((100,200,3),np.uint8)
    def predictor(model,image,device,height,width):
        assert image.shape==(60,100,3)
        mask=np.zeros((60,100),np.uint8);mask[:,20:40]=1
        return mask,np.eye(2,dtype=np.float32)[mask].transpose(2,0,1)
    mask,probs=predict_region(None,image,None,64,96,[50,20,150,80],predictor)
    assert np.all(mask[20:80,70:90]==1)
    assert mask.sum()==60*20
    assert np.all(probs[0,:20]==1) and np.all(probs[1,:20]==0)
    with pytest.raises(ValueError):predict_region(None,image,None,64,96,[-1,0,50,50],predictor)


def test_low_frame_rate_is_not_a_missing_observation():
    s=AdaptiveProfileSmoother(1000,max_gap_seconds=max(.15,2.5/5))
    s.update(profile(),0)
    point=s.update(profile(510),.2)['segments'][0][0]
    assert point['smoothing_status']=='filtered' and 500<point['center_x']<510
    point=s.update(profile(520),1.)['segments'][0][0]
    assert point['smoothing_status']=='reset_gap'


def test_no_default_physical_gauge_or_implicit_metres():
    result=gauge_scenario(700,1000,1920)
    assert result['metric_status']=='not_calibrated_no_reference_gauge'
    assert result['assumed_gauge_m'] is None and result['scale_m_per_px_assumed'] is None
    assert result['ground_x_m'] is None and result['measured_gauge_m'] is None
