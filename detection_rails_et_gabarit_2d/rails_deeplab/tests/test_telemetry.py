import numpy as np
import pytest

from railtsv.telemetry import measure_section, draw_telemetry


def scene():
    mask=np.zeros((100,200),np.uint8)
    mask[:,30:40]=1
    mask[:,130:142]=2
    probabilities=np.eye(3,dtype=np.float32)[mask].transpose(2,0,1)
    return mask,probabilities


def test_inner_edge_distance_and_no_invented_units_or_accuracy():
    mask,p=scene()
    result=measure_section(mask,p)
    assert result['status']=='candidate'
    assert result['left_inner_x']==39 and result['right_inner_x']==130
    assert result['apparent_width_px']==91
    assert result['width_m'] is None and result['independent_accuracy_pct'] is None


@pytest.mark.parametrize('problem',['missing','distractor','swapped','low_score','cut','unstable'])
def test_missing_ambiguous_or_weak_rails_do_not_produce_a_measurement(problem):
    mask,p=scene()
    if problem=='missing':mask[mask==2]=0
    elif problem=='distractor':mask[:,175:181]=2
    elif problem=='swapped':mask=3-mask;mask[mask==3]=0
    elif problem=='low_score':p[1:]*=.4
    elif problem=='cut':mask[:,:40]=1
    elif problem=='unstable':mask[76:79,30:40]=0;mask[76:79,60:70]=1
    result=measure_section(mask,p)
    assert result['status']=='rejected' and result['apparent_width_px'] is None


def test_panel_draws_section_and_records_limits_without_changing_measurement():
    mask,p=scene()
    result=measure_section(mask,p)
    result['row_y']=240
    result['left_inner_x']=100;result['right_inner_x']=500
    result['apparent_width_px']=400
    image=np.full((300,640,3),100,np.uint8)
    original=result.copy()
    view=draw_telemetry(image,result,2.5)
    assert result==original
    assert view[240,300,0]>200 and view[240,300,1]>200
    assert view[40,600].mean()<30


def test_invalid_parameters_and_probabilities_are_rejected():
    mask,p=scene()
    with pytest.raises(ValueError):measure_section(mask,p,row_fraction=1)
    p[0,0,0]=np.nan
    with pytest.raises(ValueError):measure_section(mask,p)
