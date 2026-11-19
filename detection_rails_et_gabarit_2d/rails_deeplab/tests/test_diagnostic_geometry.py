import numpy as np
from scripts.diagnostic_plots import compare_profiles


def test_axis_error_and_width_error_are_separate_quantities():
    ref=np.zeros((64,100),np.uint8);ref[:,20:41]=1
    pred=np.zeros_like(ref);pred[:,23:44]=1
    prob=np.stack([(pred==0).astype(np.float32),(pred==1).astype(np.float32)])
    rows,n=compare_profiles(pred,prob,ref,'test','capture')
    assert len(rows)==n and n>0
    assert all(x['center_error_px']==3 and x['width_error_px']==0 for x in rows)
    assert all(x['center_error_pct_width']==3 for x in rows)


def test_ignored_reference_is_not_counted_as_error():
    ref=np.zeros((64,100),np.uint8);ref[:,20:41]=1;ref[:16]=255
    pred=np.zeros_like(ref);pred[:,20:41]=1
    prob=np.stack([(pred==0).astype(np.float32),(pred==1).astype(np.float32)])
    rows,n=compare_profiles(pred,prob,ref,'test','capture')
    assert len(rows)==n and all(x['y_px']>=16 for x in rows)
