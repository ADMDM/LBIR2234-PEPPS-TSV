import numpy as np
import pytest
from railtsv.egopath import egopath_mask


def test_mask_requires_two_observed_ordered_edges_and_keeps_void():
    ann=dict(left_rail=[[20,80],[30,20]],right_rail=[[80,90],[60,30]])
    m=egopath_mask(ann,(100,100))
    assert set(np.unique(m))=={0,1,255}
    assert np.all(m[:30]==255) and np.all(m[81:]==255)
    assert m[50,45]==1 and m[50,5]==0
    with pytest.raises(ValueError):egopath_mask(dict(left_rail=ann['right_rail'],right_rail=ann['left_rail']),(100,100))
