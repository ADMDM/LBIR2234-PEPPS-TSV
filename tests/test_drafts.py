import numpy as np
import pytest
from railtsv.drafts import draft_annotation
from scripts.export_labelme import rasterize


def test_model_drafts_preserve_label_contract_and_need_manual_review():
    mask=np.zeros((256,400),np.uint8)
    for y in range(256):
        x=30+y//10;mask[y,x:x+12]=1;mask[y,x+180:x+194]=2
    p=np.eye(3,dtype=np.float32)[mask].transpose(2,0,1)
    draft=draft_annotation(mask,p,'test.png','checkpoint')
    assert draft['flags']['reviewed'] is False and draft['imagePath']=='../images/test.png'
    assert {s['label'] for s in draft['shapes']}=={'rail_left','rail_right'}
    assert all(s['shape_type']=='polygon' for s in draft['shapes'])
    with pytest.raises(ValueError,match='non validée'):rasterize(draft,(400,256))
    # Software-only check: vectorization roughly retains the original predicted area.
    draft['flags']['reviewed']=True
    raster=rasterize(draft,(400,256))
    for cls in (1,2):assert ((raster==cls)&(mask==cls)).sum()/((raster==cls)|(mask==cls)).sum()>.95


def test_no_suggestion_when_one_rail_is_missing():
    mask=np.zeros((256,400),np.uint8);mask[:,30:40]=1
    p=np.eye(3,dtype=np.float32)[mask].transpose(2,0,1)
    assert draft_annotation(mask,p,'test.png','checkpoint') is None
