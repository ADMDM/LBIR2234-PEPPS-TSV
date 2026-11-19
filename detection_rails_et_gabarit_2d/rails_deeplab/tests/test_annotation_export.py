import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('export_labelme',Path(__file__).resolve().parents[1]/'scripts/export_labelme.py')
export=importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)

def fixture():
    return {'imageWidth':20,'imageHeight':20,'flags':{'reviewed':True},'shapes':[
        {'label':'rail_left','shape_type':'polygon','points':[[1,1],[4,1],[4,18],[1,18]]},
        {'label':'rail_right','shape_type':'polygon','points':[[14,1],[18,1],[18,18],[14,18]]},
        {'label':'ignore','shape_type':'polygon','points':[[0,0],[19,0],[19,3],[0,3]]}]}

def test_semantics_and_ignore_priority():
    m=export.rasterize(fixture(),(20,20))
    assert m.dtype==np.uint8
    assert m[10,2]==1 and m[10,16]==2 and m[2,2]==255 and m[10,9]==0

def test_unreviewed_rejected():
    d=fixture();d['flags']['reviewed']=False
    with pytest.raises(ValueError,match='non validée'):export.rasterize(d,(20,20))

def test_overlap_rejected():
    d=fixture();d['shapes'][1]['points']=d['shapes'][0]['points']
    with pytest.raises(ValueError,match='chevauchent'):export.rasterize(d,(20,20))

def test_labelme_canvas_edges_cover_last_pixels_without_mutating_annotation():
    points=[[0,0],[20,0],[20,20],[0,20]]
    d=fixture()
    d['shapes']=[{'label':'rail_left','shape_type':'polygon','points':points}]
    mask=export.rasterize(d,(20,20))
    assert mask.shape==(20,20) and np.all(mask==1)
    assert points==[[0,0],[20,0],[20,20],[0,20]]

@pytest.mark.parametrize('point', [[20.01,1],[1,20.01],[-0.01,1],[1,-0.01]])
def test_points_beyond_canvas_still_rejected(point):
    d=fixture()
    d['shapes'][0]['points'][0]=point
    with pytest.raises(ValueError,match='hors image'):export.rasterize(d,(20,20))
