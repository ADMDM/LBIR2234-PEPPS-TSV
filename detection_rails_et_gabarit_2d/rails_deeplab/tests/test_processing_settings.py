import json
import pytest
from railtsv.settings import processing_resolution


def test_vga_setting_does_not_supply_metric_scale(tmp_path):
    assert processing_resolution(tmp_path)==(384,640)
    (tmp_path/'configs').mkdir()
    p=tmp_path/'configs/processing_measurement.json'
    d={'processing_resolution':{'height_px':480,'width_px':640,'preserve_aspect_ratio':True},
       'metric_calibration':{'enabled':False,'reference_gauge_m':None}}
    p.write_text(json.dumps(d));assert processing_resolution(tmp_path)==(480,640)
    d['metric_calibration']['enabled']=True;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='conversion métrique'):processing_resolution(tmp_path)


def test_invalid_size_and_distortion_are_rejected(tmp_path):
    (tmp_path/'configs').mkdir();p=tmp_path/'configs/processing_measurement.json'
    d={'processing_resolution':{'height_px':480,'width_px':641,'preserve_aspect_ratio':True}}
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='multiples'):processing_resolution(tmp_path)
    d['processing_resolution'].update(width_px=640,preserve_aspect_ratio=False);p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='ratio'):processing_resolution(tmp_path)
