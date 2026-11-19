import csv
import pytest
from scripts.fiabilisation_20261018 import add_overlay_speed as module


def write_anchors(path, second_lat="0"):
    with path.open("w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["frame_index", "video_time_s", "overlay_elapsed_s", "latitude_deg", "longitude_deg"])
        w.writerow([0, 0, 0, 0, 0])
        w.writerow([30, 1, 1, second_lat, 0])


def test_anchors_are_loaded_from_explicit_local_file_and_never_extrapolated(tmp_path, monkeypatch):
    path=tmp_path/"synthetic_anchors.csv";write_anchors(path)
    monkeypatch.setattr(module,"ANCHORS_CSV",path)
    points,intervals=module.build()
    assert len(points)==2 and intervals[0]["speed_m_s"]==0
    assert module.at_frame(15,points,intervals)[1]==pytest.approx(.5)
    assert module.at_frame(31,points,intervals) is None


def test_nonfinite_gps_anchor_is_rejected(tmp_path, monkeypatch):
    path=tmp_path/"synthetic_anchors.csv";write_anchors(path,"nan")
    monkeypatch.setattr(module,"ANCHORS_CSV",path)
    with pytest.raises(ValueError,match="non fini"):
        module.build()
