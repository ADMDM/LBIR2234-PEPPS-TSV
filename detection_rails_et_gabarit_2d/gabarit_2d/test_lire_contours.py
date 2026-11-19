"""Contrats du lecteur ; fixtures synthétiques temporaires, hors livraison."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from corridor_geometry import extract_contours
from lire_contours import load_frame, get_objects, object_masks


def digest(mask):
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "frame.txt"
        self.full = np.zeros((12, 14), bool)
        self.full[1:10, 1:10] = True
        self.full[3:8, 3:8] = False
        self.full[5:6, 5:6] = True
        self.full[11, 13] = True
        self.inside = self.full.copy()
        self.inside[:, :6] = False
        self.frame = {
            "video_id": "TEST_ONLY", "frame_index": 2, "time_video_s": 0.1,
            "source_pts": 3, "time_base_num": 1, "time_base_den": 30,
            "image_width": 14, "image_height": 12, "schema_version": 1,
            "processing_status": "ok", "coordinate_convention": "original pixel centres, x right, y down",
            "rail_support": {}, "objects": [{
                "observation_id": "TEST_ONLY:f000002:o000", "class_id": 0,
                "class_name": "dechet", "model_source": "litter", "confidence_score": 0.5,
                "track_id": None, "track_confirmed": False, "observed_hits": 0,
                "analysis": {"relation_2d": "overlap_observed", "mask_pixel_count": int(self.full.sum()),
                             "overlap_pixel_count": int(self.inside.sum())},
                "full_predicted_mask_contours": extract_contours(self.full),
                "corridor_intersection_contours": extract_contours(self.inside),
                "full_mask_sha256": digest(self.full), "intersection_mask_sha256": digest(self.inside),
            }],
        }

    def tearDown(self):
        self.temp.cleanup()

    def write(self, frame=None):
        self.path.write_text(json.dumps(self.frame if frame is None else frame), encoding="utf-8")
        return self.path

    def test_read_and_reconstruct_nested_hole_island_and_point(self):
        loaded = load_frame(self.write())
        full, inside = object_masks(loaded["objects"][0], 14, 12)
        np.testing.assert_array_equal(full, self.full)
        np.testing.assert_array_equal(inside, self.inside)
        self.assertEqual(len(get_objects(self.path)), 1)

    def test_empty_is_valid(self):
        self.frame["objects"] = []
        self.assertEqual(get_objects(self.write()), [])

    def test_point_outside_rejected(self):
        self.frame["objects"][0]["full_predicted_mask_contours"][0]["points"][0][0] = 14
        with self.assertRaises(ValueError): load_frame(self.write())

    def test_noninteger_point_rejected(self):
        self.frame["objects"][0]["full_predicted_mask_contours"][0]["points"][0][0] = 1.5
        with self.assertRaises(ValueError): load_frame(self.write())

    def test_bad_parent_rejected(self):
        self.frame["objects"][0]["full_predicted_mask_contours"][0]["parent_contour_id"] = 0
        with self.assertRaises(ValueError): load_frame(self.write())

    def test_bad_hole_flag_rejected(self):
        self.frame["objects"][0]["full_predicted_mask_contours"][0]["is_hole"] = True
        with self.assertRaises(ValueError): load_frame(self.write())

    def test_duplicate_observation_rejected(self):
        self.frame["objects"].append(copy.deepcopy(self.frame["objects"][0]))
        with self.assertRaises(ValueError): load_frame(self.write())

    def test_bad_time_rejected(self):
        self.frame["time_video_s"] = 2 / 30
        with self.assertRaises(ValueError): load_frame(self.write())

    def test_nan_rejected(self):
        self.frame["objects"][0]["confidence_score"] = float("nan")
        with self.assertRaises(ValueError): load_frame(self.write())

    def test_boolean_dimension_rejected(self):
        self.frame["image_width"] = True
        with self.assertRaises(ValueError): load_frame(self.write())

    def test_bad_hash_rejected(self):
        self.frame["objects"][0]["full_mask_sha256"] = "0" * 64
        loaded = load_frame(self.write())
        with self.assertRaises(ValueError): object_masks(loaded["objects"][0], 14, 12)

    def test_duplicate_json_key_rejected(self):
        self.path.write_text('{"schema_version":1,"schema_version":2}', encoding="utf-8")
        with self.assertRaises(ValueError): load_frame(self.path)

    def test_v2_quarantine_remains_separate_and_reconstructable(self):
        self.frame["schema_version"] = 2
        obj = self.frame["objects"].pop()
        obj["review_reason"] = "ambiguous_train_tracklike"
        self.frame["quarantined_predictions"] = [obj]
        loaded = load_frame(self.write())
        self.assertEqual(get_objects(self.path), [])
        full, inside = object_masks(loaded["quarantined_predictions"][0],14,12)
        np.testing.assert_array_equal(full,self.full)
    def test_duplicate_id_across_accepted_and_review_rejected(self):
        import copy
        self.frame["schema_version"] = 2
        obj=copy.deepcopy(self.frame["objects"][0]);obj["review_reason"]="ambiguous_train_tracklike"
        self.frame["quarantined_predictions"]=[obj]
        with self.assertRaises(ValueError):load_frame(self.write())
    def test_review_reason_required(self):
        self.frame["schema_version"]=2;self.frame["quarantined_predictions"]=[self.frame["objects"].pop()]
        with self.assertRaises(ValueError):load_frame(self.write())


if __name__ == "__main__": unittest.main()
