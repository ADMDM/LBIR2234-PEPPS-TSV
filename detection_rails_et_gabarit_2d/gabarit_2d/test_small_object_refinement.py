"""CPU-only tests of coordinates, mask preservation, duplicate policy and guard."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from small_object_refinement import (
    tile_windows, predict_tiled_litter, merge_multiscale, train_guard,
)


def instance(mask, *, score=.4, view="global", edge=False, cls=0, name="dechet"):
    yy, xx = np.nonzero(mask)
    return {"mask": mask, "mask_bbox": [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1],
            "model_source": "litter", "class_id": cls, "class_name": name,
            "confidence_score": score, "view_id": view,
            "view_kind": "global" if view == "global" else "tile",
            "touches_crop_border": edge, "raw_detection_index": 0}


def rect(x0, y0, x1, y1, shape=(30, 30)):
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


class TileWindowTests(unittest.TestCase):
    def test_full_crop_coverage_offsets_and_bounds(self):
        windows = tile_windows([13, 25, 1193, 1013], 1920, 1080, 640, .25)
        covered = np.zeros((1080, 1920), bool)
        for x0, y0, x1, y1 in windows:
            self.assertTrue(13 <= x0 < x1 <= 1193 and 25 <= y0 < y1 <= 1013)
            self.assertLessEqual(x1-x0, 640)
            self.assertLessEqual(y1-y0, 640)
            covered[y0:y1, x0:x1] = True
        self.assertTrue(np.array_equal(covered, rect(13, 25, 1193, 1013, (1080, 1920))))
        self.assertEqual(len(windows), len({tuple(w) for w in windows}))

    def test_narrow_v1_crop_has_two_640_tiles(self):
        self.assertEqual(tile_windows([522, 96, 997, 1080], 1920, 1080),
                         [[522, 96, 997, 736], [522, 440, 997, 1080]])

    def test_512_alternative_has_three_tiles(self):
        self.assertEqual(tile_windows([522, 96, 997, 1080], 1920, 1080, 512),
                         [[522, 96, 997, 608], [522, 480, 997, 992], [522, 568, 997, 1080]])

    def test_small_crop_no_padding(self):
        self.assertEqual(tile_windows([4, 5, 9, 10], 20, 30), [[4, 5, 9, 10]])

    def test_invalid_geometry(self):
        for args in [([0, 0, 0, 10], 20, 30, 640, .25),
                     ([-1, 0, 10, 10], 20, 30, 640, .25),
                     ([0, 0, 21, 10], 20, 30, 640, .25),
                     ([0, 0, 10, 10], 20, 30, 0, .25),
                     ([0, 0, 10, 10], 20, 30, 640, 1),
                     ([0, 0, 10, 10], 20, 30, 640, float('nan'))]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                tile_windows(*args)


class FakeModel:
    def __init__(self, make):
        self.make = make
        self.calls = []
        self.predictor = SimpleNamespace(model=SimpleNamespace(end2end=False))

    def predict(self, image, **kwargs):
        import torch
        self.calls.append((image.copy(), kwargs))
        boxes, masks, names = self.make(image, len(self.calls)-1)
        return [SimpleNamespace(boxes=SimpleNamespace(data=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 6)),
                                masks=None if masks is None else SimpleNamespace(data=torch.tensor(masks)),
                                names=names)]


class PredictionTests(unittest.TestCase):
    def test_mask_box_offset_and_dynamic_category(self):
        def make(im, index):
            m = np.zeros((1, *im.shape[:2]), dtype=np.float32)
            m[0, 1:3, 2:4] = 1
            return [[2, 1, 4, 3, .6, 7]], m, {7: "dechet"}
        model = FakeModel(make)
        objs, empty = predict_tiled_litter(model, np.zeros((40, 30, 3), np.uint8),
                                           [10, 20, 18, 26], device="cpu")
        self.assertEqual(empty, 0)
        self.assertEqual(len(objs), 1)
        obj = objs[0]
        self.assertTrue(np.array_equal(obj["mask"], rect(12, 21, 14, 23, (40, 30))))
        self.assertEqual(obj["mask_bbox"], [12, 21, 14, 23])
        self.assertEqual(obj["model_bbox_original_px"], [12., 21., 14., 23.])
        self.assertEqual(obj["class_id"], 7)
        self.assertEqual(obj["class_name"], "dechet")
        self.assertFalse(obj["touches_crop_border"])
        self.assertEqual(model.calls[0][1]["device"], "cpu")
        self.assertTrue(model.calls[0][1]["retina_masks"])
        self.assertTrue(model.calls[0][1]["nms"])

    def test_edge_and_empty_masks(self):
        def make(im, index):
            m = np.zeros((2, *im.shape[:2]), np.float32)
            m[0, 1:3, :2] = 1
            return [[0, 1, 2, 3, .6, 0], [2, 1, 4, 3, .4, 0]], m, {0: "dechet"}
        objs, empty = predict_tiled_litter(FakeModel(make), np.zeros((40, 30, 3), np.uint8),
                                           [10, 20, 18, 26], device="cpu")
        self.assertEqual(empty, 1)
        self.assertTrue(objs[0]["touches_crop_border"])

    def test_image_border_is_not_internal_tile_truncation(self):
        def make(im, index):
            m = np.ones((1, *im.shape[:2]), np.uint8)
            return [[0, 0, im.shape[1], im.shape[0], .6, 0]], m, {0: "dechet"}
        objs, _ = predict_tiled_litter(FakeModel(make), np.zeros((8, 8, 3), np.uint8),
                                       [0, 0, 8, 8], device="cpu")
        self.assertFalse(objs[0]["touches_crop_border"])

    def test_fractional_mask_not_silently_cast(self):
        def make(im, index):
            m = np.zeros((1, *im.shape[:2]), np.float32)
            m[0, 1, 1] = 1
            m[0, 2, 2] = .5
            return [[1, 1, 3, 3, .6, 0]], m, {0: "dechet"}
        with self.assertRaisesRegex(ValueError, "binary"):
            predict_tiled_litter(FakeModel(make), np.zeros((8, 8, 3), np.uint8), [0, 0, 8, 8], device="cpu")

    def test_mask_grid_must_be_native(self):
        def make(im, index):
            return [[1, 1, 3, 3, .6, 0]], np.ones((1, 4, 4), np.uint8), {0: "dechet"}
        with self.assertRaisesRegex(ValueError, "dimensions"):
            predict_tiled_litter(FakeModel(make), np.zeros((8, 8, 3), np.uint8), [0, 0, 8, 8], device="cpu")


class MergeTests(unittest.TestCase):
    def test_identical_cross_view_keeps_one_mask(self):
        a = instance(rect(5, 5, 15, 15), score=.3)
        b = instance(a["mask"].copy(), score=.8, view="tile_0")
        a["candidate_id"], b["candidate_id"] = "global:0", "tile:0"
        before = deepcopy(a)
        kept, audit = merge_multiscale([a], [b])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["view_id"], "tile_0")
        self.assertTrue(np.array_equal(kept[0]["mask"], b["mask"]))
        self.assertEqual(a.keys(), before.keys())
        self.assertEqual(a["mask_bbox"], before["mask_bbox"])
        self.assertEqual(audit[0]["mask_iou"], 1.)
        self.assertEqual(audit[0]["suppressed_candidate_id"], "global:0")
        self.assertEqual(audit[0]["kept_candidate_id"], "tile:0")

    def test_complete_view_wins_over_truncated_higher_score(self):
        full = instance(rect(5, 5, 15, 15), score=.25)
        partial = instance(rect(5, 5, 8, 15), score=.9, view="tile_0", edge=True)
        kept, audit = merge_multiscale([full], [partial])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["view_id"], "global")
        self.assertTrue(np.array_equal(kept[0]["mask"], full["mask"]))
        self.assertEqual(audit[0]["rule"], "cross_view_truncated_mask_ios")

    def test_complete_small_contained_instance_is_not_erased(self):
        large = instance(rect(3, 3, 23, 23), score=.7)
        small = instance(rect(8, 8, 12, 12), score=.4, view="tile_0")
        kept, audit = merge_multiscale([large], [small])
        self.assertEqual(len(kept), 2)
        self.assertFalse(audit)

    def test_same_view_neighbors_never_merged(self):
        a = instance(rect(5, 5, 15, 15), view="tile_0")
        b = instance(rect(6, 5, 16, 15), view="tile_0")
        kept, audit = merge_multiscale([], [a, b])
        self.assertEqual(len(kept), 2)
        self.assertFalse(audit)

    def test_tile_index_fallback_keeps_same_view(self):
        a = instance(rect(5, 5, 15, 15), view="tile_0")
        b = instance(rect(6, 5, 16, 15), view="tile_0")
        for obj in (a, b):
            del obj["view_id"]
            obj["tile_index"] = 2
        kept, audit = merge_multiscale([], [a, b])
        self.assertEqual(len(kept), 2)
        self.assertFalse(audit)

    def test_adjacent_masks_with_overlapping_boxes_not_merged(self):
        a = np.eye(20, dtype=bool)
        b = np.fliplr(a).copy()
        kept, audit = merge_multiscale([instance(a)], [instance(b, view="tile_0")])
        self.assertEqual(len(kept), 2)
        self.assertFalse(audit)

    def test_different_classes_not_merged(self):
        m = rect(5, 5, 15, 15)
        kept, audit = merge_multiscale([instance(m)], [instance(m.copy(), cls=1, view="tile_0")])
        self.assertEqual(len(kept), 2)
        self.assertFalse(audit)

    def test_fractional_or_inconsistent_mask_rejected(self):
        a = instance(rect(5, 5, 15, 15))
        a["mask_bbox"] = [0, 0, 20, 20]
        with self.assertRaisesRegex(ValueError, "extent"):
            merge_multiscale([a], [])
        a = instance(rect(5, 5, 15, 15))
        a["mask"] = a["mask"].astype(float)
        with self.assertRaisesRegex(ValueError, "boolean"):
            merge_multiscale([a], [])


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.corridor = rect(35, 18, 65, 100, (100, 100))
        self.tracklike = instance(rect(30, 14, 70, 100, (100, 100)), score=.34, name="train")

    def test_weak_tracklike_train_quarantined(self):
        result = train_guard(self.tracklike, self.corridor)
        self.assertTrue(result["quarantine"])
        self.assertEqual(result["corridor_coverage"], 1.)
        self.assertEqual(result["reason"], "ambiguous_train_tracklike")

    def test_high_confidence_train_not_globally_excluded(self):
        self.tracklike["confidence_score"] = .8
        self.assertFalse(train_guard(self.tracklike, self.corridor)["quarantine"])

    def test_compact_low_confidence_train_retained(self):
        compact = instance(rect(30, 60, 70, 100, (100, 100)), score=.3, name="train")
        self.assertFalse(train_guard(compact, self.corridor)["quarantine"])

    def test_other_classes_never_guarded(self):
        self.tracklike["class_name"] = "person"
        self.assertFalse(train_guard(self.tracklike, self.corridor)["quarantine"])

    def test_no_corridor_does_not_certify_false_train(self):
        result = train_guard(self.tracklike, np.zeros_like(self.corridor))
        self.assertFalse(result["quarantine"])
        self.assertEqual(result["reason"], "no_observed_corridor")
        self.assertIsNone(result["corridor_coverage"])

    def test_insufficient_coverage_not_guarded(self):
        corridor = rect(0, 0, 100, 100, (100, 100))
        self.assertFalse(train_guard(self.tracklike, corridor)["quarantine"])

    def test_not_near_bottom_retained(self):
        mask = instance(rect(30, 0, 70, 80, (100, 100)), score=.3, name="train")
        corridor = rect(35, 0, 65, 80, (100, 100))
        self.assertFalse(train_guard(mask, corridor)["quarantine"])

    def test_threshold_is_strict_for_confidence(self):
        self.tracklike["confidence_score"] = .5
        self.assertFalse(train_guard(self.tracklike, self.corridor)["quarantine"])


if __name__ == "__main__":
    unittest.main()
