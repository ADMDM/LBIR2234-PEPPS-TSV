"""CPU geometry/contract tests; these do not measure detector accuracy."""
import copy
import unittest

import numpy as np

from selective_refinement import select_additions


def instance(box=(20, 20, 25, 25), *, score=.7, key="t0", source="litter",
             kind="tile", shape=(100, 100), border=False, class_id=0):
    mask = np.zeros(shape, bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    return dict(mask=mask, mask_bbox=list(box), confidence_score=score,
                candidate_id=key, observation_id=f"observation:{key}",
                model_source=source, class_id=class_id, class_name="dechet",
                view_kind=kind, view_id=key, touches_crop_border=border,
                raw_detection_index=0)


class SelectionTests(unittest.TestCase):
    def select(self, base=(), tiles=(), score=.5, area=.01, **kw):
        return select_additions(base, tiles, score, area, **kw)

    def test_addition_only_and_no_mutation_or_union(self):
        base = instance((60, 60, 70, 70), key="b0", kind="global")
        tile = instance()
        old = copy.deepcopy([base, tile])
        kept, audit = self.select([base], [tile])
        self.assertEqual([o["candidate_id"] for o in kept], ["t0"])
        self.assertIs(kept[0]["mask"], tile["mask"])
        for item, before in zip([base, tile], old):
            np.testing.assert_array_equal(item["mask"], before["mask"])
            self.assertEqual(set(item), set(before))
        self.assertTrue(audit[0]["accepted"])
        self.assertEqual(audit[0]["mask_area_pixels"], 25)
        self.assertEqual(audit[0]["mask_area_fraction"], .0025)

    def test_confidence_and_area_exact_equality_allowed(self):
        kept, audit = self.select(tiles=[instance(score=.5)], score=.5, area=.0025)
        self.assertEqual(len(kept), 1)
        self.assertEqual(audit[0]["reason"], "accepted_selective_addition")

    def test_low_confidence_and_large_area_both_audited(self):
        kept, audit = self.select(tiles=[instance(score=.49)], area=.0024)
        self.assertFalse(kept)
        self.assertEqual(audit[0]["rejection_reasons"],
                         ["below_min_confidence", "exceeds_max_area_fraction"])

    def test_crop_edge_rejected(self):
        kept, audit = self.select(tiles=[instance(border=True)])
        self.assertFalse(kept)
        self.assertIn("touches_crop_border", audit[0]["rejection_reasons"])

    def test_all_four_image_edges_rejected(self):
        for box in [(0, 20, 5, 25), (20, 0, 25, 5),
                    (95, 20, 100, 25), (20, 95, 25, 100)]:
            with self.subTest(box=box):
                kept, audit = self.select(tiles=[instance(box)])
                self.assertFalse(kept)
                self.assertIn("touches_image_border", audit[0]["rejection_reasons"])

    def test_contained_small_mask_rejected_by_ios_across_classes(self):
        base = instance((10, 10, 60, 60), source="public", class_id=7,
                        kind="global", key="b7")
        kept, audit = self.select([base], [instance()])
        self.assertFalse(kept)
        c = audit[0]["baseline_conflicts"][0]
        self.assertLess(c["mask_iou"], .1)
        self.assertEqual(c["mask_ios"], 1)
        self.assertEqual(audit[0]["baseline_ids"], ["b7"])
        self.assertEqual(c["baseline_observation_id"], "observation:b7")

    def test_iou_rule_even_when_ios_at_threshold(self):
        base = instance((20, 20, 30, 30), kind="global", key="b")
        tile = instance((28, 20, 38, 30))
        kept, audit = self.select([base], [tile])
        self.assertFalse(kept)
        c = audit[0]["baseline_conflicts"][0]
        self.assertEqual(c["mask_ios"], .2)
        self.assertGreater(c["mask_iou"], .1)

    def test_exact_iou_and_ios_thresholds_allowed(self):
        base = instance((20, 20, 32, 30), kind="global", key="b")
        tile = instance((30, 20, 40, 30))
        # 20 overlapping pixels, areas 120 and 100 => IoU=.1, IoS=.2.
        kept, audit = self.select([base], [tile])
        self.assertEqual(len(kept), 1)
        self.assertEqual(audit[0]["baseline_conflicts"], [])

    def test_bbox_overlap_without_mask_overlap_kept(self):
        base = instance((20, 20, 30, 30), kind="global")
        base["mask"][21:29, 21:29] = False
        tile = instance((22, 22, 27, 27))
        kept, audit = self.select([base], [tile])
        self.assertEqual(len(kept), 1)
        self.assertFalse(audit[0]["baseline_conflicts"])

    def test_selected_additions_overlap_keeps_higher_score(self):
        lo = instance((20, 20, 30, 30), score=.6, key="low")
        hi = instance((21, 20, 31, 30), score=.8, key="high")
        kept, audit = self.select(tiles=[lo, hi])
        self.assertEqual([o["candidate_id"] for o in kept], ["high"])
        self.assertEqual(audit[1]["reason"], "overlap_with_selected_addition")
        self.assertEqual(audit[1]["selected_addition_conflicts"][0]
                         ["selected_addition_id"], "high")

    def test_selected_addition_tie_uses_candidate_id_not_input_order(self):
        a = instance(key="a")
        z = instance(key="z")
        for items in ([z, a], [a, z]):
            kept, audit = self.select(tiles=items)
            self.assertEqual([o["candidate_id"] for o in kept], ["a"])
            self.assertEqual(audit[-1]["candidate_id"], "z")

    def test_rejected_high_score_cannot_block_valid_addition(self):
        hi = instance(score=.9, key="edge", border=True)
        lo = instance(score=.8, key="good")
        kept, audit = self.select(tiles=[hi, lo])
        self.assertEqual([o["candidate_id"] for o in kept], ["good"])

    def test_disjoint_tile_neighbors_both_kept(self):
        a = instance((20, 20, 25, 25), key="a")
        b = instance((25, 20, 30, 25), key="b")
        self.assertEqual(len(self.select(tiles=[a, b])[0]), 2)

    def test_empty_cases_and_explicit_dimensions(self):
        self.assertEqual(self.select(), ([], []))
        self.assertEqual(self.select(image_width=640, image_height=480), ([], []))
        self.assertEqual(self.select([instance(kind="global")]), ([], []))
        kept, _ = self.select(tiles=[instance()], image_width=100, image_height=100)
        self.assertEqual(len(kept), 1)

    def test_invalid_thresholds_and_dimensions(self):
        for kw in [dict(score=float("nan")), dict(score=-.1), dict(area=0),
                   dict(area=float("inf")), dict(image_width=100),
                   dict(image_width=99, image_height=100)]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.select(tiles=[instance()], **kw)

    def test_reject_bad_grid_or_bbox_or_fractional_mask(self):
        for name in ["grid", "bbox", "dtype", "empty"]:
            obj = instance()
            if name == "grid":
                obj["mask"] = obj["mask"][:-1]
            elif name == "bbox":
                obj["mask_bbox"][0] -= 1
            elif name == "dtype":
                obj["mask"] = obj["mask"].astype(float)
            else:
                obj["mask"][:] = False
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.select([instance(kind="global")], [obj])

    def test_global_or_public_not_accepted_as_tile_input(self):
        for obj in [instance(kind="global"), instance(source="public")]:
            with self.assertRaises(ValueError):
                self.select(tiles=[obj])


if __name__ == "__main__":
    unittest.main()
