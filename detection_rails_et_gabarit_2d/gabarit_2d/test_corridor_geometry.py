"""Independent synthetic geometry checks; these do not validate model accuracy."""
import copy
import unittest

import numpy as np

from corridor_geometry import (analyze_object_mask, corridor_from_rows,
                               extract_contours, reconstruct_mask)


def row(y, left=6, right=10, segment=0, valid=True):
    return {"y_px": str(y), "left_display_px": str(left),
            "right_display_px": str(right), "segment_index": str(segment),
            "display_valid": str(valid)}


class CorridorGeometryTests(unittest.TestCase):
    def test_linear_interpolation_in_original_pixels_without_extrapolation(self):
        rail, support, meta = corridor_from_rows([row(2, 3, 7), row(8, 9, 13)], 20, 12)
        expected = np.zeros((12, 20), bool)
        for y in range(2, 9):
            expected[y, y + 1:y + 6] = True
        np.testing.assert_array_equal(rail, expected)
        self.assertEqual(support.sum(), 7 * 20)
        self.assertFalse(support[:2].any())
        self.assertFalse(support[9:].any())
        self.assertEqual(meta["max_vertical_gap_px"], 12)

    def test_fractional_limits_select_integer_centres_not_rounded_edges(self):
        rail, support, _ = corridor_from_rows([row(2, 3.2, 7.8)], 12, 8)
        self.assertEqual(np.flatnonzero(rail[2]).tolist(), [4, 5, 6, 7])
        self.assertEqual(support.sum(), 12)

    def test_segment_gap_is_never_connected_even_with_large_override(self):
        rail, support, _ = corridor_from_rows(
            [row(2), row(4), row(6, segment=1), row(8, segment=1)], 20, 12,
            max_vertical_gap_px=100)
        self.assertFalse(support[5].any())
        self.assertFalse(rail[5].any())
        self.assertTrue(support[3].all())
        self.assertTrue(support[7].all())

    def test_rejected_row_splits_and_can_have_blank_display_coordinates(self):
        bad = row(6, valid=False)
        bad.update(left_display_px="", right_display_px="")
        rail, support, meta = corridor_from_rows([row(2), row(4), bad, row(8), row(10)], 20, 12)
        self.assertFalse(support[5:8].any())
        self.assertFalse(rail[5:8].any())
        self.assertEqual(meta["invalid_sample_rows"], 1)
        self.assertEqual(len(meta["observed_spans"]), 2)

    def test_robust_default_does_not_bridge_large_gap_with_six_pixel_sampling(self):
        rail, support, meta = corridor_from_rows([row(y) for y in (2, 8, 14, 80)], 20, 90)
        self.assertEqual(meta["characteristic_spacing_px"], 6)
        self.assertEqual(meta["max_vertical_gap_px"], 12)
        self.assertFalse(support[15:80].any())
        self.assertEqual(meta["gap_breaks"][0]["gap_px"], 66)
        self.assertTrue(rail[80].any())

    def test_two_far_rows_do_not_define_a_large_default_sampling_interval(self):
        _, support, meta = corridor_from_rows([row(2), row(80)], 20, 90)
        self.assertEqual(meta["max_vertical_gap_px"], 16)
        self.assertEqual(np.flatnonzero(support[:, 0]).tolist(), [2, 80])

    def test_explicit_coarse_sampling_override_is_recorded(self):
        _, support, meta = corridor_from_rows([row(2), row(40)], 20, 50, 40)
        self.assertTrue(support[2:41].all())
        self.assertEqual(meta["gap_limit_method"], "explicit")

    def test_rejected_profile_is_unknown_not_zero_overlap(self):
        rail, support, _ = corridor_from_rows([row(2, valid=False), row(8, valid=False)], 20, 12)
        mask = np.zeros_like(rail)
        mask[3:7, 6:10] = True
        answer = analyze_object_mask(mask, rail, support)
        self.assertEqual(answer["relation_2d"], "indeterminate")
        self.assertEqual(answer["support_status"], "none")
        self.assertIsNone(answer["overlap_pixel_count"])

    def test_empty_profile_is_valid_empty_support(self):
        rail, support, meta = corridor_from_rows([], 20, 12)
        self.assertFalse(rail.any())
        self.assertFalse(support.any())
        self.assertEqual(meta["valid_sample_rows"], 0)

    def test_invalid_coordinates_and_overlap_raise_instead_of_clipping(self):
        cases = [[row(-1)], [row(12)], [row(2, -1, 8)], [row(2, 4, 20)],
                 [row(2, 10, 6)], [row(2, float("nan"), 8)],
                 [row(2), row(2, segment=1)], [row(3), row(2)],
                 [row(2), row(4, segment=1), row(6, segment=0)]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                corridor_from_rows(rows, 20, 12)
        with self.assertRaises(ValueError):
            corridor_from_rows([row(2)], 20, 12, 0)

    def test_false_axis_midpoint_is_rejected(self):
        item = row(2)
        item["axis_display_px"] = "99"
        with self.assertRaises(ValueError):
            corridor_from_rows([item], 20, 12)


class ObjectMaskTests(unittest.TestCase):
    def setUp(self):
        self.rail, self.support, _ = corridor_from_rows([row(0), row(15)], 20, 16, 16)

    def test_same_bounding_box_but_different_real_mask_changes_relation(self):
        # Both masks span x=2..14, y=2..12. The hollow outline has an opening
        # over the corridor on every observed row; the other has an extra bar.
        # Restrict rail support to central rows, so upper/lower frame parts are
        # not mistakenly treated as an inferred track.
        rail, support, _ = corridor_from_rows([row(4), row(10)], 20, 16)
        hollow = np.zeros_like(rail)
        hollow[2:13, 2:4] = True
        hollow[2:13, 13:15] = True
        hollow[2, 2:15] = True
        hollow[12, 2:15] = True
        filled = hollow.copy()
        filled[7, 3:14] = True
        a, b = analyze_object_mask(hollow, rail, support), analyze_object_mask(filled, rail, support)
        self.assertEqual(a["overlap_pixel_count"], 0)
        self.assertEqual(a["relation_2d"], "indeterminate")
        self.assertEqual(b["overlap_pixel_count"], 5)
        self.assertEqual(b["relation_2d"], "overlap_observed")
        self.assertEqual(b["support_status"], "partial")

    def test_L_mask_crosses_left_boundary_and_preserves_exact_area(self):
        mask = np.zeros_like(self.rail)
        mask[3:10, 3:5] = True
        mask[8:10, 3:9] = True
        result = analyze_object_mask(mask, self.rail, self.support)
        self.assertEqual(result["mask_pixel_count"], 22)
        self.assertEqual(result["overlap_pixel_count"], 6)
        self.assertEqual(result["relation_2d"], "overlap_observed")
        self.assertEqual(result["support_fraction"], 1)

    def test_right_crossing_and_both_crossings_use_mask_pixels(self):
        right = np.zeros_like(self.rail)
        right[4:7, 9:15] = True
        both = np.zeros_like(self.rail)
        both[4:7, 2:15] = True
        self.assertEqual(analyze_object_mask(right, self.rail, self.support)["overlap_pixel_count"], 6)
        self.assertEqual(analyze_object_mask(both, self.rail, self.support)["overlap_pixel_count"], 15)

    def test_hole_contains_the_whole_observed_corridor(self):
        mask = np.zeros_like(self.rail)
        mask[2:14, 2:15] = True
        mask[3:13, 4:13] = False
        rail, support, _ = corridor_from_rows([row(4), row(10)], 20, 16)
        result = analyze_object_mask(mask, rail, support)
        self.assertEqual(result["overlap_pixel_count"], 0)
        self.assertEqual(result["relation_2d"], "indeterminate")
        self.assertEqual(sum(c["is_hole"] for c in extract_contours(mask)), 1)

    def test_disconnected_components_are_not_joined_across_corridor(self):
        mask = np.zeros_like(self.rail)
        mask[4:7, 1:3] = True
        mask[4:7, 14:16] = True
        result = analyze_object_mask(mask, self.rail, self.support)
        self.assertEqual(result["relation_2d"], "no_overlap_observed")
        self.assertEqual(result["overlap_pixel_count"], 0)
        contours = extract_contours(mask)
        self.assertEqual(len({c["component_id"] for c in contours}), 2)
        self.assertEqual(np.count_nonzero(reconstruct_mask(contours, 20, 16)[:, 6:11]), 0)

    def test_partial_support_zero_overlap_remains_indeterminate(self):
        rail, support, _ = corridor_from_rows([row(3), row(8)], 20, 16)
        mask = np.zeros_like(rail)
        mask[7:11, 1:3] = True
        result = analyze_object_mask(mask, rail, support)
        self.assertEqual(result["support_fraction"], 0.5)
        self.assertEqual(result["relation_2d"], "indeterminate")
        self.assertEqual(result["overlap_pixel_count"], 0)
        self.assertFalse(result["complete_relation_assessment"])

    def test_object_in_segment_gap_has_no_observed_support(self):
        rail, support, _ = corridor_from_rows([row(1), row(3), row(9, segment=1), row(12, segment=1)], 20, 16)
        mask = np.zeros_like(rail)
        mask[5:8, 7:9] = True
        result = analyze_object_mask(mask, rail, support)
        self.assertEqual(result["support_status"], "none")
        self.assertIsNone(result["overlap_fraction"])

    def test_empty_object_is_not_a_valid_negative_detection(self):
        result = analyze_object_mask(np.zeros_like(self.rail), self.rail, self.support)
        self.assertFalse(result["mask_valid"])
        self.assertEqual(result["relation_reason"], "empty_object_mask")
        self.assertIsNone(result["overlap_pixel_count"])
        self.assertEqual(extract_contours(np.zeros_like(self.rail)), [])

    def test_shape_nonfinite_nonbinary_and_outside_support_are_rejected(self):
        with self.assertRaises(ValueError):
            analyze_object_mask(np.zeros((2, 2)), self.rail, self.support)
        with self.assertRaises(ValueError):
            analyze_object_mask(np.full(self.rail.shape, np.nan), self.rail, self.support)
        with self.assertRaises(ValueError):
            analyze_object_mask(np.full(self.rail.shape, 0.6), self.rail, self.support)
        with self.assertRaises(ValueError):
            analyze_object_mask(self.rail, self.rail, np.zeros_like(self.support))


class ExactContourTests(unittest.TestCase):
    def test_nested_hole_and_island_hierarchy(self):
        mask = np.zeros((24, 30), bool)
        mask[2:22, 2:28] = True
        mask[5:19, 5:25] = False
        mask[9:15, 10:20] = True
        contours = extract_contours(mask)
        self.assertEqual([c["depth"] for c in contours], [0, 1, 2])
        self.assertEqual([c["parent_contour_id"] for c in contours], [None, 0, 1])
        self.assertEqual([c["component_id"] for c in contours], [0, 0, 1])
        np.testing.assert_array_equal(reconstruct_mask(contours, 30, 24), mask)

    def test_isolated_pixel_and_thin_line_are_kept_without_fake_polygons(self):
        mask = np.zeros((12, 20), bool)
        mask[2, 3] = True
        mask[7, 9:15] = True
        contours = extract_contours(mask)
        self.assertEqual({c["boundary_kind"] for c in contours}, {"point", "polyline"})
        point = next(c for c in contours if c["boundary_kind"] == "point")
        self.assertEqual(point["points"], [[3, 2]])
        np.testing.assert_array_equal(reconstruct_mask(contours, 20, 12), mask)

    def test_edge_clipping_keeps_only_visible_pixels(self):
        for ys, xs in [(slice(0, 4), slice(0, 3)), (slice(0, 4), slice(17, 20)),
                       (slice(8, 12), slice(0, 3)), (slice(8, 12), slice(17, 20))]:
            mask = np.zeros((12, 20), bool)
            mask[ys, xs] = True
            with self.subTest(ys=ys, xs=xs):
                contours = extract_contours(mask)
                points = np.array([p for c in contours for p in c["points"]])
                self.assertTrue((points >= 0).all())
                self.assertTrue((points[:, 0] < 20).all())
                self.assertTrue((points[:, 1] < 12).all())
                result = analyze_object_mask(mask, np.zeros_like(mask), np.ones_like(mask))
                self.assertTrue(result["touches_image_border"])
                np.testing.assert_array_equal(reconstruct_mask(contours, 20, 12), mask)

    def test_all_3_by_3_binary_shapes_roundtrip_including_diagonal_contacts(self):
        # Exhaustive local topology coverage: 512 masks, holes, diagonal pixels,
        # junctions, degeneracies, edge contacts, and multiple components.
        for bits in range(512):
            mask = np.array([(bits >> k) & 1 for k in range(9)], bool).reshape(3, 3)
            with self.subTest(bits=bits):
                np.testing.assert_array_equal(reconstruct_mask(extract_contours(mask), 3, 3), mask)

    def test_negative_outside_fractional_and_nonfinite_contours_are_rejected(self):
        mask = np.zeros((12, 20), bool)
        mask[2:5, 3:7] = True
        original = extract_contours(mask)
        for point in ([-1, 2], [20, 2], [3, 12], [3.5, 2], [float("nan"), 2]):
            contours = copy.deepcopy(original)
            contours[0]["points"][0] = point
            with self.subTest(point=point), self.assertRaises(ValueError):
                reconstruct_mask(contours, 20, 12)

    def test_opencv_255_masks_have_same_contours_as_boolean_masks(self):
        mask = np.zeros((12, 20), bool)
        mask[2:5, 3:7] = True
        self.assertEqual(extract_contours(mask), extract_contours(mask.astype(np.uint8) * 255))


if __name__ == "__main__":
    unittest.main(verbosity=2)
