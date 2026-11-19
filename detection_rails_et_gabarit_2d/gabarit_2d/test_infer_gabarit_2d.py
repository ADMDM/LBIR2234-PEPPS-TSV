#!/usr/bin/env python3
"""CPU-only contract tests for actual ByteTracker and native mask inference.

Uses the installed real ByteTracker. Prediction tests use an explicit model stub;
no network, model weights, GPU inference, or prepared training data are needed.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(tempfile.gettempdir()) / 'gabarit2d_test_yolo'))
os.environ.setdefault('OMP_NUM_THREADS', '1')

import cv2
import numpy as np

cv2.setNumThreads(1)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_gabarit_2d import MaskTracks, merge_instances, predict_instances


class ArrayTensor:
    """Only the CPU array interface consumed by predict_instances."""

    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class MaskResult:
    def __init__(self, values):
        self.data = ArrayTensor(values)

    def __len__(self):
        return len(self.data.values)


class PredictionStub:
    def __init__(self, boxes, masks, names=None):
        self.result = SimpleNamespace(
            boxes=SimpleNamespace(data=ArrayTensor(np.asarray(boxes, dtype=np.float32).reshape(-1, 6))),
            masks=None if masks is None else MaskResult(masks),
            names={7: 'objet_source_7'} if names is None else names,
        )
        self.predictor = SimpleNamespace(model=SimpleNamespace(end2end=False))
        self.calls = []

    def predict(self, view, **kwargs):
        self.calls.append((view.copy(), kwargs.copy()))
        return [self.result]


def object_from_mask(mask, source='public', class_id=1, confidence=.9, raw_index=0):
    mask = np.asarray(mask, dtype=bool)
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError('Test object needs a nonempty mask')
    return {
        'mask': mask,
        'mask_bbox': [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1],
        'model_source': source,
        'class_id': class_id,
        'class_name': f'{source}_{class_id}',
        'confidence_score': confidence,
        'raw_detection_index': raw_index,
    }


def rectangle_object(x=10, y=10, w=18, h=24, source='public', class_id=1, raw_index=0):
    mask = np.zeros((100, 180), dtype=bool)
    mask[y:y + h, x:x + w] = True
    return object_from_mask(mask, source, class_id, raw_index=raw_index)


class PredictionMaskTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.arange(60 * 80 * 3, dtype=np.uint16).reshape(60, 80, 3).astype(np.uint8)

    def predict(self, model, crop=None):
        return predict_instances(model, self.frame, 'litter', 640, .25, 'cpu', crop)

    def test_crop_masks_and_boxes_return_to_original_pixel_coordinates(self):
        crop = [20, 10, 50, 35]
        mask = np.zeros((1, 25, 30), dtype=np.uint8)
        mask[0, 3:7, 4:9] = 1
        model = PredictionStub([[0, 0, 29, 24, .87, 7]], mask)
        objects, empty = self.predict(model, crop)
        self.assertEqual(empty, 0)
        self.assertEqual(len(objects), 1)
        obj = objects[0]
        expected = np.zeros((60, 80), dtype=bool)
        expected[13:17, 24:29] = True
        np.testing.assert_array_equal(obj['mask'], expected)
        self.assertEqual(obj['mask_bbox'], [24, 13, 29, 17])
        self.assertEqual(obj['model_bbox_original_px'], [20., 10., 49., 34.])
        self.assertEqual(obj['class_name'], 'objet_source_7')
        self.assertEqual(obj['class_id'], 7)
        self.assertFalse(obj['touches_crop_border'])
        np.testing.assert_array_equal(model.calls[0][0], self.frame[10:35, 20:50])
        self.assertEqual(model.calls[0][1]['device'], 'cpu')
        self.assertTrue(model.calls[0][1]['retina_masks'])

    def test_full_frame_mask_is_not_replaced_by_the_detection_rectangle(self):
        mask = np.zeros((1, 60, 80), dtype=np.uint8)
        mask[0, 20:30, 35:37] = 1
        mask[0, 28:30, 35:50] = 1
        model = PredictionStub([[0, 0, 80, 60, .9, 7]], mask)
        objects, _ = self.predict(model)
        np.testing.assert_array_equal(objects[0]['mask'], mask[0].astype(bool))
        self.assertEqual(objects[0]['mask_bbox'], [35, 20, 50, 30])
        self.assertEqual(int(objects[0]['mask'].sum()), 46)

    def test_crop_border_flags_only_edges_introduced_by_crop(self):
        mask = np.zeros((1, 25, 30), dtype=np.uint8)
        mask[0, 4:8, 0] = 1
        model = PredictionStub([[0, 4, 1, 8, .9, 7]], mask)
        objects, _ = self.predict(model, [20, 10, 50, 35])
        self.assertTrue(objects[0]['touches_crop_border'])
        model = PredictionStub([[0, 4, 1, 8, .9, 7]], mask)
        objects, _ = self.predict(model, [0, 0, 30, 25])
        self.assertFalse(objects[0]['touches_crop_border'])

    def test_boxes_without_masks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'boxes without instance masks'):
            self.predict(PredictionStub([[1, 1, 5, 5, .9, 7]], None))

    def test_masks_without_boxes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Masks without boxes'):
            self.predict(PredictionStub([], np.ones((1, 60, 80), dtype=np.uint8)))

    def test_native_mask_shape_is_required_not_silently_resized(self):
        for shape in [(1, 30, 40), (2, 60, 80), (60, 80)]:
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, 'Native mask dimensions'):
                self.predict(PredictionStub([[0, 0, 80, 60, .9, 7]], np.ones(shape)))
        with self.assertRaisesRegex(ValueError, 'Native mask dimensions'):
            self.predict(PredictionStub([[0, 0, 30, 25, .9, 7]], np.ones((1, 60, 80))), [20, 10, 50, 35])

    def test_unthresholded_or_nonfinite_mask_values_are_rejected(self):
        for value in [.25, np.nan, np.inf, -1]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'binary native masks'):
                self.predict(PredictionStub([[0, 0, 80, 60, .9, 7]], np.full((1, 60, 80), value)))

    def test_empty_masks_are_counted_and_never_become_boxes(self):
        model = PredictionStub([[0, 0, 80, 60, .9, 7]], np.zeros((1, 60, 80), dtype=np.uint8))
        objects, empty = self.predict(model)
        self.assertEqual(objects, [])
        self.assertEqual(empty, 1)

    def test_empty_result_with_no_masks_is_valid(self):
        objects, empty = self.predict(PredictionStub([], None))
        self.assertEqual((objects, empty), ([], 0))


class MaskFusionTests(unittest.TestCase):
    def test_equal_bounding_boxes_with_disjoint_masks_are_not_duplicates(self):
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros_like(a)
        a[1, 1] = a[8, 8] = True
        b[1, 8] = b[8, 1] = True
        public = object_from_mask(a)
        litter = object_from_mask(b, 'litter', 0)
        self.assertEqual(public['mask_bbox'], litter['mask_bbox'])
        kept, audit = merge_instances([public], [litter])
        self.assertEqual(len(kept), 2)
        self.assertEqual(audit, [])

    def test_true_duplicate_uses_public_mask_precedence_and_logs_mask_iou(self):
        public = rectangle_object()
        public['confidence_score'] = .3
        litter = object_from_mask(public['mask'].copy(), 'litter', 0, .99, 12)
        kept, audit = merge_instances([public], [litter])
        self.assertEqual(len(kept), 1)
        self.assertIs(kept[0], public)
        self.assertEqual(audit[0]['mask_iou'], 1.)
        self.assertEqual(audit[0]['litter_raw_detection_index'], 12)
        self.assertEqual(audit[0]['rule'], 'public_mask_precedence_for_cross_model_duplicate')

    def test_threshold_boundary_uses_documented_strict_greater_than(self):
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros_like(a)
        a[3, 1:6] = True
        b[3, 1:4] = True
        public = object_from_mask(a)
        litter = object_from_mask(b, 'litter', 0)
        kept, audit = merge_instances([public], [litter], threshold=.6)
        self.assertEqual(len(kept), 2)
        self.assertEqual(audit, [])
        kept, audit = merge_instances([public], [litter], threshold=.59)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(audit[0]['mask_iou'], .6)


class RealByteTrackerTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((100, 180, 3), dtype=np.uint8)
        self.tracks = MaskTracks(.25)

    def update(self, objects, index, time_s):
        self.tracks.update(objects, self.frame, index, time_s)
        return objects

    def test_real_bytetracker_is_used(self):
        self.assertEqual(self.tracks.Tracker.__module__, 'ultralytics.trackers.byte_tracker')
        self.assertEqual(self.tracks.Tracker.__name__, 'BYTETracker')

    def test_classes_and_model_sources_have_separate_track_identity(self):
        specifications = [('public', 1), ('public', 2), ('litter', 1)]
        first = self.update([rectangle_object(source=s, class_id=c) for s, c in specifications], 0, 0.)
        first_ids = {(o['model_source'], o['class_id']): o['track_id'] for o in first}
        self.assertEqual(len(set(first_ids.values())), 3)
        self.assertNotIn(None, first_ids.values())
        self.assertTrue(all(o['observed_hits'] == 1 and not o['track_confirmed'] for o in first))
        second = self.update([rectangle_object(source=s, class_id=c) for s, c in reversed(specifications)], 1, .033)
        for obj in second:
            key = obj['model_source'], obj['class_id']
            self.assertEqual(obj['track_id'], first_ids[key])
            self.assertEqual(obj['observed_hits'], 2)
            self.assertTrue(obj['track_confirmed'])

    def test_creating_new_class_tracker_never_reissues_existing_active_id(self):
        original = self.update([rectangle_object()], 0, 0.)[0]['track_id']
        self.update([rectangle_object(), rectangle_object(x=70, class_id=2)], 1, .033)
        self.update([rectangle_object(), rectangle_object(x=70, class_id=2), rectangle_object(x=130)], 2, .066)
        current = self.update([rectangle_object(), rectangle_object(x=70, class_id=2), rectangle_object(x=130)], 3, .099)
        self.assertEqual(current[0]['track_id'], original)
        self.assertIsNotNone(current[2]['track_id'])
        self.assertNotEqual(current[2]['track_id'], original)
        self.assertEqual(len({o['track_id'] for o in current}), 3)
        active_ids = [t.track_id for t in self.tracks.trackers[('public', 1)].tracked_stracks]
        self.assertEqual(len(active_ids), len(set(active_ids)))

    def test_empty_observation_does_not_export_an_extrapolated_mask(self):
        original = self.update([rectangle_object()], 0, 0.)[0]['track_id']
        empty = []
        self.update(empty, 1, .033)
        self.assertEqual(empty, [])
        seen_again = self.update([rectangle_object()], 2, .066)[0]
        self.assertEqual(seen_again['track_id'], original)
        self.assertEqual(seen_again['observed_hits'], 2)

    def test_kalman_predictions_during_absence_do_not_increase_observed_hits(self):
        original = self.update([rectangle_object()], 0, 0.)[0]['track_id']
        for index in range(1, 7):
            self.update([], index, index / 30)
            self.assertEqual(self.tracks.hits[original], 1)
        returned = self.update([rectangle_object()], 7, 7 / 30)[0]
        self.assertEqual(returned['track_id'], original)
        self.assertEqual(returned['observed_hits'], 2)

    def test_expired_track_id_is_not_recycled_after_prolonged_observation_absence(self):
        old_id = self.update([rectangle_object()], 0, 0.)[0]['track_id']
        for index in range(1, 24):
            self.update([], index, index / 30)
        self.update([rectangle_object()], 24, 24 / 30)
        returned = self.update([rectangle_object()], 25, 25 / 30)[0]
        self.assertEqual(self.tracks.generation, 0)
        self.assertIsNotNone(returned['track_id'])
        self.assertNotEqual(returned['track_id'], old_id)
        self.assertEqual(returned['observed_hits'], 2)

    def test_late_class_is_confirmed_after_two_actual_mask_observations(self):
        self.update([rectangle_object()], 0, 0.)
        self.update([rectangle_object(), rectangle_object(x=70, class_id=2)], 1, .033)
        second_sighting = self.update(
            [rectangle_object(), rectangle_object(x=70, class_id=2)], 2, .066
        )[1]
        self.assertIsNotNone(second_sighting['track_id'])
        self.assertEqual(second_sighting['observed_hits'], 2)
        self.assertTrue(second_sighting['track_confirmed'])

    def test_late_object_in_existing_class_counts_its_tentative_first_observation(self):
        self.update([rectangle_object()], 0, 0.)
        self.update([rectangle_object(), rectangle_object(x=130)], 1, .033)
        second_sighting = self.update(
            [rectangle_object(), rectangle_object(x=130)], 2, .066
        )[1]
        self.assertIsNotNone(second_sighting['track_id'])
        self.assertEqual(second_sighting['observed_hits'], 2)
        self.assertTrue(second_sighting['track_confirmed'])

    def test_large_time_gap_resets_identity_generation_and_hit_counts(self):
        first = self.update([rectangle_object()], 0, 0.)[0]
        self.update([rectangle_object()], 1, .033)
        old_id = first['track_id']
        self.update([rectangle_object()], 2, .8)
        self.assertEqual(self.tracks.generation, 1)
        self.assertNotIn(old_id, self.tracks.hits)
        after_gap = self.update([rectangle_object()], 3, .833)[0]
        self.assertIsNotNone(after_gap['track_id'])
        self.assertTrue(after_gap['track_id'].startswith('1:public:1:'))
        self.assertNotEqual(after_gap['track_id'], old_id)
        self.assertEqual(after_gap['observed_hits'], 2)
        self.assertTrue(after_gap['track_confirmed'])

    def test_exact_half_second_gap_does_not_reset(self):
        old_id = self.update([rectangle_object()], 0, 0.)[0]['track_id']
        obj = self.update([rectangle_object()], 1, .5)[0]
        self.assertEqual(self.tracks.generation, 0)
        self.assertEqual(obj['track_id'], old_id)
        self.assertEqual(obj['observed_hits'], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
