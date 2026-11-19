"""Selection-policy tests with synthetic metric fixtures; never model scores."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from evaluate_litter import mask_confidence_curve
import select_litter_settings as selection


def curve(constant=False):
    if constant:
        p, r = [0.5] * 8, [0.5] * 8
    else:
        p, r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1], [1, 0.95, 0.9, 0.85, 0.7, 0.5, 0.2, 0]
    return {"class_id": 0, "class_name": "dechet", "iou_threshold": 0.5,
            "confidence_thresholds": [0, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1], "precision": p, "recall": r}


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(folder, size, score, split="val", constant=False):
    folder.mkdir()
    args = {"imgsz": size, "split": split, "nms": True, "iou": 0.5, "conf": 0.001,
            "max_det": 100, "retina_masks": True, "augment": False,
            "project": str(folder.parent), "name": folder.name, "batch": 2,
            "data": "/synthetic/data.yaml", "device": "cpu"}
    resolved = dict(args, save_dir=str(folder), task="segment", mode="val")
    dump(folder / "RESOLVED_EVAL_ARGS.json", resolved)
    metrics = {"status": "complete", "split": split, "checkpoint_sha256": "a" * 64,
               "dataset_sha256": "b" * 64, "original_coco_mask_evaluation": False,
               "segmentation_backend_end2end": False, "image_count": 10, "ground_truth_instances": 12,
               "mask_metrics": {"metrics/mAP50-95(M)": score}, "mask_confidence_curve_iou50": curve(constant)}
    dump(folder / "METRICS.json", metrics)
    manifest = {"status": "complete", "split": split, "dataset_sha256": "b" * 64,
                "checkpoint": {"sha256": "a" * 64}, "metrics_json": {"sha256": selection.sha(folder / "METRICS.json")},
                "evaluation_arguments": args, "runtime": {"torch": "fixture", "ultralytics": "fixture", "resolved_device": "cpu"},
                "artifacts": [{"path": str(folder / "RESOLVED_EVAL_ARGS.json"), "sha256": selection.sha(folder / "RESOLVED_EVAL_ARGS.json")}]}
    dump(folder / "EVALUATION_MANIFEST.json", manifest)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def pair(self, gain=0.0049, constant=False):
        a, b = self.root / "640", self.root / "960"
        fixture(a, 640, 0.4, constant=constant)
        fixture(b, 960, 0.4 + gain, constant=constant)
        return a, b

    def test_640_retained_below_required_gain_and_maxF2_threshold_selected(self):
        a, b = self.pair()
        result = selection.run(a, b, self.root / "selection.json")
        self.assertEqual(result["selected"]["imgsz"], 640)
        self.assertEqual(result["selected"]["confidence_threshold"], 0.25)
        self.assertAlmostEqual(result["selected"]["F2"], 5 * 0.4 * 0.85 / (4 * 0.4 + 0.85))
        self.assertFalse(result["test_data_read"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            selection.run(a, b, self.root / "selection.json")

    def test_960_selected_at_exact_005_gain(self):
        a, b = self.pair(gain=0.005)
        result = selection.run(a, b, self.root / "selection.json")
        self.assertEqual(result["selected"]["imgsz"], 960)

    def test_tied_F2_selects_highest_confidence(self):
        a, b = self.pair(constant=True)
        result = selection.run(a, b, self.root / "selection.json")
        self.assertEqual(result["selected"]["confidence_threshold"], 0.5)

    def test_linear_interpolation_not_nearest_curve_sample(self):
        simple = dict(curve(), confidence_thresholds=[0, 1], precision=[0, 1], recall=[1, 0])
        row = selection.threshold_table(simple)[2]
        self.assertEqual(row["confidence_threshold"], 0.25)
        self.assertAlmostEqual(row["mask_precision_iou50"], 0.25)
        self.assertAlmostEqual(row["mask_recall_iou50"], 0.75)

    def test_test_split_is_refused(self):
        folder = self.root / "test"
        fixture(folder, 640, 0.4, split="test")
        with self.assertRaisesRegex(ValueError, "val only"):
            selection.load_validation(folder, 640)

    def test_changed_checkpoint_dataset_or_effective_settings_are_refused(self):
        a, b = self.pair()
        first, second = selection.load_validation(a, 640), selection.load_validation(b, 960)
        changed = deepcopy(second)
        changed["metrics"]["checkpoint_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "checkpoints differ"):
            selection.choose(first, changed)
        changed = deepcopy(second)
        changed["metrics"]["dataset_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "datasets/ground truth differ"):
            selection.choose(first, changed)
        changed = deepcopy(second)
        changed["resolved"]["batch"] = 4
        with self.assertRaisesRegex(ValueError, "Resolved validation settings differ"):
            selection.choose(first, changed)

    def test_native_curve_export_checks_class_shapes_and_values(self):
        x = np.linspace(0, 1, 1000)
        native = SimpleNamespace(names={0: "dechet"}, seg=SimpleNamespace(
            ap_class_index=np.array([0]), px=x, p_curve=x[None, :], r_curve=(1 - x)[None, :]))
        result = mask_confidence_curve(native)
        self.assertEqual(len(result["precision"]), 1000)
        self.assertEqual(result["iou_threshold"], 0.5)
        native.seg.r_curve[0, 5] = np.nan
        with self.assertRaisesRegex(RuntimeError, "invalid values"):
            mask_confidence_curve(native)


if __name__ == "__main__":
    unittest.main(verbosity=2)
