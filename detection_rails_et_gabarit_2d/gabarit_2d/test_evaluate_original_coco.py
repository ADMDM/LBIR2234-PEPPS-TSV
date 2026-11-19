"""CPU-only oracle/contract tests. Synthetic scores are not model performance."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import evaluate_original_coco as scoring


def mapping_row():
    return {"image_id": 7, "file_name": "batch_val/000.jpg", "split": "val", "batch": "batch_val",
            "status": "ok", "download_kind": "original", "source_width": 8, "source_height": 8,
            "decoded_width": 8, "decoded_height": 8, "prepared_width": 4, "prepared_height": 4,
            "exif_orientation": 1, "orientation_action": "raw_pixel_grid_orientation_1",
            "image_transform": "RGB_JPEG_quality95_maxside1280_no_upscale", "scale_x": 0.5, "scale_y": 0.5,
            "prepared_image": "/synthetic/images/taco_00007.jpg"}


def original_gt():
    return {"info": {"description": "synthetic geometry oracle, no model"},
            "images": [{"id": 7, "file_name": "batch_val/000.jpg", "width": 8, "height": 8}],
            "annotations": [{"id": 11, "image_id": 7, "category_id": 9, "iscrowd": 0,
                             "bbox": [2, 2, 4, 4], "area": 16,
                             "segmentation": [[2, 2, 6, 2, 6, 6, 2, 6]]}],
            "categories": [{"id": 9, "name": "synthetic_packaging"}]}


def prediction(mask=None):
    if mask is None:
        mask = np.zeros((4, 4), np.uint8)
        mask[1:3, 1:3] = 1
    return {"image_id": "taco_00007", "file_name": "taco_00007.jpg", "category_id": 1,
            "score": 0.9, "bbox": [1, 1, 2, 2], "segmentation": scoring.encode_mask(mask)}


def make_fixture(root):
    root = root.resolve()
    dataset, evaluation = root / "dataset", root / "evaluation"
    for path in (dataset / "images", dataset / "labels", evaluation, root / "sources/images/batch_val"):
        path.mkdir(parents=True)
    row, coco = mapping_row(), original_gt()
    raw = root / "sources/images/batch_val/000.jpg"
    prepared, label = dataset / "images/taco_00007.jpg", dataset / "labels/taco_00007.txt"
    Image.new("RGB", (8, 8), (40, 90, 150)).save(raw, quality=95)
    Image.new("RGB", (4, 4), (40, 90, 150)).save(prepared, quality=95)
    label.write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")
    row.update(path=str(raw), prepared_image=str(prepared), prepared_label=str(label),
               sha256=scoring.sha(raw), prepared_image_sha256=scoring.sha(prepared))
    original_file = root / "sources/annotations.json"
    scoring.write_json(original_file, coco)
    scoring.write_json(dataset / "COCO_ORIGINAL_val.json", coco)
    scoring.write_json(dataset / "manifest_val.json", [row])
    scoring.write_json(dataset / "MANIFEST_ALL.json", [row])
    scoring.write_json(dataset / "PROVENANCE.json", {"preserved_source_annotations_sha256": scoring.sha(original_file)})
    scoring.write_json(dataset / "VERIFICATION.json", {"passed": True, "scope": "synthetic fixture only"})
    snapshot = {"root": str(dataset), "names": {"0": "dechet"}, "dataset_sha256": "a" * 64,
                "source_evidence": [scoring.fingerprint(dataset / name) for name in ("COCO_ORIGINAL_val.json", "manifest_val.json", "MANIFEST_ALL.json")],
                "splits": {"val": {"image_count": 1, "pairs": [{"image": scoring.fingerprint(prepared), "label": scoring.fingerprint(label)}]}}}
    scoring.write_json(evaluation / "DATASET_SNAPSHOT.json", snapshot)
    scoring.write_json(evaluation / "predictions.json", [prediction()])
    metrics = {"status": "complete", "split": "val", "image_count": 1, "prediction_count": 1,
               "dataset_sha256": snapshot["dataset_sha256"], "checkpoint_sha256": "b" * 64,
               "predictions_json": scoring.fingerprint(evaluation / "predictions.json"),
               "scope": "synthetic oracle, no model or evaluation of a trained model"}
    scoring.write_json(evaluation / "METRICS.json", metrics)
    manifest = {"status": "complete", "split": "val", "image_count": 1, "prediction_count": 1,
                "checkpoint": {"sha256": "b" * 64},
                "dataset_sha256": snapshot["dataset_sha256"],
                "metrics_json": scoring.fingerprint(evaluation / "METRICS.json"),
                "predictions_json": scoring.fingerprint(evaluation / "predictions.json"),
                "dataset_snapshot": scoring.fingerprint(evaluation / "DATASET_SNAPSHOT.json")}
    scoring.write_json(evaluation / "EVALUATION_MANIFEST.json", manifest)
    return dataset, evaluation


class GeometryOracleTests(unittest.TestCase):
    def test_original_polygon_oracle_scores_perfect_after_exact_aligned_resize(self):
        row, original = mapping_row(), original_gt()
        mapping = scoring.build_mapping([row], original, "val")
        restored, audit = scoring.restore_prediction(prediction(), mapping)
        gt, grouped = scoring.single_class_ground_truth(original)
        expected = scoring.mask_api.decode(gt.annToRLE(gt.anns[11]))
        actual = scoring.mask_api.decode(restored["segmentation"])
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(audit["source_mask_pixel_count"], 16)
        self.assertNotIn("bbox", restored)
        self.assertEqual(grouped["annotations"][0]["segmentation"], original["annotations"][0]["segmentation"])
        metrics = scoring.compute_metrics(gt, [restored])
        self.assertAlmostEqual(metrics["segmentation_metrics"]["AP_IoU_0.50_0.95"], 1.0)

    def test_multipart_ground_truth_is_canonical_union_without_bridge(self):
        original = original_gt()
        polygons = [[0, 0, 2, 0, 2, 2, 0, 2], [6, 6, 8, 6, 8, 8, 6, 8]]
        original["annotations"][0].update(segmentation=polygons, area=8, bbox=[0, 0, 8, 8])
        coco, grouped = scoring.single_class_ground_truth(original)
        mask = scoring.mask_api.decode(coco.annToRLE(coco.anns[11]))
        self.assertEqual(mask.sum(), 8)
        self.assertFalse(mask[2:6, 2:6].any())
        self.assertEqual(grouped["annotations"][0]["segmentation"], polygons)
        self.assertEqual(original["annotations"][0]["category_id"], 9)
        self.assertEqual(grouped["annotations"][0]["category_id"], 1)

    def test_exif_six_uses_annotation_grid_and_does_not_rotate_back(self):
        row = mapping_row()
        row.update(source_width=4, source_height=8, decoded_width=8, decoded_height=4,
                   prepared_width=2, prepared_height=4, exif_orientation=6,
                   orientation_action="exif_transpose_to_annotation_display_grid")
        original = original_gt()
        original["images"][0].update(width=4, height=8)
        mapping = scoring.build_mapping([row], original, "val")
        small = np.zeros((4, 2), np.uint8)
        small[0, 0] = 1
        pred = prediction(small)
        pred["bbox"] = [0, 0, 1, 1]
        restored, audit = scoring.restore_prediction(pred, mapping)
        mask = scoring.mask_api.decode(restored["segmentation"])
        self.assertEqual(mask.shape, (8, 4))
        self.assertEqual(mask[:2, :2].sum(), 4)
        self.assertEqual(mask.sum(), 4)
        self.assertFalse(audit["inverse_exif_applied"])

    def test_unknown_or_ambiguous_orientation_is_rejected(self):
        row, original = mapping_row(), original_gt()
        row["orientation_action"] = "rotated_somehow"
        with self.assertRaisesRegex(ValueError, "Unknown orientation"):
            scoring.build_mapping([row], original, "val")

    def test_unknown_image_id_and_filename_mismatch_are_rejected(self):
        mapping = scoring.build_mapping([mapping_row()], original_gt(), "val")
        for key in (7, "taco_99999"):
            item = prediction()
            item["image_id"] = key
            with self.assertRaisesRegex(ValueError, "Unknown prediction image_id"):
                scoring.restore_prediction(item, mapping)
        item = prediction()
        item["file_name"] = "wrong.jpg"
        with self.assertRaisesRegex(ValueError, "filename/ID"):
            scoring.restore_prediction(item, mapping)

    def test_wrong_mask_shape_bad_coordinates_and_wrong_category_fail(self):
        mapping = scoring.build_mapping([mapping_row()], original_gt(), "val")
        item = prediction(np.zeros((3, 4), np.uint8))
        with self.assertRaisesRegex(ValueError, "RLE shape"):
            scoring.restore_prediction(item, mapping)
        for bbox in ([-1, 0, 1, 1], [3, 0, 2, 1], [float("nan"), 0, 1, 1]):
            item = prediction()
            item["bbox"] = bbox
            with self.assertRaisesRegex(ValueError, "bbox outside"):
                scoring.restore_prediction(item, mapping)
        item = prediction()
        item["category_id"] = 0
        with self.assertRaisesRegex(ValueError, "category_id=1"):
            scoring.restore_prediction(item, mapping)

    def test_empty_predictions_give_zero_recall_without_loading_a_model(self):
        gt, _ = scoring.single_class_ground_truth(original_gt())
        metrics = scoring.compute_metrics(gt, [])
        self.assertEqual(metrics["segmentation_metrics"]["AR_max100"], 0)

    def test_nonfinite_ground_truth_geometry_is_rejected(self):
        original = original_gt()
        original["annotations"][0]["segmentation"][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite original polygon"):
            scoring.single_class_ground_truth(original)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset, self.evaluation = make_fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_full_cpu_oracle_run_preserves_every_source_fingerprint(self):
        before = {str(path): scoring.sha(path) for folder in (self.dataset, self.evaluation, self.root / "sources")
                  for path in folder.rglob("*") if path.is_file()}
        output = self.root / "new_coco_results"
        result = scoring.run(self.dataset, self.evaluation, output)
        self.assertAlmostEqual(result["segmentation_metrics"]["AP_IoU_0.50_0.95"], 1)
        self.assertTrue(result["original_coco_mask_evaluation"])
        self.assertFalse(scoring.read_json(output / "MANIFEST.json")["model_loaded"])
        self.assertEqual(before, {path: scoring.sha(path) for path in before})
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            scoring.run(self.dataset, self.evaluation, output)

    def test_changed_original_image_is_rejected_before_scoring(self):
        row = scoring.read_json(self.dataset / "manifest_val.json")[0]
        Path(row["path"]).write_bytes(b"changed source fixture")
        with self.assertRaisesRegex(ValueError, "Source fingerprint mismatch"):
            scoring.preflight(self.dataset, self.evaluation)

    def test_changed_original_polygon_file_is_rejected(self):
        path = self.dataset / "COCO_ORIGINAL_val.json"
        document = scoring.read_json(path)
        document["annotations"][0]["segmentation"][0][0] = 3
        scoring.write_json(path, document)
        with self.assertRaisesRegex(ValueError, "not frozen in evaluation"):
            scoring.preflight(self.dataset, self.evaluation)

    def test_other_split_evaluation_is_rejected(self):
        path = self.evaluation / "METRICS.json"
        document = scoring.read_json(path)
        document["split"] = "test"
        scoring.write_json(path, document)
        manifest_path = self.evaluation / "EVALUATION_MANIFEST.json"
        manifest = scoring.read_json(manifest_path)
        manifest["metrics_json"] = scoring.fingerprint(path)
        scoring.write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "inconsistent evaluation split"):
            scoring.preflight(self.dataset, self.evaluation)


if __name__ == "__main__":
    unittest.main(verbosity=2)
