"""Configuration/provenance tests only: never train, evaluate or predict a model."""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

import evaluate_litter as evaluation
import train_litter as training


def args(root):
    return argparse.Namespace(data=root / "data.yaml", model=root / "yolo26n-seg.pt",
                              out=root / "new_run", epochs=12, device="auto", batch=2,
                              imgsz=640, split="val", conf=0.001, iou=0.7)


class ConfigTests(unittest.TestCase):
    def test_training_exact_requested_configuration(self):
        parameters = training.build_train_kwargs(args(Path("/tmp/litter_config_only")), "mps")
        expected = dict(epochs=12, device="mps", batch=2, imgsz=640, freeze=10,
                        optimizer="AdamW", lr0=0.0003, seed=42, workers=0, cache=False,
                        amp=False, mosaic=0.25, close_mosaic=2, degrees=3.0, translate=0.05,
                        scale=0.15, hsv_h=0.01, hsv_s=0.3, hsv_v=0.25, fliplr=0.5,
                        flipud=0.0, mixup=0.0, copy_paste=0.0, resume=False)
        for key, value in expected.items():
            self.assertEqual(parameters[key], value, key)
        self.assertEqual(parameters["name"], "new_run")
        self.assertEqual(parameters["split"], "val")

    def test_native_mask_evaluation_configuration(self):
        parameters = evaluation.build_eval_kwargs(args(Path("/tmp/litter_config_only")), "cpu")
        for key in ("save_json", "retina_masks", "plots"):
            self.assertTrue(parameters[key])
        self.assertFalse(parameters["augment"])
        self.assertEqual(parameters["split"], "val")
        self.assertEqual(parameters["conf"], 0.001)

    def test_invalid_configuration_is_rejected(self):
        sample = args(Path("/tmp/litter_config_only"))
        sample.batch = 0
        with self.assertRaises(ValueError):
            training.build_train_kwargs(sample, "cpu")
        sample.batch = 2
        sample.split = "train"
        with self.assertRaises(ValueError):
            evaluation.build_eval_kwargs(sample, "cpu")

    def test_mps_device_request_never_silently_falls_back(self):
        no_mps = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
        yes_mps = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
        self.assertEqual(training.choose_device("auto", no_mps), "cpu")
        with self.assertRaises(RuntimeError):
            training.choose_device("mps", no_mps)
        with patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}):
            self.assertEqual(training.choose_device("auto", yes_mps), "mps")
        with patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
            with self.assertRaises(RuntimeError):
                training.choose_device("mps", yes_mps)

    def test_native_oom_retry_guard_and_batch_change_detection(self):
        trainer = SimpleNamespace(batch_size=2, args=SimpleNamespace(batch=2), _oom_retries=0)
        training.guard_no_oom_retry(trainer, 2)
        self.assertEqual(trainer._oom_retries, 3)
        trainer.batch_size = 1
        with self.assertRaises(RuntimeError):
            training.guard_no_oom_retry(trainer, 2)

    def test_missing_evaluation_dependency_fails_without_installation(self):
        with patch.object(importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "No installation attempted"):
                evaluation.require_native_mask_json_dependency()

    def test_installed_native_argument_schema_accepts_both_configs(self):
        # Configuration parsing only: no model construction or inference.
        training.configure_local_runtime()
        from ultralytics.cfg import get_cfg
        values = args(Path("/tmp/litter_config_only"))
        self.assertEqual(get_cfg(overrides=training.build_train_kwargs(values, "cpu")).batch, 2)
        self.assertTrue(get_cfg(overrides=evaluation.build_eval_kwargs(values, "cpu")).save_json)
        proof = training.verify_oom_guard_contract()
        self.assertEqual(len(proof["trainer_source"]["sha256"]), 64)

    def test_existing_output_refused_before_any_model_or_data_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = args(root)
            values.out.mkdir()
            keep = values.out / "keep.txt"
            keep.write_text("existing output")
            with self.assertRaises(FileExistsError):
                training.run_training(values)
            with self.assertRaises(FileExistsError):
                evaluation.run_evaluation(values)
            self.assertEqual(keep.read_text(), "existing output")
            self.assertEqual(list(values.out.iterdir()), [keep])


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "images").mkdir()
        (self.root / "labels").mkdir()
        self.images = {}
        for number, split in enumerate(("train", "val", "test")):
            image = self.root / "images" / f"{split}.jpg"
            # These are deliberately just hash fixtures, never opened as images
            # and never fed to an ML model.
            image.write_bytes(f"hash-fixture-no-training-{number}".encode())
            self.images[split] = image
            (self.root / "labels" / f"{split}.txt").write_text("0 0.1 0.1 0.8 0.1 0.8 0.8\n")
            (self.root / f"{split}.txt").write_text(str(image) + "\n")
        self.config = {"path": str(self.root), "names": {0: "dechet"},
                       "train": "train.txt", "val": "val.txt", "test": "test.txt"}
        self.data = self.root / "data.yaml"
        self.save_config()

    def tearDown(self):
        self.temporary.cleanup()

    def save_config(self):
        self.data.write_text(yaml.safe_dump(self.config))

    def test_exact_files_hashes_counts_and_original_coco_evidence(self):
        coco = self.root / "COCO_ORIGINAL_ALL.json"
        coco.write_text('{"annotations": []}\n')
        proof = training.dataset_provenance(self.data)
        self.assertEqual(proof["names"], {0: "dechet"})
        self.assertEqual(proof["splits"]["train"]["instance_count"], 1)
        self.assertEqual(proof["splits"]["test"]["image_count"], 1)
        self.assertIn(training.fingerprint(coco), proof["source_evidence"])
        self.assertEqual(len(proof["dataset_sha256"]), 64)
        self.assertIn("not an evaluation", proof["ground_truth_note"])

    def test_mutated_label_changes_dataset_fingerprint(self):
        first = training.dataset_provenance(self.data)
        (self.root / "labels/train.txt").write_text("0 0.2 0.2 0.7 0.2 0.7 0.7\n")
        second = training.dataset_provenance(self.data)
        self.assertNotEqual(first["dataset_sha256"], second["dataset_sha256"])

    def test_empty_label_is_an_explicit_negative(self):
        (self.root / "labels/train.txt").write_text("")
        proof = training.dataset_provenance(self.data)
        self.assertEqual(proof["splits"]["train"]["instance_count"], 0)

    def test_missing_label_and_detection_box_are_rejected(self):
        label = self.root / "labels/train.txt"
        label.unlink()
        with self.assertRaisesRegex(ValueError, "Missing explicit label"):
            training.dataset_provenance(self.data)
        label.write_text("0 0.5 0.5 0.2 0.3\n")
        with self.assertRaisesRegex(ValueError, "not a box"):
            training.dataset_provenance(self.data)

    def test_image_content_leakage_across_splits_is_rejected(self):
        self.images["test"].write_bytes(self.images["train"].read_bytes())
        with self.assertRaisesRegex(ValueError, "Identical image content across"):
            training.dataset_provenance(self.data)

    def test_class_dictionary_and_relative_root_are_rejected(self):
        self.config["names"] = {0: "person"}
        self.save_config()
        with self.assertRaisesRegex(ValueError, "Expected exactly"):
            training.dataset_provenance(self.data)
        self.config["names"] = {0: "dechet"}
        self.config["path"] = "relative"
        self.save_config()
        with self.assertRaisesRegex(ValueError, "must be absolute"):
            training.dataset_provenance(self.data)

    def test_download_hook_is_refused(self):
        self.config["download"] = "https://invalid.example/data.zip"
        self.save_config()
        with self.assertRaisesRegex(ValueError, "download hooks"):
            training.dataset_provenance(self.data)

    def test_json_unknown_metric_remains_null_not_zero(self):
        path = self.root / "metric_fixture.json"
        training.write_json(path, {"metric": float("nan"), "value": 0.0})
        self.assertEqual(json.loads(path.read_text()), {"metric": None, "value": 0.0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
