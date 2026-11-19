"""Packaging tests with synthetic provenance, never real trained-model claims."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import package_gabarit_2d as package
from test_verify_gabarit_2d import make_delivery, dump
import verify_gabarit_2d as verifier


def make_package_fixture(root):
    root = root.resolve()
    run = make_delivery(root)
    manifest = verifier.read_json(run / "MANIFESTE.json")
    model = root / "source/synthetic_provenance_only.pt"
    manifest["parameters"]["litter_model"] = str(model)
    manifest["elapsed_s"] = 1.0
    with (run / "RAILS.csv").open() as handle:
        rail_rows = list(csv.DictReader(handle))
    audit_path = run / "AUDIT_FUSION.jsonl"
    audit = [json.loads(line) for line in audit_path.read_text().splitlines()]
    for record in audit:
        rows = [row for row in rail_rows if int(row["frame_index"]) == record["frame_index"]]
        corridor, _, _ = verifier.independent_corridor(rows, 32, 24)
        for obj in record["merged_predictions"]:
            x0, y0, x1, y1 = obj["mask_bbox"]
            obj["bbox_overlap_pixel_count"] = int(corridor[y0:y1, x0:x1].sum())
    audit_path.write_text("".join(json.dumps(record) + "\n" for record in audit))
    manifest["output_sha256"]["AUDIT_FUSION.jsonl"] = verifier.sha(audit_path)
    dump(run / "MANIFESTE.json", manifest)
    verifier.verify_delivery(run)
    training, evaluation, code = root / "training", root / "evaluation", root / "code"
    for path in (training, evaluation, code):
        path.mkdir()
    snapshot = {"splits": {split: {"image_count": 1, "instance_count": 1} for split in ("train", "val", "test")}}
    dump(training / "DATASET_SNAPSHOT.json", snapshot)
    train = {"status": "complete", "completed_epochs": 1, "dataset_sha256": "synthetic-fixture",
             "checkpoints": {"best": {"sha256": verifier.sha(model)}},
             "dataset_snapshot": {"sha256": verifier.sha(training / "DATASET_SNAPSHOT.json")}}
    dump(training / "TRAINING_MANIFEST.json", train)
    metrics = {"status": "complete", "split": "test", "image_count": 1,
               "checkpoint_sha256": verifier.sha(model), "dataset_sha256": "synthetic-fixture",
               "mask_metrics": {"metrics/mAP50(M)": 0.4, "metrics/mAP50-95(M)": 0.2},
               "scope": "Synthetic packaging fixture only; no training or model evaluation occurred."}
    dump(evaluation / "METRICS.json", metrics)
    dump(evaluation / "EVALUATION_MANIFEST.json", {"status": "complete", "metrics_json": {"sha256": verifier.sha(evaluation / "METRICS.json")}})
    notice = root / "notice.txt"
    notice.write_text("Synthetic packaging fixture notice.\n")
    for name in ("lire_contours.py", "corridor_geometry.py"):
        (code / name).write_text("# Synthetic packaging fixture, never executed.\n")
    out = root / "package.zip"
    args = ["--run", str(run), "--training", str(training), "--evaluation", str(evaluation),
            "--out", str(out), "--notice", str(notice), "--code", str(code)]
    return run, out, args


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.run, self.out, self.args = make_package_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def assert_clean_failure(self):
        self.assertFalse(self.out.exists())
        self.assertFalse(self.out.with_suffix(".verification.json").exists())
        self.assertFalse((self.run / "STATISTIQUES.json").exists())
        self.assertFalse((self.run / "RESULTATS_OBTENUS.txt").exists())
        self.assertEqual(list(self.root.glob(".gabarit_package_*")), [])

    def test_valid_package_matches_exact_verified_txt_set_and_contents(self):
        package.main(self.args)
        proof = verifier.read_json(self.out.with_suffix(".verification.json"))
        prior = verifier.read_json(self.run / "VERIFICATION.json")
        self.assertEqual(proof["status"], "passed")
        self.assertEqual(proof["txt_files_checked"], 3)
        self.assertEqual(proof["txt_path_and_content_aggregate_sha256"], prior["txt_path_and_content_aggregate_sha256"])
        with zipfile.ZipFile(self.out) as archive:
            names = [name for name in archive.namelist() if "/TXT/" in name]
            self.assertEqual(len(names), 3)
            for name in names:
                self.assertEqual(archive.read(name), (self.run / "TXT" / Path(name).name).read_bytes())

    def test_modified_or_missing_txt_is_rejected_before_any_publication(self):
        path = self.run / "TXT/frame_000001.txt"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "TXT contents changed"):
            package.main(self.args)
        self.assert_clean_failure()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "TXT set differs"):
            package.main(self.args)
        self.assert_clean_failure()

    def test_missing_reader_is_rejected_before_run_reports_are_written(self):
        (self.root / "code/lire_contours.py").unlink()
        with self.assertRaises(FileNotFoundError):
            package.main(self.args)
        self.assert_clean_failure()

    def test_no_space_failure_is_clean_and_same_command_can_retry(self):
        with patch.object(package.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaises(OSError):
                package.main(self.args)
        self.assert_clean_failure()
        package.main(self.args)
        self.assertTrue(self.out.exists())

    def test_zip_failure_is_clean_and_same_command_can_retry(self):
        with patch.object(package.zipfile, "ZipFile", side_effect=OSError("synthetic disk error")):
            with self.assertRaisesRegex(OSError, "synthetic disk error"):
                package.main(self.args)
        self.assert_clean_failure()
        package.main(self.args)
        self.assertTrue(self.out.exists())

    def test_publication_failure_rolls_back_only_created_files(self):
        actual = package.publish_exclusive
        calls = []
        def fail_second(source, destination, created):
            calls.append(destination)
            if len(calls) == 2:
                raise OSError("synthetic publication failure")
            return actual(source, destination, created)
        with patch.object(package, "publish_exclusive", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "synthetic publication failure"):
                package.main(self.args)
        self.assert_clean_failure()
        package.main(self.args)
        self.assertTrue(self.out.exists())

    def test_txt_modified_after_preflight_cannot_enter_verified_archive(self):
        actual = package.verified_txt_files
        def change_after_freeze(*args):
            result = actual(*args)
            path = self.run / "TXT/frame_000001.txt"
            path.write_bytes(path.read_bytes() + b" ")
            return result
        with patch.object(package, "verified_txt_files", side_effect=change_after_freeze):
            with self.assertRaisesRegex(ValueError, "Package hash mismatch"):
                package.main(self.args)
        self.assert_clean_failure()


if __name__ == "__main__":
    unittest.main(verbosity=2)
