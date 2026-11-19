#!/usr/bin/env python3
"""Select resolution and a confidence threshold using ONLY two frozen val runs.

960 is selected only for >=0.005 absolute gain in mask mAP50-95 over 640.
At the selected resolution, maximize interpolated mask-IoU50 F2 over the fixed
candidate list; ties select the higher threshold. No model/inference/GPU access.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CANDIDATES = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
EPSILON = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_curve(curve):
    require(curve["class_id"] == 0 and curve["class_name"] == "dechet" and curve["iou_threshold"] == 0.5,
            "Expected dechet class-0 mask IoU50 confidence curve")
    arrays = [np.asarray(curve[key], dtype=float) for key in ("confidence_thresholds", "precision", "recall")]
    x, p, r = arrays
    require(x.ndim == 1 and len(x) >= 2 and p.shape == x.shape == r.shape, "Curve shape mismatch")
    require(all(np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all() for a in arrays), "Curve values must be finite fractions")
    require((np.diff(x) > 0).all() and x[0] <= min(CANDIDATES) and x[-1] >= max(CANDIDATES), "Curve cannot interpolate all fixed thresholds")
    return x, p, r


def load_validation(folder, expected_imgsz):
    folder = Path(folder).resolve(strict=True)
    metrics_path, manifest_path = folder / "METRICS.json", folder / "EVALUATION_MANIFEST.json"
    metrics, manifest = read(metrics_path), read(manifest_path)
    require(metrics.get("status") == manifest.get("status") == "complete", "Both validation runs must be complete")
    require(metrics.get("split") == manifest.get("split") == "val", "Selection accepts val only; test data are forbidden")
    require(sha(metrics_path) == manifest["metrics_json"]["sha256"], "Validation metrics fingerprint mismatch")
    require(metrics["checkpoint_sha256"] == manifest["checkpoint"]["sha256"], "Checkpoint fingerprint mismatch")
    require(metrics["dataset_sha256"] == manifest["dataset_sha256"], "Dataset fingerprint mismatch")
    require(metrics["original_coco_mask_evaluation"] is False, "Selection requires the fixed native YOLO-label protocol")
    require(metrics.get("segmentation_backend_end2end") is False, "Conventional NMS backend was not verified")
    args = manifest["evaluation_arguments"]
    require(args["imgsz"] == expected_imgsz and args["split"] == "val", "Wrong validation resolution or split")
    require(args.get("nms") is True and args["iou"] == 0.5 and args["conf"] == 0.001,
            "Required protocol is classical NMS, IoU=.5 and conf=.001")
    require(args['max_det']==100, 'Evaluation must use the same 100-instance cap as video inference')
    require(args["retina_masks"] is True and args["augment"] is False, "Unexpected mask/postprocessing protocol")
    resolved_path = folder / "RESOLVED_EVAL_ARGS.json"
    entries = [entry for entry in manifest["artifacts"] if Path(entry["path"]).name == resolved_path.name]
    require(len(entries) == 1 and entries[0]["sha256"] == sha(resolved_path), "Resolved settings are not frozen in the completed manifest")
    resolved = read(resolved_path)
    for key in ("imgsz", "split", "nms", "iou", "conf", "max_det", "retina_masks", "augment"):
        require(resolved[key] == args[key], f"Resolved validation setting differs: {key}")
    score = float(metrics["mask_metrics"]["metrics/mAP50-95(M)"])
    require(math.isfinite(score) and 0 <= score <= 1, "Invalid mask mAP50-95")
    validate_curve(metrics["mask_confidence_curve_iou50"])
    return {"folder": str(folder), "metrics": metrics, "manifest": manifest, "resolved": resolved,
            "map50_95": score, "imgsz": expected_imgsz,
            "files": [{"path": str(path), "sha256": sha(path)} for path in (metrics_path, manifest_path, resolved_path)]}


def threshold_table(curve):
    x, p, r = validate_curve(curve)
    rows = []
    for threshold in CANDIDATES:
        precision, recall = float(np.interp(threshold, x, p)), float(np.interp(threshold, x, r))
        denominator = 4 * precision + recall
        f2 = 5 * precision * recall / denominator if denominator else 0.0
        rows.append({"confidence_threshold": threshold, "mask_precision_iou50": precision,
                     "mask_recall_iou50": recall, "F2": f2})
    return rows


def choose(first, second):
    a, b = first["metrics"], second["metrics"]
    require(a["checkpoint_sha256"] == b["checkpoint_sha256"], "Validation checkpoints differ")
    require(a["dataset_sha256"] == b["dataset_sha256"] and a["image_count"] == b["image_count"]
            and a["ground_truth_instances"] == b["ground_truth_instances"], "Validation datasets/ground truth differ")
    ignored = {"imgsz", "project", "name", "save_dir"}
    for source in ("evaluation_arguments",):
        left = {k: v for k, v in first["manifest"][source].items() if k not in ignored}
        right = {k: v for k, v in second["manifest"][source].items() if k not in ignored}
        require(left == right, "Validation settings differ beyond resolution/output location")
    require({k: v for k, v in first["resolved"].items() if k not in ignored} ==
            {k: v for k, v in second["resolved"].items() if k not in ignored}, "Resolved validation settings differ")
    for key in ("torch", "ultralytics", "resolved_device"):
        require(first["manifest"]["runtime"][key] == second["manifest"]["runtime"][key], f"Validation runtime differs: {key}")
    delta = second["map50_95"] - first["map50_95"]
    selected = second if delta >= 0.005 - EPSILON else first
    table = threshold_table(selected["metrics"]["mask_confidence_curve_iou50"])
    best = max(row["F2"] for row in table)
    chosen = max((row for row in table if abs(row["F2"] - best) <= EPSILON), key=lambda row: row["confidence_threshold"])
    return {"status": "selected_on_validation", "split": "val", "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": a["checkpoint_sha256"], "dataset_sha256": a["dataset_sha256"],
            "selection_protocol": {"resolution_rule": "choose 960 only when mask mAP50-95 gain over 640 is >=0.005 absolute",
                                   "confidence_candidates": CANDIDATES, "threshold_rule": "maximize mask-IoU50 F2; ties choose higher confidence",
                                   "F2_formula": "5*precision*recall/(4*precision+recall)", "beta": 2,
                                   "interpolation": "linear on exported native precision/recall confidence curves",
                                   "numeric_tie_tolerance": EPSILON, "native_conf": 0.001, "nms_iou": 0.5},
            "resolution_comparison": [{"imgsz": item["imgsz"], "mask_mAP50_95": item["map50_95"], "selected": item is selected}
                                      for item in (first, second)],
            "mask_mAP50_95_gain_960_minus_640": delta,
            "threshold_comparison_on_selected_validation": table,
            "selected": {"imgsz": selected["imgsz"], **chosen},
            "source_files": first["files"] + second["files"],
            "test_data_read": False, "new_model_inference": False,
            "scope": "Resolution and F2 threshold selected on validation only against converted YOLO masks. These are selection results, not test or TSV accuracy.",
            "limitation": "Interpolated native IoU50 curve estimates; the selected threshold has not been rerun here. F2 emphasizes recall and is not a physical-risk metric."}


def run(val640, val960, out):
    out = Path(out).resolve()
    require(not out.exists(), f"Output already exists; refusing overwrite: {out}")
    result = choose(load_validation(val640, 640), load_validation(val960, 960))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val640", type=Path, required=True)
    parser.add_argument("--val960", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.val640, args.val960, args.out)
    print(json.dumps(result["selected"]), flush=True)


if __name__ == "__main__":
    main()
