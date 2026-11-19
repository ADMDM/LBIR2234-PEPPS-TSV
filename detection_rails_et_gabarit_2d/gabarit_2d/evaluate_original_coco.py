#!/usr/bin/env python3
"""CPU-only COCO scoring of existing litter RLE predictions against original GT.

No model is loaded and no new inference is performed. Prepared predictions are
enlarged with Pillow nearest-neighbour to the ORIGINAL ANNOTATION DISPLAY GRID.
Preparation already applied EXIF into that grid: no inverse rotation is valid.
Only category IDs are collapsed to dechet=1; original polygons, component lists,
coordinates, image/annotation IDs, area metadata and crowd flags are preserved.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image
import PIL
import faster_coco_eval
from faster_coco_eval import COCO, COCOeval_faster, mask as mask_api

METRIC_NAMES = ("AP_IoU_0.50_0.95", "AP_IoU_0.50", "AP_IoU_0.75", "AP_small", "AP_medium", "AP_large",
                "AR_max1", "AR_max10", "AR_max100", "AR_small", "AR_medium", "AR_large",
                "AR_IoU_0.50", "AR_IoU_0.75")
RESIZE_NOTE = ("Binary prediction masks at prepared-image resolution are enlarged to source annotation dimensions "
               "with Pillow nearest-neighbour. This cannot recover detail lost during preparation or prediction; "
               "scores include that resampling approximation. EXIF was already applied during preparation, "
               "so no inverse EXIF rotation or mirroring is applied.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON value {x}")))


def write_json(path, value):
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), f"Expected file: {path}")
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def check_fingerprint(path, expected):
    proof = fingerprint(path)
    require(proof["sha256"] == expected, f"Source fingerprint mismatch: {path}")
    return proof


def integer(value, label, minimum=0):
    require(not isinstance(value, bool), f"Invalid {label}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {label}") from exc
    require(math.isfinite(number) and number.is_integer() and number >= minimum, f"Invalid {label}")
    return int(number)


def validate_mapping_row(row, image, split):
    """Reject uncertain transforms; source dimensions are the annotation grid."""
    require(row["split"] == split and row["status"] == "ok" and row["download_kind"] == "original", "Wrong split or unverified original image")
    require(integer(row["image_id"], "image_id") == integer(image["id"], "COCO image ID"), "Image ID mismatch")
    sw, sh, pw, ph = [integer(row[k], k, 1) for k in ("source_width", "source_height", "prepared_width", "prepared_height")]
    require((sw, sh) == (image["width"], image["height"]) and row["file_name"] == image["file_name"], "Original COCO image metadata mismatch")
    require(row["image_transform"] == "RGB_JPEG_quality95_maxside1280_no_upscale", "Unknown image transform; no inverse invented")
    require(pw <= sw and ph <= sh and max(pw, ph) <= 1280, "Unexpected prepared dimensions")
    require(abs(float(row["scale_x"]) - pw / sw) < 1e-12 and abs(float(row["scale_y"]) - ph / sh) < 1e-12, "Scale factors disagree with manifest dimensions")
    orientation = integer(row["exif_orientation"], "EXIF orientation")
    action = row["orientation_action"]
    if orientation in (2, 3, 4, 5, 6, 7, 8):
        require(action == "exif_transpose_to_annotation_display_grid", "Ambiguous EXIF transform")
        expected_raw = (sh, sw) if orientation in (5, 6, 7, 8) else (sw, sh)
    elif orientation == 0:
        require(action == "raw_pixel_grid; EXIF_orientation_0_ignored", "Unrecognized EXIF=0 handling")
        expected_raw = (sw, sh)
    else:
        require(orientation == 1 and action == "raw_pixel_grid_orientation_1", "Unknown orientation; no inverse invented")
        expected_raw = (sw, sh)
    require((row["decoded_width"], row["decoded_height"]) == expected_raw, "Decoded dimensions do not support recorded EXIF transform")
    return {"source_width": sw, "source_height": sh, "prepared_width": pw, "prepared_height": ph,
            "annotation_grid": "original_EXIF_oriented_display_grid", "inverse_exif_applied": False}


def prepared_prediction_key(path):
    stem = Path(path).stem
    return int(stem) if stem.isnumeric() else stem


def build_mapping(rows, coco, split):
    images = {integer(im["id"], "COCO image ID"): im for im in coco["images"]}
    require(len(images) == len(coco["images"]), "Duplicate COCO image IDs")
    require(len(rows) == len(images), "Split manifest and COCO image counts differ")
    mapping, ids, paths = {}, set(), set()
    for row in rows:
        identifier = integer(row["image_id"], "manifest image ID")
        require(identifier in images and identifier not in ids, "Unknown or repeated source image ID")
        validate_mapping_row(row, images[identifier], split)
        path = Path(row["prepared_image"]).resolve()
        require(path not in paths, "Duplicate prepared image path")
        key = prepared_prediction_key(path)
        require(key not in mapping, "Ambiguous prepared filename ID")
        mapping[key] = row
        ids.add(identifier)
        paths.add(path)
    require(ids == set(images), "COCO image set differs from exact split")
    return mapping


def encode_mask(binary):
    rle = mask_api.encode(np.asfortranarray(binary, dtype=np.uint8))
    require(isinstance(rle, dict), "Expected a single binary mask RLE")
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    rle["size"] = [int(v) for v in rle["size"]]
    return rle


def restore_prediction(prediction, mapping):
    """Map exact prepared filename ID and NN-resize one compressed RLE mask."""
    key = prediction.get("image_id")
    require(isinstance(key, (str, int)) and not isinstance(key, bool) and key in mapping, f"Unknown prediction image_id: {key!r}")
    row = mapping[key]
    require(prediction.get("file_name") == Path(row["prepared_image"]).name, "Prediction filename/ID mapping mismatch")
    require(integer(prediction.get("category_id"), "prediction category") == 1, "Expected single-class native JSON category_id=1")
    score = float(prediction.get("score"))
    require(math.isfinite(score) and 0 <= score <= 1, "Invalid prediction score")
    pw, ph = row["prepared_width"], row["prepared_height"]
    bbox = prediction.get("bbox")
    require(isinstance(bbox, list) and len(bbox) == 4, "Expected prepared-space xywh bbox")
    x, y, w, h = [float(v) for v in bbox]
    require(all(math.isfinite(v) for v in (x, y, w, h)) and min(x, y, w, h) >= 0
            and x + w <= pw + 0.002 and y + h <= ph + 0.002, "Prediction bbox outside prepared image")
    rle = prediction.get("segmentation")
    require(isinstance(rle, dict) and rle.get("size") == [ph, pw] and isinstance(rle.get("counts"), str), "Prediction RLE shape/encoding differs from prepared image")
    decoded = mask_api.decode(rle)
    require(decoded.shape == (ph, pw) and np.isin(decoded, (0, 1)).all(), "Invalid decoded prepared binary mask")
    source = np.asarray(Image.fromarray(decoded.astype(np.uint8)).resize(
        (row["source_width"], row["source_height"]), resample=Image.Resampling.NEAREST), dtype=np.uint8)
    require(source.shape == (row["source_height"], row["source_width"]) and np.isin(source, (0, 1)).all(), "Invalid resized source mask")
    # Deliberately omit bbox: COCO.loadRes must derive area and bbox from the
    # restored segmentation, not take a detection-box area for mask evaluation.
    result = {"image_id": row["image_id"], "category_id": 1, "score": score, "segmentation": encode_mask(source)}
    audit = {"prepared_image_id": key, "source_image_id": row["image_id"],
             "prepared_size_hw": [ph, pw], "source_size_hw": list(source.shape),
             "prepared_mask_pixel_count": int(decoded.sum()), "source_mask_pixel_count": int(source.sum()),
             "resized": source.shape != decoded.shape, "inverse_exif_applied": False}
    return result, audit


def single_class_ground_truth(original):
    """Preserve GT geometry; canonical COCO annToRLE unions multipart polygons."""
    result = deepcopy(original)
    result["categories"] = [{"id": 1, "name": "dechet", "supercategory": "dechet"}]
    image_ids = {im["id"] for im in result["images"]}
    category_ids = {cat["id"] for cat in original["categories"]}
    annotation_ids = set()
    for ann in result["annotations"]:
        require(ann["id"] not in annotation_ids, "Duplicate annotation ID")
        annotation_ids.add(ann["id"])
        require(ann["image_id"] in image_ids and ann["category_id"] in category_ids, "Annotation outside split or unknown category")
        require(ann.get("iscrowd", 0) in (0, 1), "Invalid crowd flag")
        polygons = ann["segmentation"]
        require(isinstance(polygons, list) and polygons, "Original polygon component list required")
        for polygon in polygons:
            require(isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0,
                    "Invalid original polygon structure")
            require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in polygon),
                    "Nonfinite original polygon coordinate")
        # Original COCO coordinates are not clipped or repaired here. The
        # canonical rasterizer handles their image-boundary intersections.
        ann["category_id"] = 1
    coco = COCO()
    coco.dataset = result
    coco.createIndex()
    for ann in result["annotations"]:
        require(float(mask_api.area(coco.annToRLE(ann))) > 0, "Original polygon rasterizes to an empty mask")
    return coco, result


def compute_metrics(ground_truth, predictions):
    detections = ground_truth.loadRes(deepcopy(predictions))
    evaluator = COCOeval_faster(ground_truth, detections, iouType="segm", print_function=print,
                               boundary_cpu_count=1, rle_iou_max_workers=1)
    evaluator.params.imgIds = sorted(ground_truth.getImgIds())
    evaluator.params.catIds = [1]
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    values = [float(v) for v in evaluator.stats]
    require(len(values) >= 12 and all(math.isfinite(v) for v in values), "COCO metrics unavailable or nonfinite")
    return {"segmentation_metrics": {name: (None if value < 0 else value) for name, value in zip(METRIC_NAMES, values)},
            "raw_COCO_stats": values, "undefined_metric_convention": "COCO -1 becomes null in named metrics",
            "iou_thresholds": [float(v) for v in evaluator.params.iouThrs],
            "max_detections_per_image": [1, 10, 100], "category_ids": [1],
            "area_ranges_original_pixels_squared": [[float(v) for v in area] for area in evaluator.params.areaRng],
            "area_range_labels": list(evaluator.params.areaRngLbl)}


def preflight(dataset, evaluation):
    dataset, evaluation = Path(dataset).resolve(strict=True), Path(evaluation).resolve(strict=True)
    metrics_path, eval_manifest_path = evaluation / "METRICS.json", evaluation / "EVALUATION_MANIFEST.json"
    metrics, evaluation_manifest = read_json(metrics_path), read_json(eval_manifest_path)
    require(metrics["status"] == evaluation_manifest["status"] == "complete", "A complete existing evaluation is required")
    require(metrics["checkpoint_sha256"] == evaluation_manifest["checkpoint"]["sha256"], "Evaluation checkpoint fingerprint mismatch")
    split = metrics["split"]
    require(split in ("val", "test") and evaluation_manifest["split"] == split, "Invalid or inconsistent evaluation split")
    frozen = [check_fingerprint(metrics_path, evaluation_manifest["metrics_json"]["sha256"])]
    prediction_path, snapshot_path = evaluation / "predictions.json", evaluation / "DATASET_SNAPSHOT.json"
    frozen.append(check_fingerprint(prediction_path, evaluation_manifest["predictions_json"]["sha256"]))
    require(metrics["predictions_json"]["sha256"] == frozen[-1]["sha256"], "Prediction fingerprint differs between metrics and manifest")
    frozen.append(check_fingerprint(snapshot_path, evaluation_manifest["dataset_snapshot"]["sha256"]))
    snapshot = read_json(snapshot_path)
    require(metrics["dataset_sha256"] == evaluation_manifest["dataset_sha256"] == snapshot["dataset_sha256"], "Evaluation dataset identity mismatch")
    require(Path(snapshot["root"]).resolve() == dataset, "Evaluation belongs to another dataset root")
    require(snapshot["names"] == {"0": "dechet"}, "Expected one-class litter evaluation")
    paths = {name: dataset / name for name in (f"manifest_{split}.json", f"COCO_ORIGINAL_{split}.json", "MANIFEST_ALL.json", "PROVENANCE.json", "VERIFICATION.json")}
    for path in paths.values():
        frozen.append(fingerprint(path))
    frozen.append(fingerprint(eval_manifest_path))
    evidence = {str(Path(item["path"]).resolve()): item["sha256"] for item in snapshot["source_evidence"]}
    for name in (f"manifest_{split}.json", f"COCO_ORIGINAL_{split}.json", "MANIFEST_ALL.json"):
        path = paths[name]
        require(evidence.get(str(path)) == sha(path), f"Original source mapping/annotations not frozen in evaluation: {name}")
    require(read_json(paths["VERIFICATION.json"])["passed"] is True, "Prepared dataset did not pass verification")
    rows, original = read_json(paths[f"manifest_{split}.json"]), read_json(paths[f"COCO_ORIGINAL_{split}.json"])
    mapping = build_mapping(rows, original, split)
    all_rows = read_json(paths["MANIFEST_ALL.json"])
    require(len({row["image_id"] for row in all_rows}) == len(all_rows), "Source image ID occurs in multiple prepared splits")
    all_lookup = {row["image_id"]: row for row in all_rows}
    require({row["image_id"] for row in all_rows if row["split"] == split} == {row["image_id"] for row in rows}, "Split manifest differs from global partition")
    require(all(all_lookup[row["image_id"]] == row for row in rows), "Split records differ from global manifest")
    batches, image_hashes = {}, {}
    for row in all_rows:
        for store, key, label in ((batches, row["batch"], "batch"), (image_hashes, row["prepared_image_sha256"], "image content")):
            require(key not in store or store[key] == row["split"], f"Cross-split {label} leakage")
            store[key] = row["split"]
    selected = snapshot["splits"][split]
    require(metrics["image_count"] == evaluation_manifest["image_count"] == selected["image_count"] == len(rows), "Evaluated split size mismatch")
    pairs = {str(Path(pair["image"]["path"]).resolve()): pair for pair in selected["pairs"]}
    require(set(pairs) == {str(Path(row["prepared_image"]).resolve()) for row in rows}, "Evaluated image paths differ from exact source mapping")
    for row in rows:
        raw, prepared = Path(row["path"]), Path(row["prepared_image"])
        frozen.append(check_fingerprint(raw, row["sha256"]))
        frozen.append(check_fingerprint(prepared, row["prepared_image_sha256"]))
        require(pairs[str(prepared.resolve())]["image"]["sha256"] == row["prepared_image_sha256"], "Prepared image differs from frozen evaluation")
        label = Path(row["prepared_label"])
        require(label.resolve() == Path(pairs[str(prepared.resolve())]["label"]["path"]).resolve(), "Prepared label mapping differs from frozen evaluation")
        frozen.append(check_fingerprint(label, pairs[str(prepared.resolve())]["label"]["sha256"]))
        with Image.open(raw) as image:
            require(image.size == (row["decoded_width"], row["decoded_height"]) and int(image.getexif().get(274, 1)) == row["exif_orientation"], "Raw file EXIF/dimensions differ from mapping")
        with Image.open(prepared) as image:
            require(image.size == (row["prepared_width"], row["prepared_height"]) and image.mode == "RGB" and int(image.getexif().get(274, 1)) == 1,
                    "Prepared file dimensions/mode/orientation differ from mapping")
    provenance = read_json(paths["PROVENANCE.json"])
    annotation_candidates = {Path(row["path"]).resolve().parents[2] / "annotations.json" for row in rows}
    require(len(annotation_candidates) == 1, "Ambiguous original annotation-file location")
    untouched_path = annotation_candidates.pop()
    frozen.append(check_fingerprint(untouched_path, provenance["preserved_source_annotations_sha256"]))
    untouched = read_json(untouched_path)
    source_images = {im["id"]: im for im in untouched["images"]}
    source_annotations = {ann["id"]: ann for ann in untouched["annotations"]}
    for image in original["images"]:
        require(source_images.get(image["id"]) == image, "Split COCO image differs from untouched source")
    for annotation in original["annotations"]:
        require(source_annotations.get(annotation["id"]) == annotation, "Split polygon annotation differs from untouched source")
    selected_ids = {row["image_id"] for row in rows}
    require({ann["id"] for ann in untouched["annotations"] if ann["image_id"] in selected_ids} == {ann["id"] for ann in original["annotations"]}, "An original annotation is missing from selected images")
    predictions = read_json(prediction_path)
    require(isinstance(predictions, list) and len(predictions) == metrics["prediction_count"] == evaluation_manifest["prediction_count"], "Prediction record count mismatch")
    metrics["_evaluation_arguments"] = evaluation_manifest.get("evaluation_arguments")
    return split, original, mapping, predictions, frozen, metrics


def run(dataset, evaluation, out):
    out = Path(out).resolve()
    require(not out.exists(), f"Output already exists; refusing overwrite: {out}")
    split, original, mapping, predictions, frozen, native_metrics = preflight(dataset, evaluation)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "split": split,
                "dataset": str(Path(dataset).resolve()), "evaluation": str(Path(evaluation).resolve()),
                "program": fingerprint(__file__), "inputs": frozen,
                "versions": {"python": sys.version, "numpy": np.__version__, "Pillow": PIL.__version__,
                             "faster_coco_eval": faster_coco_eval.__version__, "architecture": platform.machine()},
                "device": "CPU", "model_loaded": False, "new_inference": False,
                "coordinate_transform": RESIZE_NOTE, "source_annotations_modified": False,
                "ground_truth_category_mapping": "all original TACO category IDs -> dechet category 1",
                "ground_truth_geometry": "untouched original polygon components; canonical COCO annToRLE union; no YOLO bridges",
                "ground_truth_area": "original COCO area metadata preserved for standard area ranges",
                "native_prediction_settings": native_metrics["_evaluation_arguments"],
                "scope": "General TACO litter segmentation on this split; not TSV field accuracy or physical hazard detection."}
    write_json(out / "MANIFEST.json", manifest)
    try:
        converted, audit = [], []
        for index, prediction in enumerate(predictions):
            result, record = restore_prediction(prediction, mapping)
            converted.append(result)
            audit.append(dict(prediction_index=index, **record))
        gt, gt_document = single_class_ground_truth(original)
        write_json(out / "GROUND_TRUTH_SINGLE_CLASS.json", gt_document)
        write_json(out / "PREDICTIONS_SOURCE_GRID.json", converted)
        write_json(out / "RESIZING_AUDIT.json", audit)
        metric_result = compute_metrics(gt, converted)
        metrics = {"status": "complete", "split": split, "image_count": len(mapping),
                   "ground_truth_instances": len(original["annotations"]), "prediction_count": len(predictions),
                   "checkpoint_sha256": native_metrics["checkpoint_sha256"], "dataset_sha256": native_metrics["dataset_sha256"],
                   "metric_units": "fractions_0_to_1", "original_coco_mask_evaluation": True,
                   "native_prediction_settings": native_metrics["_evaluation_arguments"],
                   "category_mapping": "all TACO categories -> dechet=1", "prediction_resizing": RESIZE_NOTE,
                   "original_gt_polygons_unchanged": True, "no_inverse_exif_applied": True,
                   "empty_prediction_masks_retained": sum(record["source_mask_pixel_count"] == 0 for record in audit),
                   "images_without_predictions": len(mapping) - len({p["image_id"] for p in converted}),
                   "scope": manifest["scope"], **metric_result}
        write_json(out / "METRICS_COCO.json", metrics)
        for proof in frozen:
            require(sha(proof["path"]) == proof["sha256"], f"Input changed during scoring: {proof['path']}")
        manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(), elapsed_s=time.perf_counter() - started,
                        inputs_unchanged=True, image_count=len(mapping), prediction_count=len(predictions),
                        output_files=[fingerprint(out / name) for name in ("GROUND_TRUTH_SINGLE_CLASS.json", "PREDICTIONS_SOURCE_GRID.json", "RESIZING_AUDIT.json", "METRICS_COCO.json")])
    except BaseException as exc:
        manifest.update(status="failed", stopped_at=datetime.now(timezone.utc).isoformat(), elapsed_s=time.perf_counter() - started,
                        error={"type": type(exc).__name__, "message": str(exc)})
        write_json(out / "MANIFEST.json", manifest)
        raise
    write_json(out / "MANIFEST.json", manifest)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metrics = run(args.dataset, args.evaluation, args.out)
    print(json.dumps({"status": metrics["status"], "split": metrics["split"], "segmentation_metrics": metrics["segmentation_metrics"]}), flush=True)


if __name__ == "__main__":
    main()
