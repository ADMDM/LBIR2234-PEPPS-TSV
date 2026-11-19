#!/usr/bin/env python3
"""Evaluate the one-class litter segmenter on val OR test in a NEW local folder.

Native-resolution predicted masks are serialized by Ultralytics save_json.
Ground truth is the prepared YOLO polygon dataset, including its documented
multipart-conversion approximation; this is not original-COCO-mask scoring.
Importing this file performs no model load, prediction, evaluation or install.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from train_litter import (GROUND_TRUTH_NOTE, assert_segmenter, choose_device,
                          configure_local_runtime, dataset_provenance, fingerprint,
                          require_litter_names,
                          runtime_provenance, serializable, sha256, utc_now, write_json)


def require_native_mask_json_dependency() -> str:
    """Fail before evaluation, never let Ultralytics install a missing package."""
    try:
        installed = importlib.metadata.version("faster-coco-eval")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Native segmentation save_json requires faster-coco-eval>=1.6.7; it is absent. No installation attempted.") from exc
    from packaging.version import Version
    if Version(installed) < Version("1.6.7"):
        raise RuntimeError(f"faster-coco-eval {installed} is too old; >=1.6.7 is required. No installation attempted.")
    return installed


def build_eval_kwargs(args: argparse.Namespace, device: str) -> dict:
    if args.split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    if args.batch < 1 or args.imgsz < 32 or not 0 <= args.conf <= 1 or not 0 < args.iou <= 1:
        raise ValueError("Invalid batch, image size, confidence or IoU setting")
    out = Path(args.out).expanduser().resolve()
    return dict(data=str(Path(args.data).expanduser().resolve()), split=args.split,
                device=device, batch=args.batch, imgsz=args.imgsz, workers=0,
                project=str(out.parent), name=out.name, exist_ok=True,
                conf=args.conf, iou=args.iou, max_det=100, nms=True,
                plots=True, save_json=True, save_txt=False, save_conf=True,
                retina_masks=True, augment=False, verbose=True)


def mask_confidence_curve(metrics) -> dict:
    """Native mask P/R confidence curves use tp[:,0], the validator's IoU=.50.

    Verified against local Ultralytics 8.4.192 ap_per_class/Metric.update and
    DetectionValidator.iouv. No interpolation or smoothing changes this export.
    """
    import numpy as np
    require_litter_names(metrics.names)
    segment = metrics.seg
    ids = np.asarray(segment.ap_class_index)
    confidence = np.asarray(segment.px, dtype=float)
    precision = np.asarray(segment.p_curve, dtype=float)
    recall = np.asarray(segment.r_curve, dtype=float)
    if ids.shape != (1,) or ids[0] != 0:
        raise RuntimeError("Expected one native mask curve for class 0: dechet")
    if confidence.ndim != 1 or len(confidence) < 2 or precision.shape != (1, len(confidence)) or recall.shape != precision.shape:
        raise RuntimeError("Native mask confidence curve has an unexpected shape")
    if not all(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
               for values in (confidence, precision, recall)):
        raise RuntimeError("Native mask confidence curves contain invalid values")
    if not (np.diff(confidence) > 0).all() or confidence[0] != 0 or confidence[-1] != 1:
        raise RuntimeError("Native confidence thresholds must increase from zero to one")
    return {"class_id": 0, "class_name": "dechet", "iou_threshold": 0.5,
            "confidence_thresholds": confidence.tolist(), "precision": precision[0].tolist(),
            "recall": recall[0].tolist(), "source": "metrics.seg.px,p_curve,r_curve; native ap_per_class tp[:,0]",
            "scope": "Native mask IoU50 curve against converted YOLO labels on this evaluation split; not TSV performance."}


def metrics_payload(metrics) -> dict:
    results = serializable(metrics.results_dict)
    required = [f"metrics/{key}({kind})" for kind in ("B", "M")
                for key in ("precision", "recall", "mAP50", "mAP50-95")]
    if not isinstance(results, dict) or any(results.get(key) is None for key in required):
        raise RuntimeError("Evaluation did not produce finite box and mask metrics")
    return {"box_metrics": {k: v for k, v in results.items() if k.endswith("(B)")},
            "mask_metrics": {k: v for k, v in results.items() if k.endswith("(M)")},
            "raw_results": results, "per_class": serializable(metrics.summary()),
            "mask_confidence_curve_iou50": mask_confidence_curve(metrics),
            "speed_ms_per_image": serializable(getattr(metrics, "speed", {}))}


def run_evaluation(args: argparse.Namespace) -> dict:
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Output already exists; refusing overwrite: {out}")
    checkpoint = Path(args.model).expanduser().resolve(strict=True)
    if checkpoint.suffix.lower() != ".pt":
        raise ValueError("--model must be an existing local .pt checkpoint")
    proof = fingerprint(checkpoint)
    dataset = dataset_provenance(args.data)
    dependency_version = require_native_mask_json_dependency()
    torch, ultralytics, YOLO = configure_local_runtime()
    device = choose_device(args.device, torch)
    kwargs = build_eval_kwargs(args, device)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "DATASET_SNAPSHOT.json", dataset)
    manifest_path = out / "EVALUATION_MANIFEST.json"
    manifest = {"status": "running", "started_at": utc_now(), "operation": "evaluate_litter_segmenter",
                "split": args.split, "checkpoint": proof, "dataset_sha256": dataset["dataset_sha256"],
                "dataset_snapshot": fingerprint(out / "DATASET_SNAPSHOT.json"),
                "program": fingerprint(Path(__file__)), "shared_program": fingerprint(Path(__file__).with_name("train_litter.py")),
                "cli_arguments": serializable(vars(args)), "evaluation_arguments": kwargs,
                "runtime": runtime_provenance(torch, ultralytics, args.device, device),
                "faster_coco_eval": dependency_version, "ground_truth_note": GROUND_TRUTH_NOTE,
                "original_coco_mask_evaluation": False, "threshold_tuning_performed": False,
                "physical_intrusion_assessed": False,
                "scope": "Mask and box metrics for this litter split; not validated TSV field performance."}
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    observed = {}
    try:
        if device == "mps":
            torch.mps.set_per_process_memory_fraction(0.5)
        model = YOLO(str(checkpoint))
        manifest["checkpoint_architecture"] = assert_segmenter(model, litter=True, require_nano26=True)
        write_json(manifest_path, manifest)

        def on_val_end(validator):
            if Path(validator.save_dir).resolve() != out:
                raise RuntimeError("Ultralytics changed the requested evaluation directory")
            if bool(validator.end2end):
                raise RuntimeError("Conventional NMS requested but end-to-end backend remains enabled")
            observed.update(seen=int(validator.seen), prediction_count=len(validator.jdict),
                            segmentation_backend_end2end=bool(validator.end2end),
                            image_paths=sorted(str(Path(p).resolve()) for p in validator.dataloader.dataset.im_files),
                            resolved_arguments=serializable(vars(validator.args)))
            if not validator.jdict:
                # The native writer intentionally skips the file for zero
                # predictions. Preserve that observed empty result explicitly.
                write_json(out / "predictions.json", [])

        model.add_callback("on_val_end", on_val_end)
        metrics = model.val(**kwargs)
        expected = dataset["splits"][args.split]
        expected_paths = sorted(pair["image"]["path"] for pair in expected["pairs"])
        if observed.get("seen") != expected["image_count"] or observed.get("image_paths") != expected_paths:
            raise RuntimeError("Evaluated images differ from the frozen split")
        predictions = fingerprint(out / "predictions.json")
        payload = metrics_payload(metrics)
        report = {"status": "complete", "split": args.split, "image_count": observed["seen"],
                  "ground_truth_instances": expected["instance_count"],
                  "prediction_count": observed["prediction_count"],
                  "segmentation_backend_end2end": observed["segmentation_backend_end2end"],
                  "checkpoint_sha256": proof["sha256"], "dataset_sha256": dataset["dataset_sha256"],
                  "ground_truth_note": GROUND_TRUTH_NOTE, "original_coco_mask_evaluation": False,
                  "metric_units": "fractions_0_to_1", "predicted_masks": "native_image_resolution_RLE_in_predictions.json",
                  "predictions_json": predictions, **payload}
        if sha256(checkpoint) != proof["sha256"]:
            raise RuntimeError("Checkpoint changed during evaluation")
        if dataset_provenance(args.data)["dataset_sha256"] != dataset["dataset_sha256"]:
            raise RuntimeError("Dataset changed during evaluation")
        write_json(out / "METRICS.json", report)
        write_json(out / "RESOLVED_EVAL_ARGS.json", observed["resolved_arguments"])
        manifest.update(status="complete", completed_at=utc_now(), elapsed_s=time.perf_counter() - started,
                        image_count=observed["seen"], prediction_count=observed["prediction_count"],
                        metrics_json=fingerprint(out / "METRICS.json"), predictions_json=predictions,
                        checkpoint_unchanged=True, dataset_unchanged=True,
                        artifacts=[fingerprint(p) for p in sorted(out.iterdir())
                                   if p.is_file() and p != manifest_path and not p.name.endswith(".tmp")])
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        stopped_at=utc_now(), elapsed_s=time.perf_counter() - started,
                        error={"type": type(exc).__name__, "message": str(exc)})
        write_json(manifest_path, manifest)
        raise
    write_json(manifest_path, manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "model", "out"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--split", choices=("val", "test"), default="val")
    result.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    result.add_argument("--batch", type=int, default=1)
    result.add_argument("--imgsz", type=int, default=640)
    result.add_argument("--conf", type=float, default=0.001)
    result.add_argument("--iou", type=float, default=0.7)
    return result


def main():
    manifest = run_evaluation(parser().parse_args())
    print(json.dumps({"status": manifest["status"], "split": manifest["split"],
                      "metrics_json": manifest["metrics_json"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
