#!/usr/bin/env python3
"""Train the separate, one-class litter segmenter in a NEW local run folder.

No training happens at import. The public checkpoint is read-only. No automatic
download, install, device fallback, OOM retry or parameter adjustment is allowed.
Run --help for the CLI; unit tests only exercise configuration and provenance.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import yaml

EXPECTED_NAMES = {0: "dechet"}
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}
GROUND_TRUTH_NOTE = (
    "Metrics use the converted YOLO segmentation labels. A multipart COCO object "
    "can be represented by one polygon linked with artificial bridges; conversion "
    "and rasterization can alter its mask. These metrics are not an evaluation "
    "against untouched original COCO masks. Preserve the original COCO annotation "
    "file for a separate future comparison."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serializable(value: Any) -> Any:
    """Finite JSON values; unknown/nonfinite metrics become explicit null."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if hasattr(value, "tolist"):
        return serializable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(serializable(value), indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    if not isinstance(names, dict):
        raise ValueError("Class names must be a list or a dictionary")
    try:
        normalized = {int(k): str(v) for k, v in names.items()}
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid class dictionary") from exc
    if len(normalized) != len(names):
        raise ValueError("Duplicate class IDs after normalization")
    return normalized


def require_litter_names(names: Any) -> dict[int, str]:
    result = normalize_names(names)
    if result != EXPECTED_NAMES:
        raise ValueError(f"Expected exactly {EXPECTED_NAMES}, got {result}")
    return result


def fingerprint(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected a local file: {path}")
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def _image_paths(entry: Any, root: Path) -> tuple[list[Path], list[Path]]:
    entries = entry if isinstance(entry, list) else [entry]
    images: list[Path] = []
    lists: list[Path] = []
    for item in entries:
        if not isinstance(item, str) or not item:
            raise ValueError("Split paths must be nonempty strings")
        path = (root / item).resolve(strict=True)
        if path.is_dir():
            images.extend(p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.suffix.lower() == ".txt":
            lists.append(path)
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("./"):
                    image = path.parent / line[2:]
                elif Path(line).is_absolute():
                    image = Path(line)
                else:
                    raise ValueError("Image-list entries must be absolute or begin with ./; ambiguous relative paths refused")
                image = image.resolve(strict=True)
                if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                    raise ValueError(f"Invalid image-list entry: {image}")
                images.append(image)
        else:
            raise ValueError("Use image directories or explicit .txt image lists for dataset splits")
    if not images or len(set(images)) != len(images):
        raise ValueError("A split is empty or contains repeated image paths")
    return sorted(images), lists


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    indices = [i for i, part in enumerate(parts) if part == "images"]
    if not indices:
        raise ValueError(f"Expected an images directory to locate the paired label: {image}")
    parts[indices[-1]] = "labels"
    label = Path(*parts).with_suffix(".txt")
    if not label.is_file():
        raise ValueError(f"Missing explicit label file (including empty negatives): {label}")
    return label.resolve()


def audit_label(label: Path) -> int:
    instances = 0
    for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            numbers = [float(x) for x in line.split()]
        except ValueError as exc:
            raise ValueError(f"Non-numeric label at {label}:{line_number}") from exc
        if len(numbers) < 7 or len(numbers) % 2 != 1:
            raise ValueError(f"Segmentation polygon expected, not a box, at {label}:{line_number}")
        if not all(math.isfinite(x) for x in numbers) or numbers[0] != 0:
            raise ValueError(f"Expected finite class-0 polygon at {label}:{line_number}")
        if not all(0 <= x <= 1 for x in numbers[1:]):
            raise ValueError(f"Normalized contour outside [0,1] at {label}:{line_number}")
        if len(set(zip(numbers[1::2], numbers[2::2]))) < 3:
            raise ValueError(f"Degenerate polygon at {label}:{line_number}")
        instances += 1
    return instances


def dataset_provenance(data: str | Path, splits: tuple[str, ...] = ("train", "val", "test")) -> dict[str, Any]:
    """Hash the exact YAML, image/label pairs and retained source evidence.

    Reading test files here records identity only: no predictions or test metrics
    are computed. The training call itself uses only train/val.
    """
    data = Path(data).expanduser().resolve(strict=True)
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Dataset YAML must be a mapping")
    names = require_litter_names(config.get("names"))
    if config.get("nc", 1) != 1:
        raise ValueError("The litter dataset must have nc=1")
    if config.get("download"):
        raise ValueError("Dataset download hooks are forbidden; prepare local data first")
    root = Path(config.get("path") or data.parent).expanduser()
    if not root.is_absolute():
        raise ValueError("Dataset path must be absolute to avoid Ultralytics path ambiguity")
    root = root.resolve(strict=True)
    manifest: dict[str, Any] = {"yaml": fingerprint(data), "root": str(root), "names": names,
                                "splits": {}, "ground_truth_note": GROUND_TRUTH_NOTE,
                                "test_access_during_training_preflight": "file_identity_and_label_format_only"}
    seen_hashes: dict[str, str] = {}
    all_pairs = []
    for split in splits:
        if not config.get(split):
            raise ValueError(f"Dataset YAML has no {split} split")
        images, lists = _image_paths(config[split], root)
        pairs = []
        instances = 0
        for image in images:
            label = _label_path(image)
            im, lab = fingerprint(image), fingerprint(label)
            prior_split = seen_hashes.get(im["sha256"])
            if prior_split is not None and prior_split != split:
                raise ValueError(f"Identical image content across {prior_split}/{split}: {image}")
            seen_hashes[im["sha256"]] = split
            count = audit_label(label)
            instances += count
            pairs.append({"image": im, "label": lab, "instances": count})
        manifest["splits"][split] = {"image_count": len(images), "instance_count": instances,
                                      "image_lists": [fingerprint(p) for p in lists], "pairs": pairs}
        all_pairs.extend({"split": split, **pair} for pair in pairs)
    evidence = []
    for name in ("dataset_manifest.json", "DATASET_MANIFEST.json", "manifest.json", "PREPARATION.json",
                 "annotations.json", "CONVERSION_AUDIT.json", "conversion_audit.json", "MANIFEST_ALL.json",
                 "manifest_train.json", "manifest_val.json", "manifest_test.json", "SUMMARY.json",
                 "ALL_POLYGON_AUDIT.json", "COCO_ORIGINAL_ALL.json", "COCO_ORIGINAL_train.json",
                 "COCO_ORIGINAL_val.json", "COCO_ORIGINAL_test.json"):
        path = root / name
        if path.is_file():
            evidence.append(fingerprint(path))
    for key in ("coco_annotations", "original_coco_annotations", "source_annotations"):
        if isinstance(config.get(key), str):
            evidence.append(fingerprint((root / config[key]).resolve()))
    manifest["source_evidence"] = evidence
    payload = json.dumps({"yaml_sha256": manifest["yaml"]["sha256"], "pairs": all_pairs,
                          "source_evidence": evidence}, sort_keys=True, separators=(",", ":"))
    manifest["dataset_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return manifest


def choose_device(requested: str, torch_module: Any) -> str:
    available = bool(torch_module.backends.mps.is_available())
    if requested not in {"auto", "mps", "cpu"}:
        raise ValueError("device must be auto, mps or cpu")
    if requested == "mps" and not available:
        raise RuntimeError("MPS explicitly requested but unavailable; no CPU fallback was attempted")
    chosen = "mps" if requested == "auto" and available else ("cpu" if requested == "auto" else requested)
    if chosen == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled; hidden CPU fallback is forbidden")
    return chosen


def configure_local_runtime():
    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ["YOLO_OFFLINE"] = "True"
    import torch
    import ultralytics
    import cv2
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    # Disable external reporting for this process only, without saving settings.
    for key in ("clearml", "comet", "dvc", "mlflow", "raytune", "tensorboard", "wandb", "sync"):
        if key in SETTINGS:
            dict.__setitem__(SETTINGS, key, False)
    return torch, ultralytics, YOLO


def runtime_provenance(torch_module: Any, ultralytics_module: Any, requested: str, device: str) -> dict[str, Any]:
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "torch": torch_module.__version__, "ultralytics": ultralytics_module.__version__,
            "mps_available": bool(torch_module.backends.mps.is_available()),
            "requested_device": requested, "resolved_device": device,
            "mps_memory_fraction": 0.5 if device == "mps" else None,
            "mps_fallback_enabled": False, "automatic_install": False,
            "automatic_download": False, "external_reporting": False}


def assert_segmenter(model: Any, *, litter: bool, require_nano26: bool = False) -> dict[str, Any]:
    if model.task != "segment":
        raise ValueError(f"A segmentation model is required, got task={model.task!r}")
    names = require_litter_names(model.names) if litter else normalize_names(model.names)
    config = getattr(model.model, "yaml", {})
    if require_nano26:
        yaml_file = str(config.get("yaml_file", ""))
        if "yolo26" not in yaml_file or config.get("scale") != "n":
            raise ValueError(f"Expected YOLO26 nano segmentation architecture, got {yaml_file!r}, scale={config.get('scale')!r}")
    return {"task": model.task, "names": names, "architecture": serializable(config)}


def build_train_kwargs(args: argparse.Namespace, device: str) -> dict[str, Any]:
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32:
        raise ValueError("Positive epochs/batch and imgsz >= 32 are required")
    out = Path(args.out).expanduser().resolve()
    return dict(data=str(Path(args.data).expanduser().resolve()), epochs=args.epochs,
                device=device, batch=args.batch, imgsz=args.imgsz,
                project=str(out.parent), name=out.name, exist_ok=True,
                freeze=10, optimizer="AdamW", lr0=0.0003, seed=42, deterministic=True,
                workers=0, cache=False, amp=False, pretrained=True, resume=False,
                save=True, save_period=-1, val=True, split="val", plots=True,
                patience=args.epochs, mosaic=0.25, close_mosaic=2, degrees=3.0,
                translate=0.05, scale=0.15, hsv_h=0.01, hsv_s=0.3, hsv_v=0.25,
                fliplr=0.5, flipud=0.0, mixup=0.0, copy_paste=0.0)


def guard_no_oom_retry(trainer: Any, expected_batch: int) -> None:
    """Disable the audited native first-epoch OOM batch-halving branch.

    Called at epoch start AFTER Ultralytics initializes its counter. In version
    8.4.192 the exception branch raises immediately when _oom_retries >= 3, and a
    normally completed epoch resets it to zero. No training source is modified.
    """
    if int(trainer.batch_size) != expected_batch or int(trainer.args.batch) != expected_batch:
        raise RuntimeError("The configured training batch was changed unexpectedly")
    trainer._oom_retries = 3


def verify_oom_guard_contract() -> dict[str, Any]:
    from ultralytics.engine.trainer import BaseTrainer
    source = inspect.getsource(BaseTrainer._do_train)
    if ("self._oom_retries >= 3" not in source or
            'self.run_callbacks("on_train_epoch_start")' not in source or
            "self._oom_retries = 0" not in source):
        raise RuntimeError("Unrecognized Ultralytics OOM recovery code; refusing unverified automatic retries")
    return {"method": "set_retry_counter_to_native_raise_threshold_at_epoch_start",
            "trainer_source": fingerprint(Path(inspect.getfile(BaseTrainer)))}


def read_results(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(handle)]


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Output already exists; no overwrite or automatic resume: {out}")
    model_file = Path(args.model).expanduser().resolve(strict=True)
    if model_file.suffix.lower() != ".pt":
        raise ValueError("--model must be an existing local .pt checkpoint")
    model_proof = fingerprint(model_file)
    dataset = dataset_provenance(args.data)
    torch, ultralytics, YOLO = configure_local_runtime()
    device = choose_device(args.device, torch)
    kwargs = build_train_kwargs(args, device)
    oom_guard = verify_oom_guard_contract()
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = out / "TRAINING_MANIFEST.json"
    write_json(out / "DATASET_SNAPSHOT.json", dataset)
    manifest = {"status": "running", "started_at": utc_now(), "operation": "train_litter_segmenter",
                "initial_model": model_proof, "dataset_sha256": dataset["dataset_sha256"],
                "dataset_snapshot": fingerprint(out / "DATASET_SNAPSHOT.json"),
                "program": fingerprint(Path(__file__)), "cli_arguments": serializable(vars(args)),
                "training_arguments": kwargs, "runtime": runtime_provenance(torch, ultralytics, args.device, device),
                "expected_names": EXPECTED_NAMES, "ground_truth_note": GROUND_TRUTH_NOTE,
                "requested_epochs": args.epochs, "completed_epochs": 0,
                "public_model_overwritten": False, "oom_retry_or_parameter_change": False,
                "oom_guard": oom_guard,
                "scope": "Training on prepared litter labels; no TSV field accuracy or physical danger claim."}
    write_json(manifest_path, manifest)
    (out / "EPOCHS.jsonl").touch(exist_ok=False)
    start = time.perf_counter()
    try:
        if device == "mps":
            torch.mps.set_per_process_memory_fraction(0.5)
        model = YOLO(str(model_file))
        manifest["initial_architecture"] = assert_segmenter(model, litter=False, require_nano26=True)
        write_json(manifest_path, manifest)

        def on_train_start(trainer):
            if Path(trainer.save_dir).resolve() != out:
                raise RuntimeError("Ultralytics changed the requested run directory")
            write_json(out / "RESOLVED_TRAIN_ARGS.json", vars(trainer.args))

        def on_fit_epoch_end(trainer):
            results = read_results(out / "results.csv")
            epoch = int(trainer.epoch) + 1
            # Ultralytics invokes this callback once more for final best-weight
            # validation at epoch+1; label it explicitly, not as another epoch.
            event = "epoch" if epoch <= len(results) else "final_best_validation"
            record = {"event": event, "epoch": epoch, "time_utc": utc_now(),
                      "elapsed_s": time.perf_counter() - start,
                      "metrics": serializable(trainer.metrics), "learning_rates": serializable(getattr(trainer, "lr", {})),
                      "results_csv_last_row": results[-1] if results else None}
            with (out / "EPOCHS.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
            manifest.update(completed_epochs=len(results), latest_event=record)
            write_json(manifest_path, manifest)
            print(json.dumps({"event": event, "epoch": epoch, "completed_epochs": len(results),
                              "metrics": record["metrics"]}, ensure_ascii=False), flush=True)

        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_epoch_start", lambda trainer: guard_no_oom_retry(trainer, args.batch))
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.train(**kwargs)
        rows = read_results(out / "results.csv")
        if not rows or len(rows) > args.epochs:
            raise RuntimeError("No valid recorded training epochs in results.csv")
        expected_epochs = list(range(1, len(rows) + 1))
        if [int(float(row["epoch"])) for row in rows] != expected_epochs:
            raise RuntimeError("Training results have a broken epoch sequence")
        checkpoints = {}
        for kind in ("best", "last"):
            checkpoint = out / "weights" / f"{kind}.pt"
            proof = fingerprint(checkpoint)
            reloaded = YOLO(str(checkpoint))  # CPU load only; no predictions.
            proof["reload_verification"] = assert_segmenter(reloaded, litter=True, require_nano26=True)
            checkpoints[kind] = proof
        if sha256(model_file) != model_proof["sha256"]:
            raise RuntimeError("The initial public checkpoint changed during training")
        final_dataset = dataset_provenance(args.data)
        if final_dataset["dataset_sha256"] != dataset["dataset_sha256"]:
            raise RuntimeError("Dataset content changed during training")
        manifest.update(status="complete", completed_at=utc_now(), elapsed_s=time.perf_counter() - start,
                        completed_epochs=len(rows), requested_epochs_reached=len(rows) == args.epochs,
                        results_csv=fingerprint(out / "results.csv"), last_epoch_results=rows[-1],
                        checkpoints=checkpoints, initial_model_unchanged=True, dataset_unchanged=True,
                        epoch_log=fingerprint(out / "EPOCHS.jsonl"))
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        stopped_at=utc_now(), elapsed_s=time.perf_counter() - start,
                        error={"type": type(exc).__name__, "message": str(exc)})
        write_json(manifest_path, manifest)
        raise
    write_json(manifest_path, manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "model", "out"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--epochs", type=int, default=12)
    result.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    result.add_argument("--batch", type=int, default=2)
    result.add_argument("--imgsz", type=int, default=640)
    return result


def main():
    manifest = run_training(parser().parse_args())
    print(json.dumps({"status": manifest["status"], "completed_epochs": manifest["completed_epochs"],
                      "checkpoints": manifest["checkpoints"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
