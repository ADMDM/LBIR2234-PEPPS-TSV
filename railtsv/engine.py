from pathlib import Path
import importlib.metadata
import json
import platform
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .common import TASKS, device, new_dir, dump, seed_all, rgb_read, mask_read, sha256, check_disjoint
from .data import RailDataset, letterbox, normalize
from .model import build_model, freeze_batchnorm, dice_ce, load_checkpoint, transfer
from .metrics import Metrics


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(), "architecture": platform.machine(),
            "packages": {x: importlib.metadata.version(x) for x in ("torch", "torchvision", "segmentation-models-pytorch", "albumentations", "numpy", "opencv-python-headless")},
            "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available()}


def check_device(args):
    seed_all(42)
    torch.set_num_threads(4)
    dev = device(args.device)
    model = build_model().to(dev).train()
    freeze_batchnorm(model)
    x = torch.randn(1, 3, 64, 96, device=dev)
    target = torch.zeros(1, 64, 96, dtype=torch.long, device=dev)
    target[:, :, 30:34] = 1
    target[:, :8] = 255
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    start = time.perf_counter()
    logits = model(x)
    loss = dice_ce(logits, target)
    loss.backward()
    if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError("Loss ou gradients non finis")
    optimizer.step()
    if dev.type == "mps":
        torch.mps.synchronize()
    result = {**environment(), "device": str(dev), "model": "DeepLabV3Plus/ResNet50",
              "shape": list(logits.shape), "loss": float(loss.detach().cpu()),
              "forward_backward_optimizer": "passed", "seconds": time.perf_counter()-start}
    if args.out:
        dump(new_dir(args.out) / "device.json", result)
    print(json.dumps(result, indent=2))


def validate(model, loader, dev, names):
    model.eval()
    metric = Metrics(names)
    losses = []
    with torch.inference_mode():
        for image, target in loader:
            logits = model(image.to(dev))
            loss = dice_ce(logits, target.to(dev))
            if not torch.isfinite(loss):
                raise RuntimeError("Loss de validation non finie")
            losses.append(float(loss.cpu()))
            metric.update(logits.argmax(1).cpu().numpy(), target.numpy())
    return float(np.mean(losses)), metric.result()


def train(args):
    training_only = getattr(args, "training_only_demo", False)
    if training_only and args.val:
        raise ValueError("--training-only-demo ne prend pas --val : aucun score de validation pour cet essai")
    if not training_only and not args.val:
        raise ValueError("Fournir --val indépendant, ou choisir explicitement --training-only-demo")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("epochs, batch-size et lr doivent être positifs")
    if args.pretrained and args.init:
        raise ValueError("Choisir --pretrained OU --init")
    seed_all(args.seed)
    torch.set_num_threads(args.cpu_threads)
    dev, names = device(args.device), TASKS[args.task]
    train_ds = RailDataset(args.train, len(names), args.height, args.width, True, args.seed)
    val_ds = None if training_only else RailDataset(args.val, len(names), args.height, args.width)
    if val_ds is not None:
        check_disjoint(train_ds.rows, val_ds.rows)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = None if val_ds is None else DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
    model = build_model(len(names), args.pretrained)
    init_info = transfer(model, args.init) if args.init else None
    out = new_dir(args.out)
    model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    metadata = {"arguments": vars(args), "environment": environment(), "device": str(dev),
                "initialization": init_info, "train_manifest_sha256": sha256(args.train),
                "val_manifest_sha256": sha256(args.val) if args.val else None, "batchnorm_statistics": "frozen",
                "training_only_demo": training_only,
                "metric_resolution": [args.height, args.width],
                "validation_note": ("Exploratory training only: no independent validation or accuracy estimate." if training_only else
                                    "Metrics at letterboxed training resolution; inspect original-resolution results separately.")}
    dump(out / "run.json", metadata)
    history, best, steps = [], -1., 0
    for epoch in range(args.epochs):
        model.train()
        freeze_batchnorm(model)
        losses = []
        start = time.perf_counter()
        for batch_index, (image, target) in enumerate(loader, 1):
            # Ignore-only samples remain legal; an entirely ignored batch has no update.
            if not (target != 255).any():
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = dice_ce(model(image.to(dev)), target.to(dev))
            if not torch.isfinite(loss):
                raise RuntimeError("Loss non finie")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach().cpu()))
            progress = getattr(args, 'progress_every', 0)
            if progress and (batch_index == 1 or batch_index % progress == 0 or batch_index == len(loader)):
                print(f'Passage {epoch+1}/{args.epochs} — lot {batch_index}/{len(loader)} — loss={losses[-1]:.4f} — {time.perf_counter()-start:.0f} s', flush=True)
        if not losses:
            raise ValueError("Aucun pixel valide après prétraitement")
        if training_only:
            score = None
            record = {"epoch": epoch+1, "train_loss": float(np.mean(losses)),
                      "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter()-start}
            print(f"Passage {epoch+1}/{args.epochs} — loss={record['train_loss']:.4f} — essai sans validation indépendante", flush=True)
        else:
            val_loss, metrics = validate(model, val_loader, dev, names)
            score = metrics["foreground_dice"]
            if metrics["valid_pixels"] == 0 or score is None:
                raise ValueError("Validation vide ou sans rail ni prédiction : contrôler les annotations")
            record = {"epoch": epoch+1, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
                      "metrics": metrics, "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter()-start}
            target_name={'corridor':'couloir', 'railsem_track':'zone de voie'}.get(args.task, 'rails')
            print(f"Epoch {epoch+1}/{args.epochs} — train={record['train_loss']:.4f} val={val_loss:.4f} Dice {target_name}={score:.4f}", flush=True)
        history.append(record)
        if training_only or score > best:
            best = score if score is not None else None
            torch.save({"format": "railtsv-deeplab-v1", "architecture": "DeepLabV3Plus", "encoder": "resnet50",
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "task": args.task, "classes": names, "height": args.height, "width": args.width,
                        "preprocessing": "imagenet-letterbox-v1", "ignore_index": 255,
                        "epoch": epoch+1, "steps": steps, "best_val_dice": best,
                        "synthetic_demo": args.synthetic_demo, "training_only_demo": training_only,
                        "seed": args.seed}, out / ("last.pt" if training_only else "best.pt"))
        dump(out / "history.json", history)
        scheduler.step()
    print(out / ("last.pt" if training_only else "best.pt"), flush=True)


def predict_image(model, image, dev, height, width):
    prepared, _, (top, left, nh, nw) = letterbox(image, height, width)
    with torch.inference_mode():
        logits = model(normalize(prepared)[None].to(dev)).cpu()
        logits = logits[:, :, top:top+nh, left:left+nw]
        logits = F.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False)
        probs = logits.softmax(1)[0].numpy()
    return probs.argmax(0).astype(np.uint8), probs


def infer(args):
    seed_all(42)
    torch.set_num_threads(args.cpu_threads)
    dev = device(args.device)
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint)
        names = checkpoint["classes"]
        model = build_model(len(names))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        height, width = args.height or checkpoint["height"], args.width or checkpoint["width"]
        untrained = checkpoint["synthetic_demo"] or checkpoint.get("training_only_demo", False)
        label = ("TEST TECHNIQUE - MODELE SYNTHETIQUE" if checkpoint["synthetic_demo"] else
                 "ESSAI - SANS VALIDATION INDEPENDANTE" if checkpoint.get("training_only_demo", False) else
                 "PREDICTION - A VALIDER")
    elif args.allow_untrained:
        names, model = TASKS["binary"], build_model(2)
        height, width, untrained = args.height or 384, args.width or 640, True
        label = "TEST TECHNIQUE - POIDS ALEATOIRES"
    else:
        raise ValueError("Fournir --checkpoint ou --allow-untrained pour un test technique explicite")
    model.to(dev).eval()
    source = Path(args.input).resolve()
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")) if source.is_dir() else [source]
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError("Entrée vide ou noms d'images ambigus")
    out = new_dir(args.out)
    results = []
    for path in paths:
        image = rgb_read(path)
        start = time.perf_counter()
        mask, probs = predict_image(model, image, dev, height, width)
        folder = new_dir(out / path.stem)
        Image.fromarray(mask).save(folder / "mask.png")
        colors = np.array([[0, 0, 0], [0, 220, 255], [255, 150, 0]], np.uint8)
        overlay = image.copy()
        positive = mask > 0
        corridor=checkpoint.get('task')=='corridor' if args.checkpoint else False
        opacity=.15 if corridor else .45
        overlay[positive] = ((1-opacity)*image[positive] + opacity*colors[mask[positive]]).astype(np.uint8)
        if corridor:
            from .geometry import corridor_profile,draw_axis
            overlay=draw_axis(overlay,corridor_profile(mask,probs))
        view = Image.fromarray(overlay)
        draw = ImageDraw.Draw(view)
        draw.rectangle((0, 0, view.width, 25), fill="black")
        draw.text((5, 6), label, fill="white")
        view.save(folder / "overlay.png")
        np.savez_compressed(folder / "probabilities.npz", probabilities=probs.astype(np.float16))
        result = {"source": str(path), "sha256": sha256(path), "shape": list(mask.shape), "device": str(dev),
                  "network_size": [height, width], "classes": names, "technical_test_only": untrained,
                  "seconds": time.perf_counter()-start, "probabilities": "uncalibrated softmax, not reliability",
                  "representation":'corridor' if corridor else 'rails'}
        dump(folder / "prediction.json", result)
        results.append(result)
    dump(out / "summary.json", {"checkpoint": args.checkpoint, "images": results})
    print(out)


def evaluate(args):
    checkpoint = load_checkpoint(args.checkpoint)
    names = checkpoint["classes"]
    ds = RailDataset(args.manifest, len(names), checkpoint["height"], checkpoint["width"])
    dev = device(args.device)
    torch.set_num_threads(args.cpu_threads)
    model = build_model(len(names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(dev).eval()
    metrics = Metrics(names)
    axis_metrics=None
    if checkpoint.get('task')=='corridor':
        from .axis_metrics import AxisMetrics
        axis_metrics=AxisMetrics()
    for row in ds.rows:
        image = rgb_read(row["image"])
        mask, probs = predict_image(model, image, dev, checkpoint["height"], checkpoint["width"])
        reference=mask_read(row["mask"],allowed=set(range(len(names)))|{255})
        metrics.update(mask,reference)
        if axis_metrics is not None:axis_metrics.update(mask,probs,reference,row['image'])
    result = {"resolution": "original", "checkpoint_sha256": sha256(args.checkpoint),
              "manifest_sha256": sha256(args.manifest), "synthetic_demo": checkpoint["synthetic_demo"],
              "metrics": metrics.result()}
    if axis_metrics is not None:result['axis']=axis_metrics.result()
    dump(new_dir(args.out) / "metrics.json", result)
    print(json.dumps(result, indent=2))
