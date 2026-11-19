from pathlib import Path
import csv
import json
import math
import random
import shutil
import cv2
import numpy as np
from PIL import Image
from .common import new_dir, dump, sha256, mask_read, read_manifest, write_manifest, check_disjoint


def label_mapping(config):
    labels = config["labels"]
    names = [x["name"] for x in labels]
    if len(names) != len(set(names)):
        raise ValueError("Noms de labels dupliqués")
    # Official dense masks use zero-based label-list positions, not Cityscapes IDs.
    ids = {name: int(label.get("id", i)) for i, (name, label) in enumerate(zip(names, labels))}
    if len(set(ids.values())) != len(ids) or any(i < 0 or i >= 255 for i in ids.values()):
        raise ValueError("IDs source invalides")
    for name in ("rail-raised", "rail-embedded", "rail-track", "trackbed", "tram-track"):
        if name not in ids:
            raise ValueError(f"Configuration RailSem19 incomplète : {name}")
    ignored = {ids[l["name"]] for l in labels if l.get("evaluate") is False or l["name"] in ("unlabeled", "void", "ignore")}
    rails = {ids["rail-raised"], ids["rail-embedded"]}
    if rails & ignored:
        raise ValueError("La configuration ignore une classe rail cible")
    return ids, rails, ignored


def remap(mask, ids, rails, ignored):
    unknown = set(np.unique(mask)) - set(ids.values()) - {255}
    if unknown:
        raise ValueError(f"IDs source absents de la configuration : {sorted(unknown)}")
    result = np.zeros(mask.shape, np.uint8)
    result[np.isin(mask, list(rails))] = 1
    result[np.isin(mask, list(ignored | {255}))] = 255
    return result


def index_files(root, extensions):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            if path.stem in result:
                raise ValueError(f"Nom ambigu en sous-dossiers : {path.stem}")
            result[path.stem] = path.resolve()
    if not result:
        raise ValueError(f"Aucun fichier trouvé : {root}")
    return result


def prepare_railsem(args):
    config_path = Path(args.config).resolve()
    ids, rails, ignored = label_mapping(json.loads(config_path.read_text()))
    images = index_files(args.images, {".jpg", ".jpeg", ".png"})
    masks = index_files(args.masks, {".png"})
    if images.keys() != masks.keys():
        raise ValueError(f"Paires incomplètes : sans masque={sorted(images.keys()-masks.keys())[:10]}, sans image={sorted(masks.keys()-images.keys())[:10]}")
    groups = {}
    if args.sequences:
        with Path(args.sequences).open(newline="") as f:
            for row in csv.DictReader(f):
                if row["stem"] in groups or not row["sequence"].strip():
                    raise ValueError("Table de séquences invalide")
                groups[row["stem"]] = row["sequence"]
        if set(groups) != set(images):
            raise ValueError("La table stem,sequence doit couvrir exactement les images")
    out = new_dir(args.out)
    (out / "masks").mkdir()
    rows, inventory = [], []
    totals = np.zeros(3, dtype=np.int64)
    for stem, image_path in images.items():
        raw = mask_read(masks[stem])
        with Image.open(image_path) as image:
            if image.size != (raw.shape[1], raw.shape[0]):
                raise ValueError(f"Dimensions incompatibles : {stem}")
        binary = remap(raw, ids, rails, ignored)
        target = out / "masks" / f"{stem}.png"
        Image.fromarray(binary).save(target)
        totals += [(binary == i).sum() for i in (0, 1, 255)]
        rows.append({"image": str(image_path), "mask": str(target), "sequence": groups.get(stem, "UNKNOWN")})
        inventory.append({"stem": stem, "image_sha256": sha256(image_path), "mask_sha256": sha256(masks[stem])})
    write_manifest(out / "all.csv", rows)
    shutil.copyfile(config_path, out / "source-labels.json")
    dump(out / "preparation.json", {"task": "binary", "mapping": {name: 255 if i in ignored else int(i in rails) for name, i in ids.items()},
        "source_ids": ids, "source_config_sha256": sha256(config_path), "ignore": 255,
        "counts": dict(zip(["background", "rail", "ignore"], map(int, totals))),
        "sequences_known": bool(groups), "inventory": inventory})
    print(out / "all.csv")


def split_dataset(args):
    rows = read_manifest(args.manifest)
    if any(r["sequence"] == "UNKNOWN" for r in rows):
        if not args.allow_image_split:
            raise ValueError("Séquences inconnues. Fournir les groupes de capture ou --allow-image-split pour une validation exploratoire seulement.")
        for i, row in enumerate(rows):
            row["sequence"] = f"EXPLORATORY_IMAGE_{i:06d}"
    groups = sorted({r["sequence"] for r in rows})
    if len(groups) < 3 or not 0 < args.val_fraction < .5 or not 0 < args.test_fraction < .5:
        raise ValueError("Au moins 3 séquences, fractions val/test entre 0 et 0.5")
    random.Random(args.seed).shuffle(groups)
    nv = max(1, round(len(groups) * args.val_fraction))
    nt = max(1, round(len(groups) * args.test_fraction))
    if nv + nt >= len(groups):
        raise ValueError("Pas assez de groupes pour conserver un train non vide")
    partitions = {"val": set(groups[:nv]), "test": set(groups[nv:nv+nt]), "train": set(groups[nv+nt:])}
    data = {name: [r for r in rows if r["sequence"] in values] for name, values in partitions.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        check_disjoint(data[a], data[b])
    out = new_dir(args.out)
    for name, subset in data.items():
        write_manifest(out / f"{name}.csv", subset)
    dump(out / "split.json", {"seed": args.seed, "source_sha256": sha256(args.manifest),
         "exploratory_image_split": args.allow_image_split,
         "counts": {k: len(v) for k, v in data.items()}, "groups": {k: sorted(v) for k, v in partitions.items()}})
    print(out)


def extract_frames(args):
    if args.every <= 0 or args.max_frames < 1:
        raise ValueError("every et max-frames doivent être positifs")
    video = Path(args.video).resolve()
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Vidéo illisible : {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("FPS invalide ; convertir en vidéo à FPS constant")
        out = new_dir(args.out)
        rows, frame_id, next_time = [], 0, 0.
        while len(rows) < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            time = frame_id / fps
            if time + 1e-9 >= next_time:
                name = f"frame_{frame_id:09d}.png"
                if not cv2.imwrite(str(out / name), frame):
                    raise IOError(f"Écriture impossible : {name}")
                rows.append({"file": name, "frame_index": frame_id, "estimated_seconds": time})
                next_time = (math.floor(time / args.every) + 1) * args.every
            frame_id += 1
        if not rows:
            raise ValueError("Aucune frame décodable")
        dump(out / "extraction.json", {"source": str(video), "fps": fps, "every_seconds": args.every,
             "timestamp_method": "frame_index/fps; approximate for variable frame rate", "frames": rows})
        print(f"{len(rows)} frames : {out}")
    finally:
        cap.release()
