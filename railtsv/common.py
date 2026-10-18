from pathlib import Path
import csv
import hashlib
import json
import os
import random
import time
import numpy as np
from PIL import Image
import torch

TASKS = {"binary": ["background", "rail"],
         "tsv": ["background", "rail_left", "rail_right"],
         "corridor": ["background", "couloir"],
         "railsem_track": ["background", "track_area"],
         "railsem_egopath": ["background", "ego_path"]}


def device(name="auto"):
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS indisponible dans ce processus. Utiliser --device cpu.")
    return torch.device(name)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def new_dir(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _image_array(path, rgb=False):
    # A file may be temporarily unreadable while a local/cloud provider hydrates it.
    # Retry I/O only; never replace an invalid mask or fabricate pixels.
    for attempt in range(3):
        try:
            with Image.open(path) as im:
                if rgb:
                    return np.array(im.convert("RGB"))
                if im.mode not in ("P", "L", "I", "I;16"):
                    raise ValueError(f"Masque d'IDs requis, pas une image RGB : {path}")
                return np.array(im)
        except OSError:
            if attempt == 2:
                raise
            time.sleep(.2 * (attempt + 1))


def rgb_read(path):
    return _image_array(path, rgb=True)


def mask_read(path, allowed=None):
    mask = _image_array(path)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"Masque entier 2D requis : {path}")
    if allowed is not None and not set(np.unique(mask)).issubset(set(allowed)):
        raise ValueError(f"Labels invalides : {path} : {np.unique(mask)}")
    return mask


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"image", "mask", "sequence"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requis : image,mask,sequence")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifeste vide : {path}")
    seen = set()
    for row in rows:
        if not row["sequence"].strip():
            raise ValueError("Chaque image doit avoir une sequence explicite")
        for field in ("image", "mask"):
            row[field] = str((path.parent / row[field]).resolve())
            if not Path(row[field]).is_file():
                raise FileNotFoundError(row[field])
        if row["image"] in seen:
            raise ValueError(f"Image dupliquée : {row['image']}")
        seen.add(row["image"])
    return rows


def write_manifest(path, rows):
    path = Path(path).resolve()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "mask", "sequence"])
        writer.writeheader()
        for row in rows:
            portable = {k: row[k] for k in writer.fieldnames}
            for key in ("image", "mask"):
                if Path(portable[key]).is_absolute():
                    portable[key] = os.path.relpath(portable[key], path.parent)
            writer.writerow(portable)


def check_disjoint(train, val):
    if {r["sequence"] for r in train} & {r["sequence"] for r in val}:
        raise ValueError("Fuite : une séquence figure dans les deux ensembles")
    for field in ("image", "mask"):
        if {r[field] for r in train} & {r[field] for r in val}:
            raise ValueError(f"Fuite : même fichier {field} dans train et val")
    # Identical images with different filenames are also a leak.
    if {sha256(r["image"]) for r in train} & {sha256(r["image"]) for r in val}:
        raise ValueError("Fuite : images identiques entre train et val (SHA256)")
