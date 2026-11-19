"""Export reviewed Labelme polygons to PNG IDs without importing Labelme."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
from railtsv.common import new_dir, dump, sha256, write_manifest

LABELS = {"rail_left": 1, "rail_right": 2, "ignore": 255}
TASK_LABELS = {"tsv": LABELS, "corridor": {"couloir": 1, "ignore": 255}}


def rasterize(data, size, task="tsv"):
    if task not in TASK_LABELS:
        raise ValueError(f"Mode d’annotation inconnu : {task}")
    labels = TASK_LABELS[task]
    declared_task = data.get("task")
    if declared_task is not None and declared_task != task:
        raise ValueError("Le mode du JSON ne correspond pas au mode du dataset")
    if data.get("flags", {}).get("reviewed") is not True:
        raise ValueError("Annotation non validée : cocher reviewed après vérification complète")
    w, h = size
    if (data.get("imageWidth"), data.get("imageHeight")) != (w, h):
        raise ValueError("Dimensions JSON/image incompatibles")
    layers = {label: Image.new("L", size) for label in labels}
    if not data.get("shapes"):
        raise ValueError("Aucun polygone ; une image négative doit être traitée explicitement")
    for shape in data["shapes"]:
        label = shape.get("label")
        points = np.asarray(shape.get("points"), dtype=float)
        if label not in labels or shape.get("shape_type", "polygon") != "polygon":
            raise ValueError("Utiliser uniquement les polygones " + ", ".join(labels))
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("Polygone invalide")
        # Labelme uses continuous canvas edges, including x=width/y=height.
        # Accept those edges and map them onto the last pixel of the mask.
        if (points[:,0] < 0).any() or (points[:,0] > w).any() or (points[:,1] < 0).any() or (points[:,1] > h).any():
            raise ValueError("Polygone hors image")
        points = np.clip(points, [0, 0], [w-1, h-1])
        ImageDraw.Draw(layers[label]).polygon([tuple(x) for x in points], fill=1)
    layers = {k: np.array(v).astype(bool) for k,v in layers.items()}
    if task == "tsv" and (layers["rail_left"] & layers["rail_right"] & ~layers["ignore"]).any():
        raise ValueError("Les rails gauche/droit se chevauchent hors ignore")
    mask = np.zeros((h,w),np.uint8)
    for label,value in labels.items():
        mask[layers[label]] = value
    return mask


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session",default="data/annotation_session_v1")
    p.add_argument("--out",required=True)
    p.add_argument("--task",choices=TASK_LABELS,default="tsv")
    args=p.parse_args()
    session=Path(args.session).resolve()
    with (session/'sequences.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    groups={r['stem']:r['sequence'] for r in rows}
    if len(groups)!=len(rows) or any(not x.strip() for x in groups.values()):
        raise ValueError('Table de séquences invalide')
    annotations=sorted((session/'json').glob('*.json'))
    if not annotations:
        raise ValueError('Aucune annotation sauvegardée : tracer les polygones dans Labelme, puis enregistrer')
    items=[]
    for path in annotations:
        candidates=list((session/'images').glob(path.stem+'.*'))
        if len(candidates)!=1 or path.stem not in groups:
            raise ValueError(f'Image ou séquence manquante/ambiguë : {path.name}')
        source=candidates[0]
        data=json.loads(path.read_text())
        with Image.open(source) as im:
            mask=rasterize(data,im.size,args.task)
        items.append((path,source,mask))
    out=new_dir(args.out)
    (out/'masks').mkdir(); (out/'previews').mkdir()
    manifests=[]; report=[]
    for path,source,mask in items:
        target=out/'masks'/f'{path.stem}.png'
        Image.fromarray(mask).save(target)
        with Image.open(source) as im:
            rgb=np.array(im.convert('RGB'))
        colored=rgb.copy()
        for value,color in [(1,[0,220,255]),(2,[255,150,0]),(255,[220,0,220])]:
            selected=mask==value
            colored[selected]=(.5*rgb[selected]+.5*np.array(color)).astype(np.uint8)
        Image.fromarray(colored).save(out/'previews'/f'{path.stem}.png')
        manifests.append(dict(image=str(source),mask=str(target),sequence=groups[path.stem]))
        report.append(dict(image=source.name,json_sha256=sha256(path),image_sha256=sha256(source),
                           counts={str(i):int((mask==i).sum()) for i in [0,*TASK_LABELS[args.task].values()]}))
    write_manifest(out/'all.csv',manifests)
    dump(out/'export.json',dict(task=args.task,images=report,warning='UNKNOWN sequences must be resolved before a meaningful train/val/test split'))
    print(f'{len(items)} masques exportés dans {out}')


if __name__=='__main__':
    main()
