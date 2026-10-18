"""Premier apprentissage sur dix annotations, sans validation indépendante."""
import argparse
import csv
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image
from scripts.export_labelme import rasterize


def prepare(session, limit):
    paths = sorted((session / "json").glob("*.json"))[:limit]
    if len(paths) < limit:
        raise ValueError(f"Il faut {limit} annotations sauvegardées ; {len(paths)} trouvées.")
    missing = [p.name for p in paths if json.loads(p.read_text()).get("flags", {}).get("reviewed") is not True]
    if missing:
        raise ValueError("Avant l'essai, cocher reviewed puis Save sur ces images :\n" + "\n".join(missing))
    with (session / "sequences.csv").open(newline="") as f:
        groups = {r["stem"]: r["sequence"] for r in csv.DictReader(f)}
    items = []
    for path in paths:
        raw = path.read_bytes()
        data = json.loads(raw)
        candidates = list((session / "images").glob(path.stem + ".*"))
        if len(candidates) != 1 or path.stem not in groups:
            raise ValueError(f"Image ou groupe manquant : {path.name}")
        image = candidates[0]
        with Image.open(image) as im:
            rasterize(data, im.size)
        items.append((path, image, raw, groups[path.stem]))
    return items


def run_command(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def select_video(session, items, override=None):
    if override is not None:
        source = override.resolve()
    else:
        metadata = json.loads((session / "session.json").read_text())
        sequences = {sequence for _, _, _, sequence in items}
        candidates = [v for v in metadata['videos'] if v['sequence'] in sequences]
        if not candidates:
            raise ValueError("Vidéo du lot introuvable ; préciser --video")
        source = (session / candidates[0]['video']).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def gallery(out, training_stems, video_relative='video/rails_annotes.mp4'):
    cards = []
    for folder in sorted((out / "predictions").iterdir()):
        if not folder.is_dir():
            continue
        status = "Image utilisée pour apprendre" if folder.name in training_stems else "Image non utilisée pour apprendre — contrôle visuel"
        with Image.open(folder / "overlay.png") as im:
            im.thumbnail((960, 540))
            im.save(folder / "apercu.jpg", quality=90)
        rel = f"predictions/{folder.name}/apercu.jpg"
        cards.append(f'<article><h2>{html.escape(folder.name)}</h2><p>{status}</p><a href="predictions/{folder.name}/overlay.png"><img src="{rel}" alt="Prédiction des rails"></a></article>')
    page = '''<!doctype html><html lang="fr"><meta charset="utf-8"><title>Premier essai des rails</title>
<style>body{font:17px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#182633;background:#f4f6f8}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px}article{background:white;padding:16px;border-radius:10px}h2{font-size:15px;overflow-wrap:anywhere}img{width:100%;height:auto}p{line-height:1.5}</style>
<h1>Premier essai de segmentation des rails</h1><p>Cyan : rail gauche. Orange : rail droit. Clique sur une image pour l'agrandir.</p>
<p>Ce modèle a appris sur un petit lot. Les images utilisées pour apprendre permettent de vérifier cet apprentissage ; les autres donnent un premier aperçu visuel. Aucun score de fiabilité sur une vidéo indépendante n'est calculé ici.</p><main>'''
    video = ''
    if (out / video_relative).exists():
        escaped=html.escape(video_relative,quote=True)
        video = f'<section style="grid-column:1/-1"><h2>Vidéo avec les rails prédits</h2><video controls playsinline preload="metadata" style="width:100%" src="{escaped}"></video><p><a href="{escaped}">Ouvrir la vidéo MP4</a></p></section>'
    (out / "VOIR_LES_RESULTATS.html").write_text(page + video + "\n".join(cards) + "</main></html>", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=ROOT / "data/annotation_videos_v1")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--video", type=Path, help="Autre vidéo locale à traiter ; sinon vidéo du lot annoté")
    parser.add_argument("--max-video-frames", type=int, default=0, help="0=vidéo complète ; limiter seulement pour un extrait d'essai")
    args = parser.parse_args()
    if args.limit < 1 or args.epochs < 1 or args.max_video_frames < 0:
        parser.error("limit et epochs doivent être positifs, max-video-frames >= 0")
    items = prepare(args.session.resolve(), args.limit)
    print(f"{len(items)} annotations validées et prêtes.", flush=True)
    if args.check_only:
        return
    source_video = select_video(args.session.resolve(), items, args.video)
    # ImageNet weights are already cached on this Mac. No dataset download.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    out = ROOT / "runs" / (datetime.now().strftime("essai_10_images_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
    copied = out / "session"
    (copied / "images").mkdir(parents=True, exist_ok=False)
    (copied / "json").mkdir()
    provenance = []
    with (copied / "sequences.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "sequence"])
        writer.writeheader()
        for annotation, image, raw, sequence in items:
            (copied / "json" / annotation.name).write_bytes(raw)
            shutil.copy2(image, copied / "images" / image.name)
            writer.writerow({"stem": annotation.stem, "sequence": sequence})
            provenance.append({"annotation": str(annotation), "sha256": hashlib.sha256(raw).hexdigest(), "sequence": sequence})
    (out / "sources.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("Étape 1/4 : transformer tes polygones en masques.", flush=True)
    run_command(ROOT / "scripts/export_labelme.py", "--session", copied, "--out", out / "masques")
    print(f"Étape 2/4 : apprendre sur ces {len(items)} images ({args.epochs} passages).", flush=True)
    run_command(ROOT / "scripts/train.py", "--train", out / "masques/all.csv", "--out", out / "modele",
                "--task", "tsv", "--training-only-demo", "--pretrained", "--device", "auto",
                "--height", "384", "--width", "640", "--batch-size", "1", "--epochs", args.epochs)
    print("Étape 3/4 : prédire les rails sur les 24 images extraites des vidéos.", flush=True)
    run_command(ROOT / "scripts/infer.py", "--input", args.session.resolve() / "images",
                "--checkpoint", out / "modele/last.pt", "--out", out / "predictions", "--device", "auto")
    print("Étape 4/4 : créer la vidéo MP4 avec les rails prédits sur chaque image.", flush=True)
    run_command(ROOT / "scripts/infer_video.py", "--input", source_video,
                "--checkpoint", out / "modele/last.pt", "--out", out / "video",
                "--device", "auto", "--max-frames", args.max_video_frames)
    gallery(out, {p.stem for p, _, _, _ in items})
    print(f"Terminé. Ouvre : {out / 'VOIR_LES_RESULTATS.html'}", flush=True)
    if sys.platform == "darwin":
        subprocess.run(["open", str(out / "VOIR_LES_RESULTATS.html")], check=False)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"\nEssai arrêté : {error}", file=sys.stderr)
        sys.exit(1)
