# Installer, vérifier et utiliser

## 1. Installer

Utiliser Python 3.12, créer un environnement et installer les dépendances comme indiqué au README. Pour retrouver les versions enregistrées sur le Mac de développement :

```bash
python -m pip install -r requirements-lock.txt -r requirements-plots.txt
python -m pip install --no-deps -e .
python -m pip check
```

Ce verrouillage est une photographie de l’environnement du 18 octobre 2026, pas une garantie de compatibilité avec tous les systèmes. Pour un autre système, partir des plages de `requirements.txt`, vérifier `pip check`, exécuter les tests et enregistrer les versions retenues. `scripts/setup.sh` propose une installation Mac/Linux avec environnement dans les données d’application. Aucun changement global de Python n’est nécessaire.

## 2. Vérifier sans données réelles

```bash
python -m pytest -q
python -m railtsv.cli check --device auto --out runs/verification_materiel
python scripts/make_smoke_data.py --out data/smoke_local
```

`check` vérifie une passe réseau/loss/gradient ; les images de smoke sont artificielles. Aucun de ces contrôles ne mesure la précision ferroviaire.

## 3. Ajouter les données locales

Placer les vidéos autorisées dans `data/raw/`, les images/masques dans une séance d’annotation et les checkpoints dans `runs/`. Les membres du groupe reçoivent ces fichiers séparément. Ne pas les ajouter à Git pour résoudre un message « fichier introuvable ».

Extraction d’une image toutes les deux secondes :

```bash
python scripts/extract_frames.py --video data/raw/video.mp4 --out data/frames_capture01 --every 2 --max-frames 100
```

Créer des polygones `couloir` et `ignore` avec la convention [ANNOTER_LE_COULOIR](ANNOTER_LE_COULOIR.md). Le flag `reviewed` n’est coché qu’après relecture et sauvegarde. Les contours humains sont la référence, mais restent imparfaits.

## 4. Exporter les annotations et entraîner

Une séance comprend `images/`, `json/` et `sequences.csv`. Consulter l’aide du script pour la structure exacte. Exemple :

```bash
python scripts/export_labelme.py --session data/ma_seance --task corridor --out data/prepared/ma_seance
python scripts/split_dataset.py --manifest data/prepared/ma_seance/all.csv --out data/splits/essai
python scripts/train.py --train data/splits/essai/train.csv --val data/splits/essai/val.csv --out runs/essai_corridor --task corridor --height 480 --width 640 --batch-size 1 --epochs 20 --device auto --init runs/public/modele/best.pt
```

Adapter les noms aux fichiers réellement produits. `--init` requiert un préapprentissage disponible. Sans celui-ci, `--pretrained` initialise l’encodeur ImageNet ; ce n’est pas la reproduction du modèle RailSem19/ego-path adapté. Les captures entières doivent être séparées entre apprentissage, développement et test. Une validation utilisée pour choisir le checkpoint n’est plus un test final indépendant.

## 5. Inférer et évaluer

```bash
python scripts/infer.py --input data/images_a_tester --checkpoint runs/modele_retenu/best.pt --out runs/inference_images --device auto
python scripts/infer_video.py --input data/raw/video.mp4 --checkpoint runs/modele_retenu/best.pt --out runs/inference_video --device auto --max-frames 300
python scripts/evaluate.py --checkpoint runs/modele_retenu/best.pt --manifest data/splits/essai/val.csv --out runs/evaluation --device auto
```

Chaque sortie doit utiliser un nouveau dossier. La vidéo est muette par défaut. `--max-frames 0` traite toute la vidéo. `--frame-telemetry` ajoute une source temporelle vérifiée ; voir `railtsv/frame_clock.py`. Ne pas activer le lissage sans comparaison : le lissage historique dégradait l’erreur et n’a pas été retenu dans la livraison V3.

Les exports génériques et le paquet V3 ont des schémas différents. La V3 est assemblée par les scripts datés de livraison ; l’interface exacte est décrite dans [INTERFACE_GABARIT](INTERFACE_GABARIT.md). Les scripts datés exigent les anciens manifests, références et résultats locaux, absents du dépôt.

## 6. Labelme

L’annotation utilise un environnement séparé : `sh scripts/setup_annotation.sh`, puis lancer Labelme avec `configs/labelme_corridor.yaml`. Les pointeurs actifs du Mac ne sont pas publiés. Le fichier `configs/annotation_batch.example.json` est un modèle à adapter, pas un lot fourni.
