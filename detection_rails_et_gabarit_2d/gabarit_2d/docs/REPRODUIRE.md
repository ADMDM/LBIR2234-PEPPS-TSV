# Installer et reproduire

## Pour l’équipe gabarit : lire la livraison vérifiée

La solution la plus simple consiste à récupérer le paquet final déjà vérifié dans le partage du groupe. Ses CSV/TXT et sa vidéo sont directement utilisables ; il n’est pas nécessaire de refaire l’entraînement ou les expériences de validation.

Pour lire les contours avec Python, garder `lire_contours.py` et `corridor_geometry.py` ensemble et installer seulement `numpy` et `opencv-python` :

```bash
python -m pip install numpy opencv-python
python lire_contours.py --frame-file livraison/TXT/frame_000000.txt
```

Le TXT est un document JSON, avec temps, dimensions, liste d’objets et éventuelles hypothèses à vérifier. Voir [l’interface](INTERFACE_GABARIT.md).

## Installation pour travailler sur le programme

Depuis la racine du dépôt :

```bash
cd gabarit_2d
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s . -p 'test_*.py'
```

Les versions correspondent à Python 3.12 sur Mac Apple Silicon. L’installation sur une autre plateforme peut nécessiter des versions compatibles de PyTorch et d’OpenCV ; elle n’est pas présentée comme déjà vérifiée. Les tests emploient des données synthétiques temporaires, sans télécharger de vidéo, d’annotation ou de modèle.

## Produire la livraison retenue : référence + hypothèses train à vérifier

Récupérer la vidéo source native, le dossier V1 **complet et vérifié** et les trois dossiers de preuves de validation. Le dossier V1 doit notamment contenir tous les TXT, la vidéo avec superposition, les CSV, `MANIFESTE.json`, `VERIFICATION.json` et son audit. Un petit ZIP destiné à lire les résultats ne remplace pas forcément ce dossier de travail.

Les commandes utilisent des chemins d’exemple à remplacer. Chaque sortie doit être un nouveau dossier :

```bash
python finalize_guard_only.py \
  --baseline /chemin/absolu/vers/V1_complete \
  --source /chemin/absolu/vers/video_source.AVI \
  --rejection-report /chemin/absolu/vers/VALIDATION_TILES640/METRICS.json \
  --rejection-report /chemin/absolu/vers/VALIDATION_SELECTIVE/SELECTION.json \
  --rejection-report /chemin/absolu/vers/VALIDATION_CONSENSUS/SELECTION.json \
  --out /chemin/absolu/vers/nouvelle_livraison_corrigee

python verify_guard_only.py \
  --delivery /chemin/absolu/vers/nouvelle_livraison_corrigee
```

Après vérification, créer le paquet destiné au groupe :

```bash
python package_guard_video.py \
  --run /chemin/absolu/vers/nouvelle_livraison_corrigee \
  --notice docs/NOTICE_EQUIPE_GABARIT_V2.txt \
  --code . \
  --out /chemin/absolu/vers/nouveau_paquet.zip
```

Le paquet et son reçu doivent être nouveaux. Le script contrôle les empreintes des données et des trois expériences rejetées avant la création, puis relit chaque entrée du ZIP.

Cette étape ne charge pas de nouveau réseau et ne refait pas le suivi. Les observations, coordonnées, identifiants et lignes CSV retenues restent ceux de V1. La règle de vérification retire seulement les hypothèses train concernées de la liste principale et conserve leurs masques complets dans les fichiers à vérifier.

`--max-frames 150` permet un pilote ; le résultat est déclaré partiel. Omettre cette option pour traiter toute la vidéo. Le temps de ce retraitement ne mesure pas la vitesse de la détection d’origine.

**Les preuves expérimentales gardent les chemins des entrées qui ont réellement servi.** Le programme vérifie ces fichiers et leurs empreintes, y compris les caches de validation. Copier seulement trois rapports JSON sur une autre machine ne suffit donc pas. `--source` autorise une copie identique de la vidéo par SHA ; cela ne relocalise pas automatiquement les autres preuves. Pour exploiter les résultats sur un autre ordinateur, préférer le paquet final et son lecteur. Reproduire les calculs exige une préparation documentée des données et de leurs preuves locales.

## Refaire la référence V1

Récupérer le dossier de rails contenant `SOURCE_ET_PARAMETRES.json`, `AXE_ET_LIMITES.csv` et `FRAMES.csv`, la vidéo native et deux checkpoints de **segmentation d’instances** : public et déchets. Des poids de détection avec rectangles seuls ne conviennent pas.

```bash
python infer_gabarit_2d.py \
  --rail-delivery /chemin/absolu/vers/rails_verifies \
  --source /chemin/absolu/vers/video_source.AVI \
  --public-model /chemin/absolu/vers/yolo26s-seg.pt \
  --litter-model /chemin/absolu/vers/best.pt \
  --public-imgsz 960 --litter-imgsz 960 \
  --public-conf 0.25 --litter-conf 0.15 \
  --device auto --out /chemin/absolu/vers/nouvelle_V1

python verify_gabarit_2d.py --delivery /chemin/absolu/vers/nouvelle_V1
```

`--device auto` utilise MPS s’il est disponible, sinon CPU. Une demande explicite de MPS indisponible échoue clairement. La copie publiée importe `rail_io.py` localement ; elle ne nécessite pas l’ancien dossier de rendu privé.

## Préparer et entraîner les déchets

Le jeu employé est [TACO](https://github.com/pedropro/TACO), avec les polygones officiels regroupés en une classe `dechet`. La référence exacte du dépôt, les adresses des métadonnées et leurs empreintes sont dans `provenance/TACO_SOURCE.json`. Les photos ne sont pas republiées dans ce dépôt.

Créer dans un répertoire de données extérieur à Git `01_SOURCES/TACO_20261108/`. Y placer le `annotations.json` officiel correspondant à la référence documentée et une copie de `provenance/TACO_SOURCE.json` nommée `SOURCE.json`. Le script de préparation utilise ensuite les liens officiels présents dans les annotations pour télécharger les photos. La disponibilité de ces liens peut évoluer : les fichiers manquants et exclusions sont consignés, sans inventer de remplacement.

```bash
python prepare_taco_seg.py --root /chemin/absolu/vers/donnees --mode all
python verify_taco_seg.py \
  --dataset /chemin/absolu/vers/donnees/02_DONNEES/TACO_SEG_20261108

python train_litter.py \
  --data /chemin/absolu/vers/donnees/02_DONNEES/TACO_SEG_20261108/data.yaml \
  --model /chemin/absolu/vers/yolo26n-seg.pt \
  --epochs 12 --batch 2 --imgsz 640 --device auto \
  --out /chemin/absolu/vers/nouveau_entrainement
```

Ne pas utiliser `--rebuild` sur les données gelées d'une expérience. Une nouvelle acquisition ou un nouvel entraînement peut produire un autre ensemble ou d'autres poids : il faut alors refaire une validation documentée. Cela ne recrée pas automatiquement les scores de l'expérience précédente.

Les annotations COCO d'origine sont conservées pour l'évaluation : certaines images ont plusieurs contours pour le même objet. La conversion utilisée pour l'entraînement YOLO peut introduire une liaison entre composantes ; l'évaluation dédiée ne doit pas prendre cette approximation pour la référence géométrique d'origine.

## Rejouer les expériences rejetées

Ces commandes servent à examiner le travail de recherche, **pas à recommander les réglages testés**. Le checkpoint, la grille et les critères sont gelés pour ces expériences. Les données et le dossier d’évaluation globale VAL doivent correspondre exactement à leurs preuves ; le script n’accepte pas discrètement un autre modèle ou un autre split.

```bash
python evaluate_tiled_validation.py \
  --dataset /chemin/absolu/vers/TACO_SEG_20261108 \
  --baseline-evaluation /chemin/absolu/vers/VALIDATION_TACO_960_B1 \
  --model /chemin/absolu/vers/best.pt \
  --tile-size 640 --device auto \
  --out /chemin/absolu/vers/VALIDATION_TILES640

python evaluate_selective_validation.py \
  --prior-validation /chemin/absolu/vers/VALIDATION_TILES640 \
  --out /chemin/absolu/vers/VALIDATION_SELECTIVE

python evaluate_consensus_validation.py \
  --prior-validation /chemin/absolu/vers/VALIDATION_TILES640 \
  --rejected-selective /chemin/absolu/vers/VALIDATION_SELECTIVE/SELECTION.json \
  --out /chemin/absolu/vers/VALIDATION_CONSENSUS
```

La première commande effectue l’inférence des tuiles. Les deux suivantes réutilisent son cache sur CPU, sans nouveau réseau. Chaque protocole est écrit avant les métriques. Un rapport sans configuration admissible contient `selected_config: null` ; il ne faut pas contourner ce résultat. Les rapports originaux sont conservés, sans réécriture pour faire apparaître un gain.

`refine_video.py` permet de reconstruire la vidéo expérimentale avec toutes les tuiles, à partir de V1, du modèle de déchets et des options `--tile-size 640 --tile-overlap 0.25 --imgsz 960 --conf 0.15`. Cette vidéo a produit plus de contours, mais ne représente pas la livraison retenue.

Les scripts de préparation des paquets refusent les sorties incomplètes ou dont les empreintes ne correspondent pas. La vidéo et les données de partage restent hors Git.
