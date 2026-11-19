# Détection des rails et gabarit 2D

Ce dossier regroupe le code du projet. La localisation est intégrée dans la
partie rails. Les données vidéo, les GPS personnels, les annotations et les
poids des modèles se conservent séparément ; leurs chemins sont renseignés
dans une configuration locale.

## Les trois éléments à connaître

- `rails_deeplab/` : apprentissage et détection du couloir entre les faces
  intérieures des rails. `localisation/` lit un GPS incrusté ou un CSV GPS.
  `presentation/` réunit les résultats, dessine la carte et crée les vidéos.
- `gabarit_2d/` : prédiction des silhouettes d'objets et intersection avec le
  couloir. Les contours sont de vrais masques ; une boîte rectangulaire ne les
  remplace pas. Le recouvrement est une relation 2D, pas une alerte de collision.
- `lancer.py` et `Makefile` : un point d'entrée pour appeler les étapes dans
  l'ordre, tout en gardant des programmes séparés et testables.

## Démarrage

Le modèle rails est DeepLabV3+ avec ResNet50. Le checkpoint sélectionné résulte
du préapprentissage ego-path puis de l'adaptation TSV. Ce changement de
présentation ne réentraîne pas ce modèle. Les objets utilisent les modèles
YOLO de segmentation d'instances public et spécialisé sur les déchets TACO.

1. Utiliser Python 3.12 et installer FFmpeg/ffprobe. Sur Apple Silicon, utiliser un Python arm64 pour MPS.
2. Installer `requirements-inference.txt` et `requirements-presentation.txt`
   dans deux environnements distincts. La configuration peut désigner
   deux environnements : `inference_python` pour PyTorch/YOLO et
   `presentation_python` pour PyAV, OpenCV, Pillow, NumPy, Shapely et pyproj.
   Cette livraison a été exécutée avec ces deux environnements séparés.
3. Copier `configurations/projet.example.json` en `projet.local.json` et y
   renseigner les vidéos, les modèles et des dossiers de sortie nouveaux.
   Ne pas publier le fichier local : il peut contenir des chemins personnels.
4. Vérifier les sources, puis exécuter :

```sh
make check CONFIG=configurations/projet.local.json
make all CONFIG=configurations/projet.local.json
```

Les options `--video-id` de `lancer.py` permettent de sélectionner une vidéo.
Les étapes de prédiction terminées sont réutilisées après contrôle de leur
identité. Les fichiers de sortie existants ne sont pas écrasés. Un rendu à
refaire doit utiliser un nouveau dossier de sortie.

Création des environnements sur une machine neuve :

```sh
python3 -m venv .venv-inference
.venv-inference/bin/python -m pip install -r requirements-inference.txt
python3 -m venv .venv-presentation
.venv-presentation/bin/python -m pip install -r requirements-presentation.txt
```

Sur un Mac Apple Silicon, ajouter `"python_arch": "arm64"` à la configuration
si Python est une distribution universelle. Ce paramètre s'omet sur Linux.
Les anciens scripts d'entraînement ont leurs dépendances dans leur module ;
ils ne sont pas relancés par cette commande d'inférence et de présentation.

Pour obtenir l'empreinte d'une source, vérifier sa cadence et lire une position
éventuelle dans ses métadonnées :

```sh
python rails_deeplab/presentation/inspect_source.py --source data/video.mp4 --out inspection.local.json
```

L'option `--normalize-output data/video_redressee.mp4` applique l'orientation du
fichier dans une nouvelle vidéo. Les instants et le nombre d'images sont ensuite
contrôlés. Une position de métadonnées sans horodatage est uniquement un repère
fixe : elle n'est pas transformée en trajet GPS par frame.

## GPS

Le chemin `gps_result` pointe vers un résultat GPS terminé. Pour le créer :

```sh
python rails_deeplab/localisation/run_localisation.py --help
```

Ce programme choisit entre OCR local Apple Vision (`--ocr`) et import CSV
(`--gps-csv`) ; il vérifie les temps natifs de la vidéo. L'OCR Apple Vision est
spécifique à macOS. L'import CSV et le reste du traitement sont portables.

La carte africaine utilise AfTS-Db v3 (Colombo, Pant, Young, Thomas, Russell,
Verschuur et Hall) :
https://zenodo.org/records/17861120 (CC BY 4.0).
La carte belge de comparaison utilise les voies principales d'Infrabel :
https://opendata.infrabel.be/explore/dataset/geosporen/ (CC0).
Les réseaux sont traités localement : aucune trace GPS n'est envoyée au
fournisseur de cartes. La carte belge montre le GPS sur le réseau, sans
certifier la voie exacte. Sans GPS, aucun trajet n'est inventé.

## Lissage et coordonnées

Les limites gauche et droite sont filtrées ensemble, de façon causale, par un
filtre adaptatif inspiré de One Euro. L'axe noir est leur milieu horizontal.
Paramètres usuels : coupure minimale 0,8 Hz, beta 15 et réinitialisation après
une lacune de plus de 0,15 s. Pour la vidéo à 5 images/s, la lacune limite est
0,35 s ; les limites brutes restent dans le CSV. Les grands sauts réinitialisent
le filtre. Les segments absents ne sont jamais inventés.

Les temps d'origine sont conservés, y compris les lacunes de décodage. La vidéo
de présentation peut redimensionner l'image ou masquer des bandes noires ;
les CSV conservent les pixels source et donnent la transformation d'affichage.
Un clip filmé de côté doit être redressé avant cette analyse, avec la
transformation et la nouvelle empreinte de la source documentées.

## Ce que reçoit l'équipe gabarit

Un ZIP par vidéo contient seulement : la vidéo annotée, les CSV et
`LIRE_MOI.txt`. Les journaux, preuves, TXT intermédiaires et scripts restent
dans le projet. `RAILS.csv` contient une ligne par image, avec les deux limites
et l'axe. `POINTS_RAILS.csv` détaille les sections ; `OBJETS.csv` et
`POINTS_OBJETS.csv` décrivent les observations et les contours, lorsqu'ils ont
été calculés. Les hypothèses de train ambiguës restent dans `A_VERIFIER.csv`.

La conversion transversale utilise un écartement cartographié lorsqu'il est
résolu, ou l'hypothèse dimensionnante de 1,676 m. Ce n'est pas une calibration
3D, une mesure de l'écartement ou une distance à un objet. Les erreurs réelles
inconnues restent vides. Les scores historiques sur 12 images de développement
ne deviennent pas des erreurs mesurées sur toutes les nouvelles vidéos.

## Tests

```sh
make test PYTHON=.venv-presentation/bin/python
.venv-inference/bin/python -m pytest gabarit_2d
.venv-presentation/bin/python -m unittest discover -s rails_deeplab/localisation/tests
```

Les vérifications de sortie relisent les horodatages de toutes les images et
vérifient les empreintes de tous les fichiers du ZIP. Ce sont des contrôles
logiciels et de cohérence, pas une nouvelle mesure de précision terrain.
