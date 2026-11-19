# Localisation ferroviaire V2 — lecture GPS automatique et import CSV

Cette version ajoute deux entrées au projet : **lire les coordonnées incrustées dans une vidéo** ou **importer un CSV GPS associé à une autre vidéo**. Le raccordement au réseau, le parcours et les exports utilisent ensuite le même code. Voir [les résultats et leurs limites](docs/RESULTATS_20261119.md).

## Lancement unique et livraison connectée

Utiliser **`make all CONFIG=configurations/pipeline.local.json`** pour lancer les tests, le GPS, les jointures avec le gabarit 2D, la vidéo commune et le ZIP vérifié.

- [Mode d’emploi et rôle de chaque fichier](docs/LANCEMENT_UNIQUE.md).
- [Contrat d’intégration avec le gabarit 2D](docs/INTEGRATION_GABARIT_2D.md).

## Ce que fait le programme

1. Décode la vidéo avec PyAV et conserve chaque horodatage natif : jamais `numéro_image / fps` à la place des temps réels du fichier.
2. Lit les GPS par OCR ou importe un CSV avec des colonnes et une origine du temps explicites.
3. Conserve les textes bruts, les préparations visuelles, les positions retenues et les motifs de rejet.
4. Raccorde les observations à AfTS-Db v3 et lit l’écartement des segments. Les ambiguïtés, les réseaux absents et les écartements inconnus restent signalés.
5. Produit le tracé GeoJSON, les positions, les distances estimées et, si un CSV de rails est fourni, les données enrichies de l’équipe gabarit.

Ce module ne réentraîne pas le modèle de vision. Il utilise ses contours existants. Pour une nouvelle vidéo, la détection des rails doit être exécutée avec le programme de vision habituel si l’on veut aussi fournir leurs coordonnées.

## Essai direct sur la vidéo de trois minutes

Configurer `configurations/pipeline.local.json`, puis lancer `make all CONFIG=configurations/pipeline.local.json`. Voir le guide de lancement unique. Un nouveau dossier est créé, sans remplacer les précédents.

Le profil `configurations/ocr_tsv_03383.json` définit la zone GPS relative à la taille de l’image, le format degrés/minutes/secondes, et un prélèvement toutes les secondes, première et dernière images incluses. Il a été développé pour cette incrustation : une autre mise en page nécessite de modifier ce profil et de vérifier quelques lectures.

La lecture utilise **Apple Vision sur macOS**, sans envoi d’image à un service en ligne. L’outil Swift est compilé localement au premier lancement ; les outils de développement en ligne de commande Apple sont nécessaires. L’import CSV fonctionne aussi sans cet outil OCR.

## Pour une autre vidéo avec un CSV GPS

Le CSV peut employer des noms de colonnes différents. On les décrit dans un fichier de configuration, sans renommer les données originales. Exemple :

```json
{
  "latitude_column": "latitude",
  "longitude_column": "longitude",
  "time_column": "video_time_s",
  "time_reference": "native_pts",
  "time_offset_s": 0,
  "max_time_delta_s": 0.1,
  "speed_column": "speed_kmh",
  "speed_unit": "km/h",
  "utc_column": "utc_time"
}
```

```bash
.venv/bin/python run_localisation.py \
  --video "/chemin/vers/video.mp4" \
  --gps-csv "/chemin/vers/gps.csv" \
  --video-id NOM_DE_LA_VIDEO \
  --config configurations/mon_csv.json \
  --network donnees/africa_railways_network.gpkg \
  --out resultats/nouvelle_video
```

Ajouter `--rails-csv "/chemin/vers/rails.csv"` pour joindre les bords gauche/droit. Le CSV des rails doit contenir `video_id,frame_index,video_time_s,image_width_px,source_video_sha256,left_segments_json,right_segments_json`. L’identifiant vidéo, l’empreinte SHA-256 et les temps doivent correspondre. Des noms différents ne sont pas assimilés automatiquement, car cela pourrait mélanger deux captures.

`--frames-csv` permet d’utiliser un manifeste d’images déjà vérifié à la place d’un nouveau décodage vidéo. Le compte rendu indique alors clairement qu’il ne s’agit pas d’un nouveau décodage.

### Synchronisation : choisir l’option correcte

- `native_pts` : les temps du CSV sont ceux du fichier vidéo, avec son éventuel début non nul.
- `video_elapsed` : le CSV commence à zéro au début visible de la vidéo ; le programme ajoute le premier PTS.
- `utc` : les temps sont des dates avec fuseau horaire. Fournir `video_start_utc` (heure du premier instant vidéo) et, pour une accélération constante connue, `real_seconds_per_video_second`.
- `time_offset_s` : décalage explicite ajouté après conversion, positif si le relevé doit être associé plus tard dans la vidéo.
- `input_time_scale` : facteur des temps numériques ; n’est pas une calibration de vitesse.

Une vidéo accélérée avec accélération variable doit disposer de sa propre correspondance temps vidéo–temps réel. Ne pas supposer un facteur constant. Le CSV TSV_record possède déjà cette correspondance ; ses temps vidéo et vitesses fournies sont conservés.

Les latitudes/longitudes sont en degrés décimaux WGS84. Les séparateurs `;`, les virgules décimales et les noms de colonnes sont paramétrables. Une vitesse en km/h est convertie en m/s, mais reste étiquetée « fournie, non validée indépendamment ». Un GPS trop éloigné d’une image, deux relevés sur la même image, un temps inversé ou une identité vidéo incohérente sont rejetés et consignés.

## Comment l’OCR est contrôlé

Trois lectures sont comparées : zone d’origine, ligne GPS isolée en couleur, ligne isolée sur fond noir/blanc à partir du texte jaune. Les variantes peuvent commettre les mêmes erreurs ; leur accord n’est donc pas une certitude statistique.

Le profil DMS permet de rétablir des séparateurs perdus entre degrés, minutes et secondes, mais ne remplace pas une lettre par un chiffre et ne complète pas un chiffre absent. Les orientations N/S/E/W déterminent les signes. Le score OCR est un indicateur interne, pas une probabilité d’exactitude.

Les désaccords sans majorité, champs invalides et sauts manifestement incohérents sont rejetés. Le profil rejette aussi les excursions isolées dépassant 20 m de détour en trois secondes au plus. C’est un garde-fou à évaluer selon le véhicule et la qualité du GPS, pas une preuve qu’un recul réel est impossible. Les valeurs brutes restent disponibles et ne sont pas corrigées à la main dans les sorties automatiques.

Un relevé absent ne devient pas un GPS mesuré par interpolation. Seul l’avancement peut être interpolé entre deux observations, au maximum sur cinq secondes par défaut, avec un statut explicite. Aucune extrapolation avant/après les observations n’est faite.

## Les fichiers à donner au gabarit

Dans `GABARIT/` du résultat :

- `RAILS_GAUCHE_DROIT_GPS.csv` : une ligne par image du CSV de rails d’origine. Tous les champs initiaux, les contours et les vitesses déjà présentes sont conservés. Les nouvelles observations sont dans `gps_latitude_deg`, `gps_longitude_deg`, `gps_observation_source`, etc.
- `POINTS_RAILS_METRES.csv` : une ligne par paire de points gauche/droit. Les noms `nominal` et `design` sont génériques ; la valeur de chaque écartement est indiquée dans ses propres colonnes.
- `CONTROLES_EXPORT.json` et `LIRE_MOI.txt` : compteurs, sens des colonnes et limites.

Les coordonnées `latitude_deg`/`longitude_deg` historiques d’un ancien CSV restent inchangées ; les nouvelles données V2 sont préfixées `gps_`. La colonne `route_progress_m_per_video_second` est une moyenne d’avancement par seconde **de vidéo**. Elle n’est pas une vitesse instantanée réelle. Les vitesses GPS fournies restent séparées dans `gps_speed_provided_m_s` et les anciens champs `train_speed_*` sont conservés.

L’erreur réelle de la détection, l’erreur physique et l’incertitude ne sont pas inventées : les champs correspondants restent vides lorsqu’aucune référence n’existe.

## Écartements et mètres

Le réseau disponible couvre l’Afrique. Un CSV situé ailleurs peut être importé, mais le programme signalera `no_network_coverage` et laissera la ligne, le parcours cartographique et l’écartement nominal non résolus. Il n’attribue pas un écartement nigérian à une vidéo belge. Le défaut de dimensionnement reste 1,676 m, paramétrable par `--design-gauge-m`.

L’échelle d’une coupe de l’image reste `g / (x_droit - x_gauche)`. Les positions transversales sont relatives à la verticale médiane de l’image, pas au centre physique du véhicule. Y reste en pixels. La largeur vaut g par construction ; on ne mesure pas une variation réelle de l’écartement. Les couloirs de moins de 2 % de la largeur d’image n’ont pas de conversion métrique.

La calibration caméra et des contrôles terrain restent nécessaires pour des mesures physiques validées. La géolocalisation du train n’est pas automatiquement celle d’un obstacle visible devant lui.

## Limites du raccordement

Le réseau représente des lignes et peut simplifier les voies physiques. Une séparation de segments au sein d’une même ligne, sur un nœud de degré deux, n’est pas traitée comme un aiguillage. Un nœud à plusieurs branches ou des voies parallèles reste ambigu.

Le prototype utilise une fenêtre régionale (marge 20 km) et refuse les observations situées à plus de 100 km de leur centre. Un long trajet doit être découpé en fenêtres et suivi temporellement. Ce suivi embarqué continu n’est pas encore implémenté. Le plus court chemin cartographique n’est pas une preuve de l’absence de détours ou de reculs entre observations.

## Installation et vérifications

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_network.py
.venv/bin/python -m unittest discover -s tests -v
```

Les données publiques sont téléchargées avec empreinte fixe. Les GPS privés, vidéos, préparations OCR, résultats, environnements et archives sont exclus de Git par `.gitignore`. Aucun push n’est exécuté par les lanceurs.

`rail_localisation.py` fournit le moteur de carte ; `gps_inputs.py` normalise les CSV et analyse les textes ; `video_timeline.py` conserve les temps natifs ; `gps_ocr.py` et `ocr_vision.swift` effectuent la lecture locale ; `build_handoff.py` enrichit les rails ; `run_localisation.py` relie les étapes.

Sources : [AfTS-Db v3](https://zenodo.org/records/17861120), CC BY 4.0, Colombo et al. ; réseau original Marcus Young et © contributeurs OpenStreetMap. [Apple Vision](https://developer.apple.com/documentation/vision/vnrecognizetextrequest) ; [temps natifs PyAV](https://pyav.basswood.io/docs/stable/api/time.html). Carte publiée en 2025 ; vidéo 03383 datée 2017, sans précision terrain démontrée.
