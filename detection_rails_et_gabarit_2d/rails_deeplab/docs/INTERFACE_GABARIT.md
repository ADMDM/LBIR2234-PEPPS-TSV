# Échange avec l’équipe gabarit

Le paquet de travail actuel est `GABARIT_V3_METRES_ESTIMES_20261018.zip`, conservé hors GitHub. Il contient notre sortie retenue sur **TSV_record** et **03676**. L’autre équipe de détection traite les mêmes images pour comparer ou fusionner ses sorties avec les nôtres.

## Fichiers et clés

- `FRAMES_VITESSE_VECTEUR.csv` : une ligne par frame, temps, vitesse disponible, statuts et liste des points de l’axe.
- `AXE_MEDIAN.csv` : points détaillés, limites gauche/droite, largeur et erreurs uniquement lorsqu’une référence existe.
- `IMAGES_COMMUNES/` : PNG originaux communs aux équipes.
- `SOURCE_ET_DECOUPAGE.json` : dimensions, cadence, source et transformation temporelle.

La clé est `video_id + frame_index`, puis `segment_id + point_index` ou la hauteur `y_px`. L’origine est en haut à gauche ; x augmente à droite et y vers le bas. Utiliser les dimensions originales, pas celles du réseau ni d’un lecteur vidéo redimensionné.

| Colonnes | Sens |
|---|---|
| frame_index / source_frame_index | Numéro normalisé et correspondance avec l’image décodée |
| video_time_s / real_elapsed_s | Temps de lecture et temps physique disponible |
| x_px, y_px | Axe dans le repère source |
| left_x_px, right_x_px, width_px | Limites et largeur sur la même ligne |
| axis_segments_json | Liste ordonnée de segments de points ; ne pas relier les coupures |
| tangent_dx, tangent_dy | Direction locale unitaire dans l’image, sans signification 3D automatique |
| train_speed_km_h / train_speed_m_s | Vitesse avec statut et provenance |
| error_* | Erreur connue uniquement avec référence correspondante |
| *_assumed | Estimation conditionnelle, pas une mesure physique validée |

Une cellule vide signifie inconnu. Ne jamais remplacer une vitesse inconnue par zéro ni une erreur inconnue par « parfait ». Conserver les rejets et indicateurs.

## Convention des deux vidéos

TSV_record : 1280 × 720, 7 608 frames à 5 fps de lecture. Le temps physique vient du relevé associé. 36 vitesses manquent et 831 frames portent des indicateurs à examiner. La zone analysée x=370..910 est replacée dans l’image source entière avant export.

03676 : 1920 × 1080, 5 399 frames normalisées à 30 fps à partir de 5 305 images décodées ; 94 répétitions sont identifiées. Normalisation : `fps=fps=30:start_time=0:round=near`. Les 19 ancrages GPS incrustés donnent des vitesses moyennes d’intervalles, non une mesure instantanée validée. Les 28 dernières frames restent sans vitesse.

La frame normalisée 1375 de 03676 correspond à la source 1350. Pour commencer, comparer cette frame puis 1375 à 1404 ; pour TSV_record, comparer 50 à 59. Les PNG et correspondances du paquet font autorité.

## Exemple sans donnée réelle

`examples/gabarit_synthetique/` contient une petite séquence inventée pour comprendre le format. Ce n’est pas un export du modèle ni une validation. Le lecteur autonome `scripts/fiabilisation_20261018/lire_une_frame.py` lit le paquet réel.

## Mètres

Le paquet V3 fournit une approximation latérale locale et un avancement intégré à partir des vitesses valides. Il ne fournit pas une profondeur ni une erreur 3D calibrée. Les hypothèses d’écartement du bilan ne doivent pas être copiées sur une autre voie sans confirmation. Pour la rectification, confirmer caméra, repères, synchronisation et distances de contrôle avec le groupe gabarit.
