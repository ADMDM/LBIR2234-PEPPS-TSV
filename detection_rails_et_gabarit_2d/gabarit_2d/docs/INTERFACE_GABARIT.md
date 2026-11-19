# Lire la livraison retenue

## Une frame, un temps, une grille

Une **frame** est une image décodée de la vidéo. `frame_index` commence à zéro. `time_video_s` est le temps de présentation natif de cette image, en secondes ; il ne représente pas forcément une seconde entière. `source_pts` et `time_base_num/time_base_den` gardent sa définition exacte.

Les coordonnées sont en **pixels dans l'image d'origine** : x augmente vers la droite, y vers le bas, origine en haut à gauche. La taille est donnée par `image_width` et `image_height`. Aucun changement arbitraire en VGA n'est appliqué aux coordonnées exportées.

Pour joindre deux fichiers, utiliser `video_id` + `frame_index` et vérifier le temps ainsi que les dimensions. Pour comparer avec une autre équipe, elle doit décoder le même fichier source et utiliser ces mêmes indices et temps. Une vidéo recadrée, accélérée, réencodée ou un échantillonnage par seconde n'est pas interchangeable sans une correspondance explicite.

## Rails et axe

`RAILS.csv` est repris du résultat de rails existant. Sur chaque hauteur y valide, le milieu affiché est la moyenne des limites gauche et droite :

```text
axe_x = (limite_gauche_x + limite_droite_x) / 2
```

Les lignes invalides et les séparations entre segments sont conservées. Les zones non observées ne deviennent pas un couloir confirmé. Dans la livraison complète retenue, le fichier des rails est identique octet pour octet à celui de la référence V1.

## Objets et contours

`OBJETS.csv` contient uniquement les observations retenues dont le masque recoupe la partie observée du couloir. Les observations retenues, identifiants et suivis sont conservés depuis V1. Aucun objet issu des vues rapprochées rejetées n’est ajouté. Un même objet peut produire plusieurs observations successives. `track_id` est un identifiant d'association temporelle, pas une identité certaine ; une observation peut ne pas encore avoir de piste confirmée.

Deux géométries sont conservées séparément :

- `full_predicted_mask_contours` / `geometry_kind=full_predicted_mask` : **masque complet prédit** de l'objet, même si une partie dépasse du couloir ;
- `corridor_intersection_contours` / `geometry_kind=corridor_intersection` : **intersection** entre ce masque et le couloir observé.

Un rectangle de détection n'est jamais présenté comme un contour réel. Les composantes disjointes et les trous sont préservés avec `component_id`, `parent_contour_id`, `is_hole` et `depth`. Utiliser le lecteur fourni pour reconstruire ces géométries, plutôt que de relier tous les points en un seul polygone.

Chaque TXT est un document JSON UTF-8, et non une annotation YOLO à cinq colonnes. Une frame sans objet contient une liste `objects` vide. Dans le format avec vérification, `quarantined_predictions` contient les hypothèses à examiner séparément.

```python
from lire_contours import load_frame, object_masks

frame = load_frame("livraison/TXT/frame_000000.txt")
for objet in frame["objects"]:  # objets retenus seulement
    masque_complet, intersection = object_masks(
        objet, frame["image_width"], frame["image_height"])
```

Pour examiner les hypothèses à part, lire `frame.get("quarantined_predictions", [])`. Les masques reconstruits gardent la taille de l'image source.

## Scores, inconnues et quarantaine

`confidence_score` est le score du réseau. Il n'est pas une erreur mesurée, ni la probabilité calibrée que l'objet soit dangereux. `mask_quality_score`, `actual_error_px`, `distance_m` et `physical_danger_confirmed` restent inconnus si aucune mesure adaptée n'est disponible.

Une hypothèse `train` est mise en quarantaine seulement si les cinq critères du programme sont réunis : classe train, score inférieur à 0,5, couverture d'au moins 90 % du couloir, hauteur d'au moins 75 % de l'image et arrivée au-delà de 95 % de sa hauteur. Cette règle vise une confusion précise entre la voie et un train. **Elle peut aussi concerner un vrai train proche et faiblement détecté** : c'est pourquoi la prédiction reste conservée intégralement dans `A_VERIFIER.csv` et le TXT.

Ne pas assimiler « zéro objet retenu » à « voie libre ». Il faut également regarder les hypothèses à vérifier, les portions de couloir non observées et les limites du modèle.

## Mètres et vitesse

Cette livraison est en pixels. Elle ne fournit pas de distance physique validée. Il faut une calibration de caméra/voie et des distances de contrôle pour convertir proprement les points vers un repère en mètres. Fixer une largeur d'image de 640 pixels ne fournit pas une échelle métrique.

Pour ajouter une vitesse par frame, synchroniser un GPS ou un relevé de vitesse avec le temps vidéo, documenter son unité et son décalage, puis joindre la valeur sur `time_video_s`. Une vitesse calculée ou interpolée doit porter sa méthode et son incertitude ; elle ne doit pas être remplacée par une constante supposée.
