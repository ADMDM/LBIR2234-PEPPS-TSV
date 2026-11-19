# Raccorder rails, gabarit 2D et GPS

Ce dossier peut être placé à côté de `rails_deeplab/` et `gabarit_2d/` dans le
dépôt du groupe. L'intégration ajoute la localisation et la livraison commune ;
elle conserve les modèles et les formats historiques des autres équipes.

## L'entrée du module 2D

Fournir un résultat **complet et vérifié** contenant `MANIFESTE.json`,
`VERIFICATION.json`, `FRAMES.csv`, `RAILS.csv`, `OBJETS.csv`, `A_VERIFIER.csv`,
`POINTS_OBJETS.csv`, les `TXT/` et `VIDEO_GABARIT_2D.mp4`.

Le programme compare l'empreinte de la vidéo source, son identifiant, chaque
numéro de frame, chaque horodatage et chaque paire de points des rails.
Une différence de source, de recadrage, de résolution, de temps ou de géométrie
provoque un arrêt. Les limites ne sont pas rapprochées artificiellement.

La copie locale du dépôt disponible pour cette intégration est le commit
`179a7181eb424aac0a192bada68451826feb0112`. L'accès distant doit être rétabli et
la branche actuelle relue avant publication : cette base locale ne prouve pas
qu'aucun coéquipier n'a modifié le dépôt depuis.

## Contrat commun

- Les rails sont les limites intérieures du couloir parcouru, dans l'image source.
- Les identifiants sont `video_id` et `frame_index`, avec vérification du temps.
- `video_time_s` dans les rails correspond à `time_video_s` dans les objets.
- Le temps provient des PTS natifs. Une frame manquante n'est pas recréée.
- Une cellule vide est inconnue ; elle ne vaut ni zéro ni « pas de risque ».
- L'axe reste disponible dans les anciens exports, mais le nouvel export livre
  explicitement les rails gauche et droit demandés par l'équipe.

`OBJETS_GPS.csv` conserve toutes les anciennes colonnes des objets et ajoute le
contexte GPS du train. **La position GPS appartient à la caméra, pas à l'objet.**
La distance le long de la voie n'est pas la distance entre le train et un obstacle.
Il faut une calibration et une géométrie supplémentaires pour localiser cet obstacle.

Les deux hypothèses « train » mises à vérifier restent dans `A_VERIFIER_GPS.csv`
et `quarantined_predictions` des TXT. Une observation retenue n'est pas un danger
confirmé. Le masque complet et son intersection avec le couloir restent séparés.

## Vidéo commune

Le panneau gauche reprend la vidéo 2D déjà calculée. Le panneau droit montre le
parcours issu de la carte et un point interpolé entre les observations GPS.
La vidéo conserve les 5 308 images de la capture TSV testée et leurs temps natifs.
`CORRESPONDANCE_VIDEO.csv` décrit la réduction et le décalage de l'image affichée.
Les CSV/TXT gardent les coordonnées originales : il ne faut pas lire les pixels
du montage comme si le panneau occupait toute l'image source.

## Ce qui reste à améliorer

1. Évaluer les objets avec des annotations indépendantes sur de vraies vidéos TSV,
   en séparant petits déchets, personnes, véhicules et passages difficiles.
2. Mesurer les faux positifs et les objets manqués, pas seulement regarder la vidéo.
3. Évaluer spécifiquement les branches d'aiguillage et le retard du lissage.
4. Ajouter un réseau couvrant les vidéos hors d'Afrique et le suivi des longs trajets.
5. Calibrer la caméra et obtenir des distances de contrôle avant une précision
   physique ou un gabarit 3D validé. La largeur nominale imposée ne suffit pas.

Les tests de tuiles et filtres précédents n'ont pas établi de gain sur les petits
objets. Cette intégration ne modifie pas ce résultat et n'annonce aucun nouveau
gain de détection ; elle améliore le raccordement et l'exploitation des sorties.
