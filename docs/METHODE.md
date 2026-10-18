# Du pixel au couloir, puis à l’axe

## La cible

Le polygone couvre l’intérieur de la voie parcourue, entre les faces intérieures visibles des deux rails. Il inclut traverses et ballast entre ces limites, mais ne mesure pas l’épaisseur du métal. Dans un virage, ajouter des sommets aux changements de direction. Mettre `ignore` là où la limite est trop ambiguë ; ne pas inventer la suite au loin.

Le masque contient arrière-plan 0, couloir 1 et ignore 255. Seuls les dessins relus et sauvegardés avec reviewed sont admis par l’export. Le statut reviewed ne prouve pas une exactitude au pixel près.

## Prétraitement et apprentissage

L’image est redimensionnée avec conservation des proportions, puis complétée par des bandes pour entrer dans 640 × 480. Pour une source 1920 × 1080, le contenu devient 640 × 360 avec 60 px de bande en haut et en bas. Les bandes du masque sont ignorées. Les coordonnées retournent ensuite à la résolution source.

DeepLabV3+ produit des scores par pixel ; ResNet50 extrait les caractéristiques. La loss associe entropie croisée et Dice. Les augmentations modérées modifient légèrement géométrie et apparence ; elles ne créent pas de nouvelles captures indépendantes. Le CPU est le repli et MPS accélère le calcul sur Apple Silicon.

Le parcours étudié : préapprentissage RailSem19 générique, annotations publiques ego-path, puis adaptation TSV. La phase générique a terminé deux passages sur les trois demandés avant interruption ; le checkpoint validé du deuxième a servi. Ego-path a ensuite terminé deux passages. L’adaptation TSV disposait d’un budget de vingt passages et sélectionnait un checkpoint sur douze références de développement.

## Extraction géométrique

À une hauteur y, si le couloir forme un intervalle valide :

```text
x_milieu(y) = [x_gauche(y) + x_droite(y)] / 2
largeur(y) = x_droite(y) - x_gauche(y)
```

Les segments sont interrompus quand une section est absente, ambiguë ou rejetée. Les CSV peuvent conserver un échantillon tous les huit pixels de hauteur. Un axe calculé sur une ligne horizontale n’est pas automatiquement le milieu en 3D d’une section perpendiculaire à la voie.

Les contrôles de score, géométrie, saut et continuité sont des heuristiques. Un score softmax de 0,8 n’est pas une garantie de 80 % de précision. Aux aiguillages, plusieurs paires peuvent être plausibles ; il faut évaluer la branche et permettre l’abstention.

## Vitesse et mètres

La vitesse est jointe à la bonne frame, avec identité de source, chronologie et qualité. Le temps de lecture peut différer du temps d’acquisition, notamment pour une vidéo accélérée. Une donnée absente reste absente. L’avancement est une intégrale de vitesse sur les intervalles valides ; les interruptions séparent les segments.

Avec un écartement supposé G et une largeur en pixels w, G/w donne une approximation locale d’échelle horizontale. Elle ne constitue pas une calibration 3D et ne permet pas de mesurer G lui-même. La perspective, le dévers, les virages et la pose de caméra demandent un modèle géométrique supplémentaire. Les champs de calibration restent désactivés dans la configuration fournie.

## Définir les métriques

- Dice : deux fois l’intersection des masques divisée par la somme de leurs surfaces.
- IoU : intersection divisée par union.
- MAE : moyenne des valeurs absolues de l’erreur.
- RMSE : racine de la moyenne des erreurs au carré, plus sensible aux grands écarts.
- Biais : moyenne signée ; positif signifie un décalage vers la droite dans l’image.
- Couverture : sections de référence avec prédiction comparable divisées par les sections éligibles.
- Moyenne par image : on calcule d’abord un score par image, puis leur moyenne.

Une erreur se calcule contre une référence. La dispersion entre modèles, le score réseau et la stabilité temporelle sont d’autres informations ; elles ne remplacent pas la référence. Les sections voisines sont corrélées et ne justifient pas un intervalle de confiance artificiellement étroit.
