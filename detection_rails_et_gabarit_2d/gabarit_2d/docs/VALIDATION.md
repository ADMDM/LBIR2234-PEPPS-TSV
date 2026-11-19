# Protocoles de validation et limites

**Résultat : aucune configuration de tuilage retenue.** Voir les [tableaux complets](RESULTATS_VALIDATION.md). La livraison garde la référence et la mise à vérifier des hypothèses train, sans gain annoncé sur les petits objets.

## Première expérience contrôlée, rejetée

La V2 utilise **le même checkpoint de déchets que la V1**. L'expérience ne mesure donc pas un nouvel entraînement. Elle compare une seule vue globale avec cette vue complétée par des vues rapprochées, puis un dédoublonnage des masques.

Le protocole est fixé avant de lire les nouveaux scores :

- 288 images de **validation TACO**, sans utiliser le test final pour choisir les tuiles ;
- mêmes fichiers, mêmes annotations COCO, même grille préparée et même checkpoint pour les deux méthodes ;
- toutes les composantes des annotations conservées ; aire recalculée dans la grille préparée ;
- première configuration : tuiles de 640 pixels, recouvrement nominal de 25 %, entrée réseau de 960 pixels ; la dernière tuile, décalée au bord de l’image, peut recouvrir davantage la précédente ;
- comparaison AP avec les scores à partir de 0,001 et au plus 100 prédictions par image après dédoublonnage ;
- configuration admissible si AP des petits objets augmente et si l'AP globale ne perd pas plus de 0,01 ;
- essai à 512 pixels uniquement si la configuration 640 n'est pas admissible ;
- seuil de fonctionnement à 0,15 conservé. Il est appliqué aux prédictions brutes avant le dédoublonnage ; il n'est pas choisi après lecture de ces nouveaux résultats.

Le fichier `METRICS.json` de l'exécution fait foi pour les résultats. La validation a des empreintes de fichiers et un cache de prédictions pour rendre le calcul inspectable. Le script est volontairement lié au checkpoint contrôlé ; réutiliser une autre expérience exige un nouveau protocole explicite.

## Deux explorations supplémentaires, également rejetées

Après la perte d’AP globale et l’augmentation des faux positifs, l’essai 512 prévu initialement n’a pas été exécuté. Une nouvelle grille a été déclarée : seuils des ajouts 0,35 / 0,50 / 0,65 / 0,80, aire maximale 0,1 % / 0,2 % / 0,4 % de l’image, avec conservation de tous les masques globaux, exclusion des bords et des recouvrements excessifs. Une troisième expérience ajoute la confirmation du masque par une autre tuile distincte à IoU > 0,5. Elle ne change pas les scores.

Dans les deux grilles, les cinq conditions sont obligatoires : précision au moins égale à la référence moins 3 points, rappel des petits objets strictement supérieur, F2 au moins égal, AP_small strictement supérieure et AP globale au moins égale à la référence moins 1 point. Les règles de départage sont fixées avant les métriques. Les 12 configurations de chacune des deux grilles échouent ; aucun gagnant n’est forcé.

Cette réutilisation de la même validation pour explorer plusieurs idées n’est pas un test indépendant. Les critères ne sont pas relâchés après lecture des résultats. Les masques globaux de score au moins 0,001 sont tous conservés pour AP, y compris les faibles ; le plafond COCO de 100 détections intervient dans la métrique, pas comme une suppression annoncée de la référence livrée.

## Les termes

**Masque** : ensemble des pixels attribués à un objet. **IoU** : surface commune entre masque prédit et masque annoté, divisée par leur surface totale réunie. Une IoU de 1 indique des masques identiques.

**Précision** : proportion des détections qui correspondent à un objet annoté. **Rappel** : proportion des objets annotés retrouvés. Ces deux valeurs dépendent du seuil de score et du critère de correspondance ; ici, le bilan opérationnel utilise un score d'au moins 0,15 et une IoU d'au moins 0,5. Une annotation ne peut être associée qu'à une seule prédiction.

**AP50** : aire interpolée sous la courbe précision-rappel avec une IoU de correspondance d'au moins 0,5. **AP50–95** : moyenne des AP sur plusieurs exigences d'IoU, de 0,5 à 0,95. Elle pénalise davantage les contours imprécis. **AP_small** : AP50–95 sur les petites annotations, définies ici dans la grille préparée par une aire d'au plus 32 × 32 pixels selon la convention COCO.

**F2** : combinaison de précision et rappel qui donne plus de poids au rappel. Ce n'est pas un taux de détection global ; le bilan doit toujours conserver aussi les faux positifs et les objets manqués.

## Ce qu'il ne faut pas conclure

Une hausse d'AP_small sur TACO ne prouve pas automatiquement que les petits obstacles ferroviaires sont mieux détectés. TACO contient des déchets dans d'autres environnements. Le terrain, la taille apparente, les classes d'objets et leur rareté peuvent différer fortement.

Plus de contours sur une vidéo ne suffit pas à conclure à une amélioration : une partie peut être fausse. Les exemples visuels servent à comprendre le comportement ; un véritable score TSV demanderait des annotations d'objets indépendantes et complètes sur des séquences réservées.

La catégorie « petit » dépend de la grille utilisée. Les chiffres de cette comparaison sur la grille préparée ne doivent pas être comparés directement à une ancienne AP_small calculée dans la résolution COCO d'origine.

La V2 ne calcule pas une erreur vraie pour chaque frame vidéo. Sans référence humaine ou mesure indépendante pour cette frame, cette erreur reste inconnue. Le traitement hors ligne d'une vidéo ne démontre pas non plus un fonctionnement en temps réel : il faut mesurer la cadence complète et la latence sur le matériel cible.
