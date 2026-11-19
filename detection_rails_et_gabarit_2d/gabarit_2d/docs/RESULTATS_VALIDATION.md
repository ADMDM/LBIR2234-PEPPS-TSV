# Résultats du 9 novembre 2026 : conservation de la référence

**Aucun des 25 réglages d’ajout de petits objets n’a satisfait tous les critères annoncés.** La livraison retenue conserve les masques de référence et corrige la présentation des hypothèses « train » ambiguës en les plaçant à vérifier. Elle n’annonce aucun gain validé sur les petits objets.

Les 25 réglages correspondent à une variante avec toutes les tuiles, puis deux grilles exploratoires de 12 configurations. Ils utilisent le même checkpoint et les mêmes données de validation ; ce ne sont pas 25 entraînements ou 25 tests indépendants.

## Livraison vidéo effectivement vérifiée

La version retenue correspond au fichier source `03383_Vides of track 1_MOV_0071_20170102ok.AVI` (`video_id=TSV_03383`). Elle contient **5 308 frames**, de 0,033333… s à 180,0 s selon les temps natifs. Ce n’est pas un CSV échantillonné une fois par seconde.

- 4 206 observations d’objets retenues, contre 4 208 dans la référence ; les deux autres hypothèses train sont conservées à vérifier.
- 863 200 lignes de points de contours pour les objets retenus et leurs intersections avec le couloir.
- Rails identiques octet pour octet ; masques retenus, identifiants, suivis et enregistrements correspondants préservés.
- Aucune nouvelle inférence, aucun nouvel entraînement et aucun ajout de petit objet dans cette livraison.

Une observation est une prédiction sur une frame : ces 4 206 observations ne signifient ni 4 206 objets distincts, ni 4 206 vrais obstacles. Le [résumé de livraison](../results/LIVRAISON_CORRIGEE_20261109.json) conserve les comptes et leur portée. La vérification établit la cohérence des exports ; elle ne transforme pas la référence en modèle parfait.

## Données et définitions

Les comparaisons portent sur les mêmes **288 images TACO de validation et 871 objets annotés**, dont 415 petits objets dans la grille préparée. Les masques de référence sont les polygones COCO d’origine ramenés dans cette grille. Aucun fichier d’image ou d’annotation du test final n’est utilisé pour ces choix.

AP_small est ici l’AP50–95 des masques dont l’aire est au plus égale à 32 × 32 pixels dans cette grille préparée. Ces chiffres ne sont pas directement comparables à une AP_small calculée dans la résolution d’origine. Précision, rappel et F2 utilisent le seuil global de 0,15 et des correspondances de masques à IoU ≥ 0,5. Chaque ajout doit aussi respecter son seuil plus strict indiqué dans les tableaux.

Les résultats numériques complets, sans images ni chemins privés, sont dans [VALIDATION_COMPARISONS_20261109.json](../results/VALIDATION_COMPARISONS_20261109.json). Les deux grilles exploratoires ont été explicitement déclarées avant leurs propres calculs. Comme elles suivent le diagnostic des résultats précédents, elles restent de la recherche de réglages sur un jeu de développement.

## Référence et ajout de toutes les vues rapprochées

| Mesure | Vue globale de référence | Global + tuiles de 640 pixels |
|---|---:|---:|
| AP50–95, tous objets | 34,55 % | 32,70 % |
| AP50–95, petits objets | 10,26 % | 10,90 % |
| Précision au seuil 0,15 | 51,92 % | 21,16 % |
| Rappel au seuil 0,15 | 51,21 % | 66,25 % |
| Rappel des petits objets au seuil 0,15 | 24,82 % | 50,12 % |
| Objets associés correctement aux annotations | 446 | 577 |
| Faux positifs selon les annotations TACO | 413 | 2 150 |

Les tuiles retrouvent davantage d’objets, dont davantage de petits objets. Elles créent aussi beaucoup plus de fausses détections. La perte d’AP globale de **1,85 point** dépasse le maximum admis de 1 point : la configuration est rejetée par la règle fixée avant le calcul. La hausse d’AP_small de 0,64 point ne suffit pas à la retenir.

Le repli initial vers des tuiles de 512 pixels n’a pas été exécuté. Le protocole initial a été explicitement abandonné après ce diagnostic pour étudier une fusion plus conservatrice sur le cache déjà calculé, sans nouvelle inférence du réseau.

## Grille exploratoire : ajouts sélectifs

Tous les masques globaux sont conservés. Les nouveaux candidats doivent être petits, dépasser un seuil de score, ne toucher aucun bord et ne pas chevaucher les masques globaux ou un ajout déjà retenu au-delà des limites fixées. Les critères sont cumulés : précision au moins égale à la référence moins 3 points ; rappel des petits objets supérieur ; F2 au moins égal ; AP_small supérieure ; AP globale ne perdant pas plus de 1 point. Les configurations admissibles seraient départagées par le rappel des petits objets, puis la précision.

| Score minimal des ajouts | Aire maximale dans l’image | Ajouts | AP_small | AP50–95 globale | Précision | Rappel petits | F2 | Retenu |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0,35 | 0,1 % | 298 | 7,94 % | 34,13 % | 42,61 % | 36,14 % | 53,11 % | Non |
| 0,35 | 0,2 % | 336 | 7,97 % | 33,99 % | 41,51 % | 36,39 % | 53,00 % | Non |
| 0,35 | 0,4 % | 367 | 7,97 % | 33,94 % | 40,78 % | 36,39 % | 53,08 % | Non |
| 0,50 | 0,1 % | 118 | 8,25 % | 34,28 % | 48,21 % | 30,84 % | 52,79 % | Non |
| 0,50 | 0,2 % | 134 | 8,27 % | 34,18 % | 47,63 % | 31,08 % | 52,83 % | Non |
| 0,50 | 0,4 % | 151 | 8,27 % | 34,14 % | 47,23 % | 31,08 % | 53,07 % | Non |
| 0,65 | 0,1 % | 30 | 8,64 % | 34,29 % | 50,96 % | 26,51 % | 51,80 % | Non |
| 0,65 | 0,2 % | 33 | 8,58 % | 34,17 % | 50,78 % | 26,51 % | 51,76 % | Non |
| 0,65 | 0,4 % | 38 | 8,58 % | 34,07 % | 50,61 % | 26,51 % | 51,81 % | Non |
| 0,80 | 0,1 % | 1 | 10,03 % | 34,48 % | 51,86 % | 24,82 % | 51,34 % | Non |
| 0,80 | 0,2 % | 2 | 10,03 % | 34,41 % | 51,80 % | 24,82 % | 51,32 % | Non |
| 0,80 | 0,4 % | 4 | 10,03 % | 34,38 % | 51,80 % | 24,82 % | 51,41 % | Non |

La variante « score 0,65 ; aire maximale 0,1 % » ajoute 30 prédictions : +7 associations correctes, mais +23 faux positifs. Elle satisfait quatre critères sur cinq, mais AP_small diminue de 10,26 % à 8,64 %. Elle reste **rejetée**. La meilleure AP_small de cette grille atteint seulement 10,03 %, sous la référence.

## Troisième expérience : confirmation entre deux tuiles

La même grille et les mêmes critères sont conservés. Un candidat doit en plus être retrouvé dans au moins deux fenêtres distinctes avec une IoU de masque strictement supérieure à 0,5, démontrée par la trace de fusion. Une simple inclusion IoS ou une correspondance avec la vue globale ne suffit pas. Ces fenêtres viennent de la même image et du même réseau : elles ne constituent pas des observations statistiquement indépendantes.

| Score minimal des ajouts | Aire maximale dans l’image | Ajouts | AP_small | AP50–95 globale | Précision | Rappel petits | F2 | Retenu |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0,35 | 0,1 % | 193 | 8,30 % | 34,21 % | 45,34 % | 32,29 % | 52,58 % | Non |
| 0,35 | 0,2 % | 216 | 8,42 % | 34,15 % | 44,65 % | 32,53 % | 52,64 % | Non |
| 0,35 | 0,4 % | 234 | 8,42 % | 34,17 % | 44,19 % | 32,53 % | 52,76 % | Non |
| 0,50 | 0,1 % | 84 | 8,67 % | 34,39 % | 49,20 % | 29,16 % | 52,41 % | Non |
| 0,50 | 0,2 % | 95 | 8,79 % | 34,33 % | 48,85 % | 29,40 % | 52,50 % | Non |
| 0,50 | 0,4 % | 106 | 8,79 % | 34,36 % | 48,60 % | 29,40 % | 52,71 % | Non |
| 0,65 | 0,1 % | 23 | 9,13 % | 34,42 % | 51,25 % | 26,27 % | 51,76 % | Non |
| 0,65 | 0,2 % | 25 | 9,13 % | 34,32 % | 51,13 % | 26,27 % | 51,74 % | Non |
| 0,65 | 0,4 % | 28 | 9,13 % | 34,32 % | 51,07 % | 26,27 % | 51,82 % | Non |
| 0,80 | 0,1 % | 1 | 10,03 % | 34,48 % | 51,86 % | 24,82 % | 51,34 % | Non |
| 0,80 | 0,2 % | 2 | 10,03 % | 34,41 % | 51,80 % | 24,82 % | 51,32 % | Non |
| 0,80 | 0,4 % | 3 | 10,03 % | 34,44 % | 51,86 % | 24,82 % | 51,43 % | Non |

La confirmation limite une partie des faux positifs, mais aucune configuration ne satisfait tous les critères. Par exemple, « score 0,65 ; aire maximale 0,1 % » ajoute 23 prédictions : +6 associations correctes et +17 faux positifs. Le rappel des petits objets passe à 26,27 %, mais AP_small reste inférieure à la référence (9,13 % contre 10,26 %). Les deux rapports exploratoires gardent `selected_config: null`. Le finalisateur expérimental refuse une livraison présentée comme sélectionnée à partir de ces rapports.

## Ce que l’on conserve et ce qui manque

La correction ciblée « train » est séparée de ces essais : les hypothèses correspondantes sont gardées dans les fichiers de vérification, et ne sont plus dessinées comme des objets acceptés. La référence de détection est conservée pour les autres masques. Cela ne démontre pas que toutes ses détections sont correctes et ne certifie pas une voie libre.

Pour améliorer réellement les petits objets ferroviaires, la prochaine étape est un apprentissage ciblé avec des images TSV annotées complètement : petits objets, images sans obstacle, ballast, traverses, végétation et autres confusions. Il faut réserver des séquences entières pour un test indépendant. Les seuils supplémentaires n’ont pas suffi dans les expériences ci-dessus.

Les scores sont des mesures agrégées de validation TACO. Ils ne fournissent ni l’erreur vraie de chaque frame vidéo, ni une distance en mètres, ni une mesure validée du danger physique.
