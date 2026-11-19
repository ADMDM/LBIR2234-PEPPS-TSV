# Résultats et limites de la livraison

PRÉCISION — CE QUI EST MESURÉ, ET SUR QUELLES DONNÉES
Recalcul du 19 novembre 2026, à partir des prédictions et références conservées.

RAILS : 12 images de développement de la capture 03676, 1 229 sections disponibles.
Ces images ont déjà servi à choisir les modèles. Elles ne sont pas un test final
indépendant et ne mesurent pas l'erreur de chacune des frames 03383 ou TSV_record.
Les sections voisines sont corrélées. Ne pas les présenter comme 1 229 essais indépendants.

Dice moyen par image : 97.27 % ; IoU moyen : 94.69 %.
Dice/IoU décrivent le recouvrement des zones prédites et annotées, pas une précision en mètres.

Moyenne des erreurs absolues par image, en pixels de référence 1920 × 1080 :
- rail gauche : 3.74 px ;
- rail droit : 5.43 px ;
- axe médian : 3.82 px.
En regroupant les sections : MAE gauche 3.73 px, droite 5.40 px,
axe 3.79 px ; RMSE de l'axe 4.66 px.
MAE = moyenne de la taille des erreurs ; RMSE pénalise davantage les grosses erreurs.
La limite droite est moins précise sur cet échantillon. Ces résultats portent
sur le modèle ego-path adapté TSV, pas sur une nouvelle annotation des vidéos livrées.

OBJETS : 173 photos publiques TACO de test, 452 objets de référence.
Recalcul indépendant avec pycocotools ; les AP reproduisent exactement les
résultats conservés du précédent moteur d'évaluation. Aucun entraînement nouveau.
Au seuil de score 0,15 et avec une IoU de masque d'au moins 0,50 :
- 230 bonnes correspondances, 242 fausses détections,
  222 objets de référence manqués ;
- précision : 48.73 % (part des prédictions correctes) ;
- rappel : 50.88 % (part des objets de référence retrouvés).
AP50 : 49,53 % ; AP50–95 : 31,77 % ; AP petits objets : 1,65 % sur la grille originale.
L'AP résume plusieurs seuils de score ; ce n'est pas le pourcentage d'images parfaites.
La petite taille est définie en pixels sur cette grille, pas en centimètres.
Ces scores concernent le modèle déchets, pas toutes les catégories du modèle public.
Ils ne sont pas une mesure sur le couloir TSV. La détection de petits objets reste
la limite principale. Les 25 réglages précédemment rejetés ne sont pas activés.

GPS : 169 positions acceptées sur 181 images examinées pour 03383 (93,4 %).
C'est un taux de lecture retenue, pas une exactitude de géolocalisation.
Sur 19 références visuelles : 17 lectures retenues, dont 16 concordantes à la
tolérance numérique ; une différence d'environ 0,31 m dans les coordonnées lues ;
2 références rejetées. Ce n'est pas une mesure de l'erreur du récepteur GPS.
Les 7 608 positions CSV de TSV_record sont importées ; 7 572 vitesses sont présentes
et 36 restent inconnues. Aucun relevé GPS fourni n'a été validé par un relevé terrain.

VIDÉOS ET GABARIT : cohérence contrôlée, précision terrain non mesurée.
Les 570 554 paires de points des rails 03383 correspondent exactement au gabarit 2D.
Ses 4 206 observations retenues peuvent représenter plusieurs fois un même objet
au cours de la vidéo. Deux hypothèses train restent à vérifier ; absence d'observation
ne veut pas dire voie libre. Le GPS est celui de la caméra, pas celui de l'objet.
Le montage conserve les temps natifs ; les CSV/TXT restent en pixels d'origine.

MÈTRES : le parcours cartographique estimé de 03383 vaut environ 1 108 m.
La ligne cartographiée indique 1,067 m d'écartement nominal ; 1,676 m reste le cas
de dimensionnement séparé. Ces deux hypothèses ne sont pas deux mesures indépendantes.
La conversion transversale impose une échelle locale ; elle ne valide ni profondeur,
ni hauteur, ni écartement physique mesuré. L'erreur physique et son incertitude
restent inconnues en l'absence de calibration et de distances de contrôle.
TSV_record est hors du réseau africain installé : parcours et écartement nominal
restent non résolus, même si le fichier GPS et ses vitesses sont exploitables.

POUR AMÉLIORER LA SUITE
Réserver une nouvelle capture annotée pour évaluer les rails et les objets, y
compris aiguillages, virages, petites tailles et occultations. Compter les fausses
alertes et les objets manqués ; mesurer le retard du lissage. Calibrer la caméra
et obtenir des distances de contrôle pour annoncer une erreur physique en mètres.
