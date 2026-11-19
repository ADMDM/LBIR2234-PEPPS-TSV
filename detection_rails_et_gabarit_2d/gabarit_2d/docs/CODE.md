# Rôle des programmes

| Programme | Ce qu'il fait |
|---|---|
| `finalize_guard_only.py` | Produit la livraison retenue : V1 préservée et hypothèses train concernées conservées à vérifier, sans nouvelle inférence ni nouveau suivi |
| `verify_guard_only.py` | Vérifie indépendamment les masques, lignes CSV, points, rails, temps et règles de mise à vérifier |
| `package_guard_video.py` | Crée le paquet final corrigé, vérifie toutes les entrées du ZIP et écrit son reçu |
| `selection_evidence.py` | Vérifie les preuves et critères des expériences sélectives sans transformer un rejet en validation |
| `guard_only_common.py` | Préserve le texte exact des lignes CSV et vérifie les preuves des expériences rejetées |
| `selective_refinement.py` | Variante expérimentale rejetée : filtre les ajouts par score, aire, bords et recouvrement en préservant la référence |
| `multiview_confirmation.py` | Variante expérimentale rejetée : exige un accord entre deux fenêtres distinctes à IoU > 0,5 |
| `evaluate_selective_validation.py` | Rejoue les 12 configurations sélectives sur le cache de validation, sans réseau |
| `evaluate_consensus_validation.py` | Rejoue les 12 configurations avec confirmation entre fenêtres, sans réseau |
| `prepare_taco_seg.py` | Télécharge depuis les liens présents dans les annotations officielles, prépare les images et une classe « dechet », vérifie les polygones et sépare les groupes |
| `verify_taco_seg.py` | Contrôle les fichiers préparés et leurs associations |
| `train_litter.py` | Entraîne le modèle de segmentation de déchets, fige les données et journalise les étapes |
| `evaluate_litter.py` | Évalue un checkpoint sur le split explicitement demandé ; exporte les masques et les mesures natives |
| `select_litter_settings.py` | Applique le protocole historique de sélection sur validation ; jamais sur test |
| `evaluate_original_coco.py` | Compare les prédictions aux polygones COCO d'origine, en conservant les composantes disjointes |
| `evaluate_tiled_validation.py` | Compare global seul et global + tuiles sur les mêmes 288 images de validation préparées, avec cache des tuiles |
| `infer_gabarit_2d.py` | Produit la référence V1 à partir d'une vidéo, des rails déjà calculés, d'un segmentateur public et du modèle de déchets |
| `refine_video.py` | Réutilise V1 et produit la vidéo expérimentale avec toutes les tuiles ; cette variante n’est pas retenue |
| `small_object_refinement.py` | Découpe les vues, prédit leurs masques, dédoublonne et applique la règle de quarantaine |
| `corridor_geometry.py` | Transforme les sections de rails valides en zone observée, calcule le recouvrement et encode les contours |
| `rail_io.py` | Lit les lignes CSV de rails, vérifie leur alignement avec les frames et dessine les limites et le milieu |
| `lire_contours.py` | Lit les TXT V1/V2 et reconstruit les masques avec leurs trous et composantes |
| `verify_gabarit_2d.py` | Contrôle indépendamment une livraison V1 complète ou partielle |
| `verify_refinement.py` | Contrôle indépendamment une livraison V2, sa provenance, ses tuiles et sa quarantaine |
| `package_gabarit_2d.py` | Prépare le paquet historique V1 après vérification |
| `package_refinement.py` | Prépare le paquet V2 après vérification et validation du choix de configuration |

## Pourquoi plusieurs vérifications ?

Trois questions sont distinctes :

1. **Le programme calcule-t-il et exporte-t-il correctement ?** Les tests synthétiques et le vérificateur contrôlent cela.
2. **Le modèle reconnaît-il correctement des objets annotés ?** La comparaison de masques sur un jeu de validation répond à cette question dans le domaine de ce jeu.
3. **L'objet est-il réellement dangereux pour le train ?** Le recouvrement 2D ne suffit pas. La profondeur, la hauteur et le gabarit physique nécessitent une autre étape.

Un vérificateur « passed » signifie que les fichiers concordent avec les calculs annoncés. Il ne signifie pas que toutes les prédictions sont justes.

## Comment une frame est traitée dans la livraison retenue

1. Vérifier la référence V1 et les trois rapports qui concluent au rejet des ajouts expérimentaux.
2. Lire la frame native, son temps et les mêmes rails.
3. Garder les objets et leurs identifiants de V1 sans modifier les masques ni refaire leur association temporelle.
4. Placer à vérifier les seules hypothèses train satisfaisant la règle, en conservant leur masque complet.
5. Conserver exactement les lignes CSV et les points des objets retenus ; conserver le CSV des rails à l’identique.
6. Redessiner les images concernées et écrire la nouvelle vidéo, sans inventer de frames.
7. Vérifier indépendamment tous les fichiers avant le partage.

## Comment fonctionne l’expérience de tuiles non retenue

1. Lire l'image originale et son temps natif.
2. Reprendre les mêmes limites de rails et l'axe médian que la V1.
3. Reconstituer les masques complets V1 depuis leurs contours et vérifier leurs empreintes.
4. Définir une zone autour du couloir observé. Les portions sans rails valides restent non observées.
5. Découper cette zone en vues qui se recouvrent. Le réseau voit ainsi les petits objets à une échelle plus grande.
6. Replacer chaque masque dans les coordonnées de l'image originale.
7. Retirer les doublons entre vues par leur recouvrement réel, en gardant une trace de la décision. Le code ne fusionne pas arbitrairement les surfaces des masques.
8. Repérer les hypothèses « train » correspondant à la confusion ciblée et les conserver à part pour examen.
9. Mesurer l'intersection de chaque masque retenu avec le couloir observé.
10. Associer les observations proches dans le temps, sans inventer de contour entre deux détections.
11. Écrire la vidéo, les contours complets, les intersections, les CSV et les traces de provenance.
12. Vérifier indépendamment toute la livraison avant de la mettre dans le paquet de partage.

Les scripts travaillent dans de nouveaux dossiers. Ne pas modifier un ancien résultat pour le faire passer comme une nouvelle exécution ; conserver son manifeste et ses empreintes.
