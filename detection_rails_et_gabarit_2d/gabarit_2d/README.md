# Objets dans le couloir ferroviaire — gabarit 2D

Ce module fournit des **masques d’objets dans le couloir observé**, avec les rails et l’axe médian déjà calculés. Les coordonnées restent dans l’image d’origine et chaque frame conserve son temps natif.

**La livraison retenue conserve la détection de référence et met à vérifier les hypothèses « train » ambiguës. Aucun gain sur les petits objets n’a été validé.** L’ajout de vues rapprochées et deux variantes de filtrage ont été testés ; leurs 25 réglages n’ont pas satisfait les critères annoncés. Les résultats négatifs et le code expérimental sont conservés pour que le groupe puisse les comprendre et les reproduire.

Les poids ne sont pas modifiés dans cette livraison. Les autres masques, identifiants et suivis de référence sont préservés, tout comme le CSV des rails. Les hypothèses « train » visées restent disponibles intégralement pour examen ; elles ne sont pas transformées en preuve d’une voie libre.

## Commencer

- [Lire les CSV et les contours TXT](docs/INTERFACE_GABARIT.md).
- [Installer et produire la livraison retenue](docs/REPRODUIRE.md).
- [Comprendre le rôle de chaque script](docs/CODE.md).
- [Lire les résultats réels et les 25 réglages non retenus](docs/RESULTATS_VALIDATION.md).
- [Comprendre le protocole et les métriques](docs/VALIDATION.md).

Les vidéos, annotations réelles, fichiers GPS, données d’apprentissage et poids sont à récupérer séparément dans le partage autorisé du groupe. Les tests de ce dossier fabriquent uniquement des images et masques synthétiques temporaires.

## Ce que reçoit l’équipe gabarit

| Fichier | Usage |
|---|---|
| `VIDEO_GABARIT_2D.mp4` | Voir les rails, l’axe et les contours retenus |
| `FRAMES.csv` | Retrouver chaque frame, son temps, ses dimensions et ses compteurs |
| `RAILS.csv` | Limites et axe médian, conservés à l’identique |
| `OBJETS.csv` | Une ligne par observation retenue et par frame |
| `POINTS_OBJETS.csv` | Points des contours complets et de leur intersection avec le couloir |
| `TXT/frame_XXXXXX.txt` | Détail JSON d’une frame, y compris absence d’objet et hypothèses à vérifier |
| `A_VERIFIER.csv` | Hypothèses « train » mises à part et conservées dans les TXT |

Les clés communes sont `video_id`, `frame_index` et `time_video_s`. Le score réseau n’est pas une précision mesurée. Les distances et erreurs vraies inconnues restent vides ; elles ne valent pas zéro. Une intersection dans l’image ne confirme pas un danger physique.

## Programmes à utiliser

`finalize_guard_only.py` produit la version retenue depuis une référence V1 complète et vérifiée ; `verify_guard_only.py` contrôle cette nouvelle livraison indépendamment. Le lecteur `lire_contours.py` permet ensuite d’exploiter les fichiers sans modèle ni GPU.

`refine_video.py`, `selective_refinement.py` et `multiview_confirmation.py` correspondent aux variantes expérimentales rejetées. Les scripts d’évaluation expliquent comment leurs résultats ont été obtenus. Le finalisateur sélectif refuse un rapport de sélection sans configuration admissible ; il ne faut pas contourner ce contrôle pour présenter une variante rejetée comme meilleure.

## Reproductibilité

Les sorties sont créées dans de nouveaux dossiers, avec empreintes SHA-256 des entrées et des résultats. Les vérificateurs reconstruisent les masques depuis les contours et contrôlent l’alignement vidéo/CSV/TXT. Un statut « passed » certifie cette cohérence, pas la justesse de toutes les détections.

La copie publiée utilise `rail_io.py` localement et le dossier temporaire du système. L’option `--source` permet de fournir une copie identique de la vidéo en gardant son contrôle SHA ; elle ne relocalise pas automatiquement tous les chemins des anciennes preuves expérimentales. Les adaptations de portabilité sont documentées dans `provenance/PORTABILITY.json`.

Environnement de référence : Python 3.12 sur Mac Apple Silicon, versions dans `requirements.txt`. Tests CPU :

```bash
python -m unittest discover -s . -p 'test_*.py'
```

Ces tests portent sur le programme et les exports. Pour réellement améliorer les petits objets, la prochaine étape est un apprentissage ciblé avec des images ferroviaires annotées complètement et un test réservé par séquence.
