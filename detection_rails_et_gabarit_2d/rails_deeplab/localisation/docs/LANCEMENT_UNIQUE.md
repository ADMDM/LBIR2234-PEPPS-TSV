# Une commande pour préparer la livraison

Les petits fichiers sont des modules : chacun a un rôle précis. On les conserve
pour pouvoir tester ou corriger une étape sans toucher aux autres.
**Le point d'entrée pour l'utilisateur est `run_pipeline.py` ou `make all`.**
Il n'est pas nécessaire de lancer les modules à la main.

## Installation

Depuis ce dossier, avec Python 3.12 :

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
make network
```

`make network` télécharge le réseau public africain épinglé dans le projet.
L'OCR local requiert macOS et les outils de compilation Apple ; le parcours CSV
n'a pas cette dépendance. L'installation et les vidéos livrées ont été vérifiées
sur Mac Apple Silicon, pas sur toutes les plateformes.

## Utilisation normale

1. Copier `configurations/pipeline.example.json` en `configurations/pipeline.local.json`.
2. Remplacer les chemins et l'identifiant vidéo. Les chemins relatifs partent du
   dossier du fichier de configuration, pas du terminal.
3. Utiliser le profil OCR adapté à l'incrustation ; l'exemple fourni correspond à
   la vidéo TSV de trois minutes. Pour un CSV, partir de `pipeline_csv.example.json`.
4. Lancer :

```bash
make check CONFIG=configurations/pipeline.local.json
make all CONFIG=configurations/pipeline.local.json
```

`check` vérifie seulement la configuration et les chemins. `all` exécute les tests
puis le traitement. Chaque sortie doit être un nouveau dossier : rien n'est écrasé.

Sans Make, la même opération s'exécute avec :

```bash
.venv/bin/python run_pipeline.py --config configurations/pipeline.local.json
```

## Ce que cette commande enchaîne

```text
Vidéo + GPS incrusté OU CSV GPS
            ↓
Horodatages natifs → lecture/import → contrôles → réseau ferroviaire
            ↓
CSV des rails + GPS + avancement + conversions conditionnelles
            ↓
Gabarit 2D existant : vérification de la même vidéo, des temps et des rails
            ↓
Objets + contexte GPS → vidéo avec carte → ZIP vérifié
```

Si le GPS est déjà traité, utiliser `pipeline_reuse.example.json` pour le réutiliser
sans relire la vidéo. Il faut choisir une seule entrée : `gps` ou
`localisation_result`. La commande ne transforme jamais un ancien CSV en résultat
d'une nouvelle vidéo en changeant simplement son nom.

Les sorties des modèles de rails et d'objets sont les **entrées** de cette chaîne.
Le lanceur ne réentraîne pas les modèles et ne masque pas le coût de leur inférence.
Pour une nouvelle vidéo, produire d'abord les prédictions avec `rails_deeplab`
et, si nécessaire, `gabarit_2d`, puis fournir leurs exports. Voir
[l'intégration du gabarit](INTEGRATION_GABARIT_2D.md).

## Résultats et erreurs

`TERMINE.json` est écrit uniquement lorsque les contrôles et le ZIP sont terminés.
`ECHEC.json` signale un échec ; un dossier partiel ne doit pas être livré.
Un contrôle réussi signifie que les données sont raccordées correctement. Il
ne transforme pas les prédictions du modèle en vérité terrain.

Pour recalculer les métriques à partir de références déjà disponibles :

```bash
.venv/bin/python -m pip install -r requirements-evaluation.txt
.venv/bin/python evaluate_precision.py --config configurations/evaluation.local.json --out resultats/nouvelle_evaluation
```

Le fichier de configuration d'évaluation contient les chemins `ocr_observations`,
`ocr_references`, `rail_evaluation` et `coco_evaluation`. On peut omettre une paire
OCR ou un domaine entier si les références ne sont pas disponibles. Les métriques
de chaque domaine restent séparées ; aucune « précision globale » n'est fabriquée.

## Rôle des fichiers

| Fichier | Rôle |
|---|---|
| `Makefile` | Raccourcis `check`, `test`, `run`, `all`, `network` |
| `run_pipeline.py` | Lance les étapes et signale la réussite ou l'échec |
| `run_localisation.py` | Relie lecture GPS, carte et export des rails |
| `video_timeline.py` | Décode les images et conserve leurs temps natifs |
| `gps_ocr.py`, `ocr_vision.swift` | Lisent le texte incrusté sur Mac |
| `gps_inputs.py` | Interprète les coordonnées et synchronise un CSV |
| `rail_localisation.py` | Recherche la ligne, l'écartement cartographié et le parcours |
| `build_handoff.py` | Ajoute les informations GPS aux rails |
| `handoff_bundle.py` | Compare les géométries, joint les objets et vérifie le ZIP |
| `render_connected_video.py` | Affiche la vidéo et la progression cartographique ensemble |
| `evaluate_precision.py` | Recalcule les erreurs lorsqu'une référence existe |
| `tests/` | Vérifie les cas de réussite et les erreurs à refuser |

Les configurations `.local.json`, vidéos, GPS, poids, résultats et carte ne doivent
pas être committés. Le `.gitignore` les exclut. Aucun script ne fait de `git push`.
