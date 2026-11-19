# Partie rails et axe médian dans le dépôt du groupe

Intégration du 29 octobre 2026, à partir de la version `ecbffae22aac9a2a7e5399e64d9292852ddd632a` du dépôt `ADMDM/LBIR2234-PEPPS-TSV` (bilan scientifique du 18 octobre 2026).

## Où travailler

Depuis la racine de `Data_science_project_LBIRE2234` :

```bash
cd rails_deeplab
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-plots.txt
python -m pip install --no-deps -e .
python -m pytest -q
python -m railtsv.cli --help
```

Sur Windows, activer `.venv\Scripts\Activate.ps1` à la place de la commande `source`. Toutes les commandes du README de cette partie et de ses guides partent du dossier `rails_deeplab`.

Le module est autonome. Les programmes historiques à la racine et les dossiers `YOLO26` et `gabarit` sont conservés. Leur `main.py` n’est pas automatiquement raccordé à cette partie : l’échange actuel avec le gabarit repose sur les CSV décrits dans [INTERFACE_GABARIT.md](INTERFACE_GABARIT.md).

## Ce que le dépôt contient

- `railtsv/` : modèle, préparation des images, géométrie, métriques et temps.
- `scripts/` : entraînement, inférence, annotation et protocoles datés.
- `tests/` : contrôles logiciels sur des données construites pour les tests.
- `docs/`, `results/`, `configs/`, `examples/` : guides, chiffres agrégés, configurations et exemples synthétiques.
- `../.github/workflows/rails-deeplab-tests.yml` : tests GitHub Actions limités à cette partie, exécutés depuis `rails_deeplab`.

## À récupérer séparément

Pour traiter une vraie vidéo, il faut obtenir une vidéo autorisée et le checkpoint retenu auprès du groupe. Placer les données locales dans `data/` et les modèles/sorties dans `runs/`, à l’intérieur de `rails_deeplab`. Ces fichiers restent ignorés par Git. Les tests logiciels et l’aide des commandes peuvent être exécutés sans ces données réelles.

Le paquet d’échange V3 contient les résultats déjà calculés pour `TSV_record` et `03676`, avec les informations de temps et de vitesse. Il se partage séparément : il n’est pas reconstruit simplement en clonant ce dépôt.

## Portée des résultats

Cette copie réutilise le bilan du 18 octobre 2026 et n’annonce aucun nouvel apprentissage. Les 12 références TSV sont des images de développement, l’erreur réelle sans référence reste inconnue et les mètres sont conditionnels à des hypothèses géométriques. Les tests logiciels ne prouvent pas une précision ferroviaire ou physique.
