# RailSem19 et annotations ego-path

RailSem19 est une source d’images et de cibles ferroviaires. Ses masques génériques ne désignent pas toujours la voie effectivement parcourue. Les annotations ego-path apportent des polylignes de cette voie, converties en masque entre les lignes. Les conventions publiques ne sont pas identiques aux faces intérieures du couloir TSV.

## Pipeline générique

Obtenir les archives officielles et leurs licences depuis le portail WildDash. Ne pas déposer les images dans GitHub. Le script `scripts/pretrain_railsem.py` attend localement `data/raw/railsem19/rs19_val` et `data/raw/railsem19/rs19_splits4000`.

Le mapping actif est rail-track (12) et tram-track (3) vers track_area=1 ; les autres classes connues donnent 0 et les pixels ignorés 255. Les listes de l’article réservent 3 000 images train, 500 validation et 500 test ; 4 500 autres images restent hors de cette première préparation. L’indépendance par vidéo source n’est pas démontrée par cette séparation par images.

```bash
python scripts/pretrain_railsem.py --action prepare
python scripts/pretrain_railsem.py --action train --epochs 3 --device auto
python scripts/pretrain_railsem.py --action status
```

Les pointeurs créés dans configs/ sont locaux et ignorés par Git. Un modèle synthétique, incomplet sans validation exploitable, incompatible ou dont l’empreinte a changé ne doit pas être utilisé comme préapprentissage vérifié.

## Ego-path et adaptation

`prepare_railsem_egopath.py` accepte les chemins `--annotation`, `--images`, `--splits`, `--out`. Lire l’aide et utiliser les annotations officielles autorisées. `run_egopath_experiment.py` et `compare_egopath_experiment.py` décrivent l’expérience historique ; leurs chemins d’entrée exigent les artefacts locaux correspondants.

État observé : phase générique interrompue au troisième passage après deux passages complets ; checkpoint validé du deuxième conservé. Ensuite, deux passages ego-path sur 2 796 images publiques avec 466 de développement, puis adaptation sur 130 annotations TSV avec un budget de vingt passages. La tête est adaptée à background/couloir. Le modèle B comparé est notre DeepLab entraîné avec ego-path, pas les poids officiels TEP-Net.

Les résultats actuels et les dénominateurs se trouvent dans [RESULTATS](RESULTATS.md). La comparaison appariée moyenne par image passe de 4,42 à 3,77 px sur les 12 références et 1 171 sections communes. Le lissage historique n’est pas retenu dans la V3.

Source : [RailSem19](https://www.wilddash.cc/railsem19) et [annotations ego-path](https://github.com/irtrailenium/train-ego-path-detection). Les gains restent à vérifier sur de nouvelles captures TSV indépendantes.
