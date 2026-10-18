# Protocoles datés

Les sous-dossiers scripts/*_20261018 conservent les calculs de comparaison, synchronisation, exports V2/V3 et test externe. Ils attendent les noms et artefacts des expériences historiques. Pour une nouvelle vidéo, commencer par la CLI générique ; pour reproduire une expérience, fournir ses manifests et checkpoints et adapter explicitement les chemins de sortie. Les scripts ne doivent pas être lancés tous successivement.

Le script add_overlay_speed.py requiert data/telemetry/overlay_anchors.csv. Son gabarit vide se trouve dans examples/. Les ancrages doivent être relus sur la bonne vidéo ; aucune coordonnée réelle n’est stockée dans ce dépôt.
