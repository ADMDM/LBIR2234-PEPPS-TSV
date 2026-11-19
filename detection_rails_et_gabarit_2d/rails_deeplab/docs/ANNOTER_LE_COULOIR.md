# Annoter le couloir correctement

L’objectif est l’axe de la voie parcourue. Dessiner un polygone **couloir** entre les faces intérieures visibles des deux rails, en incluant traverses et ballast entre elles. Le métal extérieur n’est pas une seconde classe à tracer.

1. Commencer au bas visible, suivre le bord intérieur gauche jusqu’à la dernière portion lisible.
2. Rejoindre le bord intérieur droit à cette hauteur, redescendre et fermer le polygone.
3. Dans un virage, ajouter des sommets aux changements de direction pour que les segments ne coupent pas la courbe.
4. Tracer **ignore** sur la partie ambiguë : rail caché, portion trop lointaine ou aiguillage dont la bonne branche ne peut pas être déterminée. Ne pas inventer la continuation.
5. Vérifier le contour, cocher **reviewed**, puis sauvegarder. Un brouillon automatique doit être relu.

Dans Labelme, utiliser `configs/labelme_corridor.yaml` pour disposer des deux labels et du flag. Avec l’environnement séparé déjà installé :

```bash
.labelme-venv/bin/python -m labelme --config configs/labelme_corridor.yaml
```

L’état actuel du modèle est A=50, B=30, C=50 annotations utilisables, soit 130 ; douze références d’une autre capture servent au développement. Les nouveaux renforts restent en pause jusqu’à fixation d’un protocole. Le nombre d’images ne garantit pas la qualité : privilégier diversité de captures et situations difficiles.

Pour travailler à plusieurs, attribuer des lots distincts, conserver l’identité des images et échanger les JSON via l’espace du groupe. GitHub sert ici au code ; les données réelles restent séparées. Relire ensemble quelques exemples avant un lot complet et demander une seconde lecture des cas ambigus.

Voir [contrôle qualité](CONTROLE_QUALITE_ANNOTATIONS.md) et [collaboration](../CONTRIBUTING.md).
