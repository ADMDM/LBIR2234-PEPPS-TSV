# Travailler à plusieurs

Le propriétaire du dépôt ajoute les membres dans Settings → Collaborators. Aucun nom d’utilisateur n’est ajouté automatiquement.

1. Cloner le dépôt puis créer une branche pour une modification : `git switch -c amelioration-description`.
2. Installer l’environnement et recevoir séparément les données/checkpoints autorisés.
3. Modifier le code ou la documentation, lancer `python -m pytest -q`.
4. Examiner `git status` et `git diff --cached` avant le commit. Ne pas forcer l’ajout des vidéos, annotations, GPS, poids ou ZIP.
5. Pousser la branche et ouvrir une pull request avec le problème, le changement et les vérifications.

Pour les annotations, attribuer des lots sans chevauchement. Renvoyer les JSON et le manifeste d’identité des images par l’espace de fichiers du groupe ; conserver le statut reviewed. Ne pas redessiner ou fusionner un lot au hasard. Les scripts de réception vérifient le lot et préservent l’historique local.

Pour les expériences, conserver configuration, graine, manifeste, empreinte du checkpoint, versions et métriques. Séparer les captures entières. Une frame recadrée reste issue de la même capture. Ne modifier aucun test final après consultation de ses résultats en continuant à l’appeler indépendant.

La branche principale doit rester exécutable. Les modifications d’algorithme doivent être comparées avec les mêmes références, la couverture et les cas d’échec.
