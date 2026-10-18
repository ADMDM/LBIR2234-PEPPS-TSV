# Convention commune et contrôle des annotations

But : retrouver le milieu de la voie parcourue en pixels. Un bon dessin suit les mêmes limites sur toutes les images. Le nombre de points et le fait de cocher reviewed ne garantissent pas sa justesse.

## Quoi dessiner

1. Un polygone **couloir** entre les **bords intérieurs** des deux rails de la voie parcourue. Les traverses et le ballast entre eux sont inclus. Le métal des rails et l’extérieur restent hors du couloir.
2. Partir du bas visible, remonter le long du bord intérieur gauche, rejoindre le bord droit à la dernière hauteur où les deux bords sont identifiables, redescendre et fermer en bas. Un changement gauche/droite dans le nom n’est pas nécessaire : il n’y a qu’une classe couloir.
3. Dans un virage, ajouter des sommets là où la direction change. Entre deux sommets Labelme trace une droite : si elle coupe le virage, ajouter un point. Ne pas multiplier les petits points sur une portion droite.
4. Utiliser **ignore** là où la limite de la voie est indécidable : rail caché, portion lointaine trop petite, branchement ambigu. Couvrir la portion potentielle de voie incertaine plutôt qu’inventer un bord. La zone ignore touche ou chevauche légèrement la dernière portion fiable ; elle peut aussi couper le couloir sur une occultation. Les portions visibles avant/après restent utiles.
5. Ne pas mettre tous les objets gênants ou tout l’extérieur en ignore. Un objet situé entre les rails, qui ne cache pas leurs limites, reste dans le couloir. Les autres voies, clôtures, câbles et objets extérieurs clairement identifiables restent du fond : ces exemples apprennent au modèle à ne pas les confondre avec la voie suivie.
6. Aux aiguillages, suivre la voie empruntée si elle est identifiable. Si ce n’est pas déterminable avec l’image et sa provenance, noter le nom du fichier pour discussion et ne pas inventer le trajet. Si aucune portion fiable n’est annotable, mettre l’image de côté pour remplacement contrôlé ; ne pas cocher reviewed pour forcer l’avancement.

Les propositions automatiques sont des brouillons, jamais des annotations de référence. Une proposition très irrégulière peut être plus rapide à redessiner. Supprimer uniquement la forme avec Delete Shapes, pas le fichier image ni Delete File.

## Avant de faire les 50 images

Chacun choisit cinq images de son propre lot : une voie droite, un virage, une vue lointaine difficile, une image avec éléments gênants et une autre source si disponible. Comparer ensemble les dessins. Vérifier surtout : même bord intérieur, bonne voie, arrêt cohérent au loin, ignore justifié. Corriger ces cinq images avant de poursuivre. Les lots restent distincts ; cette revue ne crée pas de doublons d’apprentissage.

## Pour chaque image, avant Save

- Bonne voie et deux limites intérieures suivies ?
- Segments du polygone assez proches des bords, surtout dans le virage ?
- Aucun contour croisé, angle parasite ou morceau extérieur inclus ?
- Portion incertaine couverte par ignore, sans masquer une grande portion pourtant visible ?
- Partie éloignée clairement visible annotée, sans arrêt prématuré dû au brouillon ?

Quand ces points sont vérifiés : cocher **reviewed**, puis **Save**. Si le contour est corrigé ensuite, le vérifier de nouveau avant la sauvegarde finale.

## Relecture en équipe et limites

Faire relire par une autre personne au moins cinq images choisies au hasard par lot, ainsi que les images signalées difficiles (virages serrés, aiguillages, occultations). Faire relire les douze images de validation, car elles servent de référence aux scores. C’est un protocole humain recommandé : les contrôles logiciels ne remplacent pas cette relecture.

Le contrôleur `scripts/after_annotation.py --action check` vérifie fichiers, dimensions, labels et reviewed lorsque le lot local est configuré. Le mode standard attend un lot complet ; l’utilisation partielle est explicite. Le modèle actuel utilise 130 annotations TSV et 12 références de développement. Ces références servent à choisir le modèle, pas à produire un test final indépendant.

150 images constituent un premier essai. Il faut ensuite regarder l’erreur de l’axe en pixels, la couverture (portions où un axe est obtenu), les masques prédits et les vidéos, en particulier les virages. Ni les tests logiciels ni cette liste ne garantissent une précision réelle. Sans calibration, la largeur apparente reste en pixels ; elle n’est pas un écartement physique mesuré en mètres.
