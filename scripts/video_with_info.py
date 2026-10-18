"""Créer une nouvelle vidéo avec les informations indicatives, sans entraînement."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from railtsv.annotation_config import active_task


def model_candidates(root,task):
    root=Path(root)
    if task=='corridor':return list(root.glob('runs/entrainement_couloir_*/modele/best.pt'))
    return [*root.glob('runs/essai_10_images_*/modele/last.pt'),
            *(p for p in root.glob('runs/*/modele/best.pt')
              if not p.parent.parent.name.startswith('entrainement_couloir_'))]


def result_page(folder):
    folder=Path(folder)
    report_path=folder/'video.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    corridor=report.get('task')=='corridor' or report.get('telemetry',{}).get('representation')=='corridor'
    page='''<!doctype html><html lang="fr"><meta charset="utf-8">
<title>Vidéo et mesures indicatives</title>
<style>body{font:18px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.5}video,img{width:100%;height:auto}</style>
<h1>Vidéo avec les informations du modèle</h1>
<p>L'axe vert est le milieu des bords intérieurs prédits. Ses interruptions signalent des portions insuffisamment étayées.
La ligne jaune relie les bords intérieurs prédits sur une section horizontale de l'image.
La valeur est un écartement apparent en pixels, à vérifier. Le score du modèle est non calibré
et ne mesure pas la précision réelle. Les mètres nécessitent une calibration physique.</p>
<video controls muted playsinline preload="metadata" src="rails_annotes.mp4"></video>
<p><a href="rails_annotes.mp4">Ouvrir la vidéo</a> · <a href="mesures_par_seconde.csv">Mesures par seconde</a>
· <a href="axe_par_seconde.csv">Coordonnées de l'axe par seconde</a> · <a href="axe_toutes_images.csv">Tous les points de l’axe CSV</a> · <a href="mesures.csv">Toutes les images et statuts CSV</a>
· <a href="video.json">Rapport de calcul</a></p>
<p>Une mesure rejetée ou une image répétée ne reçoit pas d'écartement. Une mesure affichée
peut encore être fausse : les contrôles de géométrie ne remplacent pas la validation.</p>
<img src="apercu.png" alt="Aperçu du panneau et de la section mesurée"></html>'''
    if corridor:
        page=page.replace('Vidéo avec les informations du modèle','Vidéo du couloir et de son axe médian')
        page=page.replace("L'axe vert est le milieu des bords intérieurs prédits.",
                          "L'axe vert est le milieu des deux limites du couloir prédit.")
        page=page.replace("La ligne jaune relie les bords intérieurs prédits sur une section horizontale de l'image.",
                          "La ligne jaune montre la largeur apparente du couloir à une hauteur de l'image.")
        page=page.replace('Le score du modèle est non calibré',
                          'Les surfaces métalliques et leur épaisseur ne sont pas prédites dans ce mode. Le score du couloir est non calibré')
    if report.get('display_smoothing',{}).get('enabled'):
        page=page.replace("L'axe vert est le milieu des deux limites du couloir prédit.",
            "L'axe vert est le milieu des deux limites après un filtre temporel. Le fond coloré reste la segmentation brute.")
        page=page.replace('<p>Une mesure rejetée',
            '<p><a href="axe_brut_et_lisse_par_seconde.csv">Axe brut et lissé par seconde</a> · <a href="axe_brut_et_lisse_toutes_images.csv">Tous les points bruts et lissés</a> · <a href="mesures_lissees.csv">Mesures brutes et lissées par image</a></p><p>Le lissage peut retarder les mouvements et ne corrige pas les confusions du modèle.</p><p>Une mesure rejetée')
    path=folder/'VOIR_VIDEO_ET_MESURES.html'
    if path.exists():raise FileExistsError(path)
    path.write_text(page,encoding='utf-8')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--max-frames',type=int,default=0,help='0=toute la vidéo')
    parser.add_argument('--preview',action='store_true',help='Aperçu du traitement en cours dans le navigateur')
    parser.add_argument('--smooth-display',action='store_true',help='Lisser l’axe pour l’affichage avec export des mesures brutes et lissées')
    args=parser.parse_args()
    if args.max_frames < 0:parser.error('max-frames doit être >= 0')
    source=args.input
    if source is None:
        videos=sorted(p for p in (ROOT/'data/raw/tsv_videos').iterdir()
                      if p.suffix.lower() in {'.avi','.mp4','.mov','.mkv'})
        if not videos:raise ValueError('Ajouter une vidéo dans data/raw/tsv_videos')
        for i,p in enumerate(videos,1):print(f'{i}. {p.name}')
        choice=input('Numéro de la vidéo (Entrée = 1) : ').strip() or '1'
        if not choice.isdigit() or not 1<=int(choice)<=len(videos):
            raise ValueError('Numéro de vidéo invalide')
        source=videos[int(choice)-1]
    checkpoint=args.checkpoint
    if checkpoint is None:
        task=active_task(ROOT)
        candidates=model_candidates(ROOT,task)
        if not candidates:
            raise ValueError('Le modèle du couloir doit encore être entraîné : vérifier les annotations avec le bouton 04, puis entraîner avec 05.'
                             if task=='corridor' else 'Aucun modèle trouvé ; préciser --checkpoint')
        checkpoint=max(candidates,key=lambda p:p.stat().st_mtime)
    out=ROOT/'runs'/('video_avec_infos_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    print(f'Modèle sauvegardé : {checkpoint}',flush=True)
    print('Création d’une nouvelle vidéo muette. Aucun entraînement.',flush=True)
    command=[sys.executable,str(ROOT/'scripts/infer_video.py'),'--input',str(source.resolve()),
                    '--checkpoint',str(checkpoint.resolve()),'--out',str(out),'--device','auto',
                    '--max-frames',str(args.max_frames)]
    if args.preview:command.append('--preview')
    if args.smooth_display:command.append('--smooth-display')
    subprocess.run(command,cwd=ROOT,check=True)
    page=result_page(out)
    print(f'Résultat : {page}',flush=True)
    if sys.platform=='darwin':subprocess.run(['open',str(page)],check=False)


if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError,subprocess.CalledProcessError) as error:
        print(f'Création interrompue : {error}',file=sys.stderr)
        sys.exit(1)
