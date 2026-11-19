"""Create a simple index of actual local results, with optional browser opening."""
import argparse
import html
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def build_index(root=ROOT):
    root=Path(root);out=root/'03_RESULTATS';out.mkdir(exist_ok=True)
    rows=[]
    for result in sorted(out.glob('ESSAI_RAILSEM_EGOPATH_*'), reverse=True):
        if not (result/'INVENTAIRE.json').is_file():continue
        links=[]
        for filename,label in [('01_COMPARAISON_CAPTURE_VALIDATION.mp4','Ancien / nouveau : vidéo complète'),
                ('02_COMPARAISON_AUTRE_CAPTURE_60S.mp4','Autre capture : 60 secondes'),
                ('03_VIDEO_NOUVEAU_MODELE.mp4','Nouveau modèle avec mesures'),
                ('00_LIRE_LES_RESULTATS.txt','Lire les résultats et les limites')]:
            if (result/filename).is_file():
                target=result.name+'/'+filename
                links.append(f'<a href="{html.escape(target,quote=True)}">{label}</a>')
        rows.append(f'<article><h2>{html.escape(result.name)}</h2><p>{" · ".join(links)}</p></article>')
    for run in sorted((root/'runs').iterdir(),key=lambda p:p.stat().st_mtime,reverse=True):
        if not run.is_dir() or not run.name.startswith(('essai_10_images_','demo_infos_video_','entrainement_tsv_',
                'entrainement_couloir_','essai_couloir_lot_','video_avec_infos_','couloir_preparation_','export_gabarit_','preapprentissage_railsem19_','figures_modele_','video_validation_lissee_')):continue
        links=[]
        for relative,label in [('VOIR_LE_NOUVEL_ESSAI.html','Nouvel essai : résultats et vidéo'),('diagnostic_validation/VOIR_LA_VALIDATION.html','Validation du nouvel essai'),('video/VOIR_VIDEO_ET_MESURES.html','Vidéo et CSV du nouvel essai'),('LIRE_LE_RESULTAT.md','Lire le résultat'),('VOIR_LA_VALIDATION.html','Validation et comparaisons'),('LIRE_LES_FIGURES.md','Figures expliquées'),('VOIR_LES_RESULTATS.html','Galerie'),
              ('VOIR_VIDEO_ET_MESURES.html','Vidéo et mesures'),('video_infos/VOIR_VIDEO_ET_MESURES.html','Extrait avec mesures'),
              ('video_reparee/rails_annotes_sans_son.mp4','Vidéo complète sans son'),('rails_annotes.mp4','Vidéo'),
              ('mesures_par_seconde.csv','Mesures par seconde'),('axe_brut_et_lisse_par_seconde.csv','Axe brut et lissé par seconde'),('DONNEES_POUR_EQUIPE_GABARIT.zip','Données pour l’équipe gabarit'),('metriques_par_image.csv','Scores par image'),('erreurs_axe_sections.csv','Coordonnées et erreurs de l’axe'),('axe_par_seconde.csv','Axe par seconde'),
              ('axe_toutes_images.csv','Tous les points de l’axe'),('mesures.csv','Toutes les images CSV'),('LIRE_AVANT_UTILISATION.md','Lire les limites'),
              ('APPRENTISSAGE.log','Progression RailSem19'),('modele/history.json','Historique d’apprentissage'),('video/rails_annotes.mp4','Vidéo du premier essai'),('video/mesures_par_seconde.csv','Mesures du premier essai par seconde')]:
            if (run/relative).is_file():
                target='../runs/'+run.name+'/'+relative
                links.append(f'<a href="{html.escape(target,quote=True)}">{label}</a>')
        if links:rows.append(f'<article><h2>{html.escape(run.name)}</h2><p>{" · ".join(links)}</p></article>')
    page='''<!-- GENERATED TSV -->
<!doctype html><html lang="fr"><meta charset="utf-8"><title>Résultats TSV</title>
<style>body{font:18px system-ui;max-width:1100px;margin:35px auto;padding:20px;background:#f5f6f8;color:#182633}article{background:white;padding:15px;margin:15px 0;border-radius:10px}h2{font-size:18px;overflow-wrap:anywhere}a{display:inline-block;margin:5px}</style>
<h1>Résultats du projet TSV</h1><p>Choisis une vidéo ou une galerie ci-dessous.
Les scores du modèle ne sont pas une précision physique en mètres.</p>'''
    target=out/'RESULTATS.html'
    if target.exists() and not target.read_text().startswith('<!-- GENERATED TSV -->'):
        raise FileExistsError('Ce fichier de résultats contient un document non généré par le projet')
    target.write_text(page+'\n'.join(rows)+'</html>',encoding='utf-8')
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-open',action='store_true')
    args=parser.parse_args()
    target=build_index()
    print(target)
    if not args.no_open and sys.platform=='darwin':subprocess.run(['open',str(target)],check=False)
