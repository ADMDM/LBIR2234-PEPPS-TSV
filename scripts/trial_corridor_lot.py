"""Explicit trial on one complete reviewed lot, without independent validation."""
import argparse
from datetime import datetime
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from railtsv.annotation_config import active_batch, active_task
from railtsv.common import dump, new_dir, sha256, write_manifest
from railtsv.railsem import tsv_initialization
from scripts.export_labelme import rasterize


def verify_lot(root=ROOT, lot_id='A', expected_count=50):
    root=Path(root).resolve();batch=active_batch(root)
    if active_task(root)!='corridor':raise ValueError('Un lot couloir actif est requis')
    lot=batch/f'lot_{lot_id}';manifest=json.loads((lot/'manifest.json').read_text())
    if manifest['lot_id']!=lot_id or manifest['batch_id']!=batch.name:raise ValueError('Attribution incohérente')
    rows=manifest['images'];stems={r['stem'] for r in rows}
    if len(rows)!=expected_count or len(stems)!=len(rows):raise ValueError('Lot incomplet ou noms dupliqués')
    if {p.stem for p in (lot/'images').glob('*.png')}!=stems:raise ValueError('Images différentes du manifeste')
    if {p.stem for p in (lot/'json').glob('*.json')}!=stems:raise ValueError('Chaque image doit avoir son JSON, sans fichier étranger')
    records=[]
    for row in rows:
        image=lot/'images'/f'{row["stem"]}.png';annotation=lot/'json'/f'{row["stem"]}.json'
        if row['assignee']!=f'lot_{lot_id}' or sha256(image)!=row['image_sha256']:
            raise ValueError('Image canonique modifiée : '+image.name)
        raw=annotation.read_bytes();data=json.loads(raw)
        with Image.open(image) as im:mask=rasterize(data,im.size,task='corridor')
        if not (mask==1).any():raise ValueError('Couloir absent : '+image.name)
        records.append({'image':image,'annotation':annotation,'raw':raw,'data':data,
                        'sequence':row['sequence'],'image_sha256':row['image_sha256'],
                        'json_sha256':sha256(annotation)})
    return records


def prepare_trial(root, output, lot_id='A', expected_count=50):
    records=verify_lot(root,lot_id,expected_count)
    out=new_dir(output);dataset=out/'dataset'
    for name in ['images','json','masks']:(dataset/name).mkdir(parents=True)
    rows=[];sources=[]
    for record in records:
        image=record['image'];annotation=record['annotation']
        shutil.copy2(image,dataset/'images'/image.name)
        (dataset/'json'/annotation.name).write_bytes(record['raw'])
        with Image.open(image) as im:mask=rasterize(record['data'],im.size,task='corridor')
        destination=dataset/'masks'/f'{image.stem}.png';Image.fromarray(mask).save(destination)
        rows.append({'image':str(dataset/'images'/image.name),'mask':str(destination),'sequence':record['sequence']})
        sources.append({k:str(v) if isinstance(v,Path) else v for k,v in record.items()
                        if k not in {'raw','data'}})
    write_manifest(dataset/'train.csv',rows)
    dump(out/'sources.json',sources)
    dump(out/'annotations.json',{'lot':lot_id,'task':'corridor','reviewed_valid':len(records),
                                'training_images':len(records),'validation_images':0,
                                'training_only_demo':True,'originals_modified':False,
                                'note':'Explicit user-authorized trial; normal all-lots gate unchanged.'})
    return out,records


def run_logged(out, args):
    command=[sys.executable,*map(str,args)]
    with (out/'ESSAI.log').open('a',buffering=1) as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,cwd=ROOT)
        for line in process.stdout:
            log.write(line);print(line,end='',flush=True)
        if process.wait():raise subprocess.CalledProcessError(process.returncode,command)


def gallery(out):
    cards=[]
    for folder in sorted((out/'predictions').iterdir()):
        if not folder.is_dir():continue
        with Image.open(folder/'overlay.png') as im:
            im.thumbnail((960,540));im.save(folder/'apercu.jpg')
        rel=f'predictions/{folder.name}/apercu.jpg'
        cards.append(f'<article><h2>{html.escape(folder.name)}</h2><p>Image utilisée pour apprendre : contrôle visuel.</p><a href="predictions/{folder.name}/overlay.png"><img src="{rel}"></a></article>')
    page='''<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Essai couloir — 50 images A</title>
<style>body{font:18px/1.6 system-ui;max-width:1200px;margin:auto;padding:24px;background:#f2f6f7;color:#173047}video,img{width:100%}main{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}article,section{background:white;padding:18px;border-radius:12px}h2{font-size:18px;overflow-wrap:anywhere}a{color:#056f67}@media(max-width:750px){main{grid-template-columns:1fr}}</style>
<h1>Premier essai du couloir avec le lot A</h1><p>50 images reviewed utilisées pour apprendre, avec initialisation RailSem19. Cyan : couloir ; vert : axe détecté. Les portions insuffisamment étayées peuvent être rejetées.</p>
<p><b>Essai sans validation indépendante.</b> La vidéo vient de la capture utilisée pour apprendre. Les aperçus ci-dessous sont des images d’apprentissage. Aucun pourcentage de précision réelle ni mesure en mètres n’est établi.</p>
<section><h2>Extrait vidéo et données</h2><video controls playsinline preload="metadata" src="video/rails_annotes.mp4"></video><p><a href="video/rails_annotes.mp4">Ouvrir le MP4</a> · <a href="video/axe_toutes_images.csv">Axe par image</a> · <a href="video/axe_par_seconde.csv">Axe par seconde</a> · <a href="video/mesures.csv">Mesures par image</a> · <a href="video/mesures_par_seconde.csv">Mesures par seconde</a></p></section><main>'''
    (out/'VOIR_LES_RESULTATS.html').write_text(page+'\n'.join(cards)+'</main></html>',encoding='utf-8')


def main():
    from railtsv.settings import processing_resolution
    default_height, default_width = processing_resolution(ROOT)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lot',choices=['A','B','C'],default='A')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--height',type=int,default=default_height)
    p.add_argument('--width',type=int,default=default_width)
    p.add_argument('--max-video-frames',type=int,default=900,help='900 = environ 30 s ; 0 = vidéo complète')
    p.add_argument('--initialization',choices=['auto','railsem','imagenet'],default='auto')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if args.epochs<1 or args.max_video_frames<0 or min(args.height,args.width)<32 or args.height%32 or args.width%32:p.error('Paramètres invalides')
    records=verify_lot(ROOT,args.lot)
    print(f'Lot {args.lot} : {len(records)}/50 reviewed et masques valides.',flush=True)
    if args.check_only:return
    init_args,init_info=tsv_initialization(ROOT,args.initialization)
    source=ROOT/'data/raw/drive_20261005/video_03383.AVI'
    if not source.is_file():raise FileNotFoundError(source)
    suffix=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    out,records=prepare_trial(ROOT,ROOT/'runs'/f'essai_couloir_lot_{args.lot}_{suffix}',args.lot)
    dump(out/'initialisation.json',init_info)
    state={'status':'running','stage':'training','training_images':50,'validation_images':0,'run':str(out.relative_to(ROOT))}
    dump(out/'essai.json',state)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    print('Essai sans validation indépendante. Sortie : '+str(out),flush=True)
    try:
        run_logged(out,[ROOT/'scripts/train.py','--train',out/'dataset/train.csv','--out',out/'modele',
                       '--task','corridor','--training-only-demo',*init_args,'--device','auto',
                       '--height',args.height,'--width',args.width,'--batch-size','1','--epochs',args.epochs,'--progress-every','25'])
        preview=out/'images_apercu';preview.mkdir()
        selected=[records[int(i)] for i in np.linspace(0,len(records)-1,8)]
        for record in selected:shutil.copy2(record['image'],preview/record['image'].name)
        state.update(stage='predictions');dump(out/'essai.json',state)
        run_logged(out,[ROOT/'scripts/infer.py','--input',preview,'--checkpoint',out/'modele/last.pt','--out',out/'predictions','--device','auto'])
        state.update(stage='video');dump(out/'essai.json',state)
        run_logged(out,[ROOT/'scripts/infer_video.py','--input',source,'--checkpoint',out/'modele/last.pt',
                       '--out',out/'video','--device','auto','--max-frames',args.max_video_frames])
        gallery(out)
        changed=[str(r['annotation']) for r in records if sha256(r['annotation'])!=r['json_sha256']]
        state.update(status='completed',stage='done',source_json_changes_during_trial=changed)
        dump(out/'essai.json',state)
        (out/'LIRE_LE_RESULTAT.md').write_text('# Essai du lot A\n\n50 images reviewed, 10 passages et départ RailSem19. '
            'Aucune validation indépendante : le résultat sert à examiner les prédictions et les erreurs. '
            'Les coordonnées des CSV sont en pixels de l’image originale. Les mètres restent indisponibles.\n')
        run_logged(out,[ROOT/'scripts/open_results.py','--no-open'])
        print('Terminé : '+str(out/'VOIR_LES_RESULTATS.html'),flush=True)
    except BaseException as error:
        state.update(status='failed',error=str(error));dump(out/'essai.json',state);raise


if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError,subprocess.CalledProcessError) as error:
        print('Essai interrompu : '+str(error),file=sys.stderr);sys.exit(1)
