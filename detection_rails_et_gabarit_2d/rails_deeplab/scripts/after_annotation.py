"""Check assigned lots, snapshot annotations and train with a held-out capture."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image
import numpy as np
from railtsv.common import check_disjoint,dump,new_dir,sha256,write_manifest
from scripts.export_labelme import rasterize
from scripts.merge_team_annotations import inspect
from railtsv.annotation_config import active_batch,active_task,active_validation,latest_returns


def collect(root=ROOT,lot_b=None,allow_partial_lots=False):
    root=Path(root)
    batch=active_batch(root)
    returned=latest_returns(root,batch)
    if lot_b is not None:returned['B']=lot_b
    lot_b=returned.get('B')
    items,lots=inspect(batch,returned)
    preparation=json.loads((batch/'preparation.json').read_text())
    train_sequence=preparation['sequence']
    train_sequences=set(preparation.get('train_sequences',[train_sequence]))
    records=[dict(image=image,json=path,data=data,sequence=row['sequence'],role='train',lot_id=lot_id)
             for lot_id,row,image,path,data in items]
    if any(r['sequence'] not in train_sequences for r in records):
        raise ValueError('Le nouveau lot ne correspond pas à la capture d’entraînement')
    task=active_task(root)
    invalid={};old_train=0;val_count=0;optional_invalid={}
    if task=='corridor':
        validation=active_validation(root)
        supplemental_value=preparation.get('supplemental_session')
        sessions=[(validation,'val')]
        if supplemental_value:
            supplemental=(root/supplemental_value).resolve()
            if not supplemental.is_relative_to(root.resolve()/'data'):
                raise ValueError('Séance complémentaire requise dans data/')
            sessions.append((supplemental,'train'))
        with (validation/'sequences.csv').open(newline='') as f:
            val_expected={row['stem'] for row in csv.DictReader(f)}
    else:
        old=root/'data/annotation_videos_v1'
        sessions=[(old,None)]
        with (old/'sequences.csv').open(newline='') as f:
            groups={row['stem']:row['sequence'] for row in csv.DictReader(f)}
        val_expected={stem for stem,seq in groups.items() if seq not in train_sequences}
    for session,fixed_role in sessions:
        with (session/'sequences.csv').open(newline='') as f:
            groups={row['stem']:row['sequence'] for row in csv.DictReader(f)}
        if fixed_role=='val' and any(seq in train_sequences for seq in groups.values()):
            raise ValueError('La validation doit provenir d’une autre capture')
        if fixed_role=='train' and any(seq not in train_sequences for seq in groups.values()):
            raise ValueError('Séance complémentaire hors des captures d’entraînement')
        for path in sorted((session/'json').glob('*.json')):
            if path.stem not in groups:raise ValueError('Annotation sans groupe de capture')
            image=session/'images'/f'{path.stem}.png'
            try:
                data=json.loads(path.read_text())
                with Image.open(image) as im:mask=rasterize(data,im.size,task=task)
                if not np.any((mask==1)|(mask==2)):
                    raise ValueError('Aucune portion visible annotée')
            except (ValueError,TypeError,KeyError) as error:
                destination=optional_invalid if task=='corridor' and fixed_role=='train' else invalid
                destination[path.name]=str(error);continue
            role=fixed_role or ('train' if groups[path.stem] in train_sequences else 'val')
            records.append(dict(image=image,json=path,data=data,sequence=groups[path.stem],role=role))
            if role=='train':old_train+=1
            else:val_count+=1
    if len({r['image'].stem for r in records})!=len(records):
        raise ValueError('Une image apparaît plusieurs fois dans les annotations à réunir')
    lots_complete=all(r['reviewed_valid']==r['expected'] for r in lots.values())
    exclusions=[dict(lot=lot_id,stem=stem,reason='Annotation absente')
                for lot_id,status in lots.items() for stem in status['missing']]
    exclusions += [dict(lot=lot_id,stem=stem,reason=reason)
                   for lot_id,status in lots.items() for stem,reason in status['invalid'].items()]
    ready=((lots_complete or allow_partial_lots) and bool(records)
           and any(r['role']=='train' for r in records) and not invalid
           and bool(val_expected) and val_count==len(val_expected))
    report=dict(task=task,optional_invalid_annotations=optional_invalid,lots=lots,returned_lot_b=str(lot_b) if lot_b else None,
                old_training_annotations=old_train,training_annotations=len(items)+old_train,
                validation_annotations=val_count,validation_expected=len(val_expected),
                invalid_old_annotations=invalid,ready=ready,
                lots_complete=lots_complete,partial_lots_authorized=bool(allow_partial_lots),
                assigned_training_expected=sum(r['expected'] for r in lots.values()),
                exclusions=exclusions)
    if ready:
        train=[dict(image=str(r['image']),mask=str(r['json']),sequence=r['sequence']) for r in records if r['role']=='train']
        val=[dict(image=str(r['image']),mask=str(r['json']),sequence=r['sequence']) for r in records if r['role']=='val']
        check_disjoint(train,val)
    return records,report


def print_status(report):
    for name,r in report['lots'].items():print(f'Lot {name} : {r["reviewed_valid"]}/{r["expected"]} annotations vérifiées')
    print(f'Annotations complémentaires vérifiées : {report["old_training_annotations"]}')
    print(f'Total pour apprendre : {report["training_annotations"]}')
    print(f'Validation sur la seconde vidéo : {report["validation_annotations"]}/{report["validation_expected"]}')
    if report.get('partial_lots_authorized') and not report.get('lots_complete'):
        print(f"Essai partiel autorisé : {len(report['exclusions'])} images attribuées exclues ; les annotations non validées restent exclues.")
    print('Prêt pour l’entraînement.' if report['ready'] else 'À compléter avant l’entraînement : tous les lots attribués et les images de validation.')
    for name,why in report['invalid_old_annotations'].items():print(f'À vérifier : {name} — {why}')
    for r in report['lots'].values():
        for name,why in r['invalid'].items():
            if not why.startswith('Annotation non validée'):
                print(f'À vérifier : {name} — {why}')


def snapshot(records,report,output):
    if not report['ready']:raise ValueError('Annotations incomplètes : utiliser le bouton VÉRIFIER pour voir ce qui manque')
    task=report.get('task','tsv')
    out=new_dir(output)
    dataset=out/'dataset'
    for name in ['images','json','masks']:(dataset/name).mkdir(parents=True)
    rows={'train':[],'val':[]};sources=[]
    for record in records:
        source=record['image'];path=record['json'];stem=source.stem
        shutil.copy2(source,dataset/'images'/source.name)
        with Image.open(source) as image:mask=rasterize(record['data'],image.size,task=task)
        if not np.any((mask==1)|(mask==2)):
            raise ValueError(f'Pas de portion visible annotée dans {stem} : vérifier cette image')
        target=dataset/'masks'/f'{stem}.png'
        Image.fromarray(mask).save(target)
        dump(dataset/'json'/path.name,dict(record['data'],imagePath='../images/'+source.name,imageData=None))
        rows[record['role']].append(dict(image=str(dataset/'images'/source.name),mask=str(target),sequence=record['sequence']))
        sources.append(dict(stem=stem,role=record['role'],sequence=record['sequence'],lot_id=record.get('lot_id'),
                            original_image=str(source),image_sha256=sha256(source),
                            original_json=str(path),json_sha256=sha256(path)))
    check_disjoint(rows['train'],rows['val'])
    for role,values in rows.items():write_manifest(dataset/f'{role}.csv',values)
    dump(out/'sources.json',sources);dump(out/'annotations.json',report)
    return out


def main():
    from railtsv.settings import processing_resolution
    default_height, default_width = processing_resolution(ROOT, (576, 1024))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['check','train'],default='check')
    parser.add_argument('--lot-b',type=Path)
    parser.add_argument('--allow-partial-lots',action='store_true',
                        help='Essai explicitement demandé avec les seuls dessins reviewed valides ; validation complète obligatoire')
    parser.add_argument('--no-open',action='store_true',help='Ne pas ouvrir le dossier de résultats automatiquement')
    parser.add_argument('--progress-every',type=int,default=25)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--height',type=int,default=default_height)
    parser.add_argument('--width',type=int,default=default_width)
    parser.add_argument('--initialization', choices=['auto','railsem','egopath','imagenet'], default='auto')
    args=parser.parse_args()
    if args.epochs<1 or min(args.height,args.width)<32 or args.height%32 or args.width%32:
        parser.error('epochs positif ; dimensions multiples de 32')
    records,report=collect(lot_b=args.lot_b,allow_partial_lots=args.allow_partial_lots)
    print_status(report)
    if args.action=='check':return
    if not report['ready']:raise ValueError('Compléter les annotations et leur validation avant ce nouvel entraînement')
    from railtsv.railsem import tsv_initialization
    init_args, init_info = tsv_initialization(ROOT, args.initialization)
    suffix=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    task=report.get('task','tsv')
    prefix='entrainement_couloir_' if task=='corridor' else 'entrainement_tsv_'
    out=snapshot(records,report,ROOT/'runs'/(prefix+suffix))
    dump(out/'initialisation.json', init_info)
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    print('Les polygones sont convertis en masques. Apprentissage du nouveau modèle…',flush=True)
    print('Initialisation : '+init_info['source'], flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/train.py'),'--train',str(out/'dataset/train.csv'),
        '--val',str(out/'dataset/val.csv'),'--out',str(out/'modele'),'--task',task,*init_args,
        '--device','auto','--height',str(args.height),'--width',str(args.width),'--batch-size','1',
        '--epochs',str(args.epochs),'--progress-every',str(args.progress_every)],cwd=ROOT,check=True)
    print('Vérification du modèle sur la seconde capture, aux dimensions originales…',flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/evaluate.py'),'--checkpoint',str(out/'modele/best.pt'),
        '--manifest',str(out/'dataset/val.csv'),'--out',str(out/'evaluation_validation'),'--device','auto'],cwd=ROOT,check=True)
    evaluation=json.loads((out/'evaluation_validation/metrics.json').read_text())
    metrics=evaluation['metrics']
    lines=['# Résultat du nouvel entraînement','',f'Images pour apprendre : {report["training_annotations"]}.',
           f'Images de validation : {report["validation_annotations"]}, issues d’une autre capture.','',
           '| Classe | Dice | IoU |','|---|---:|---:|']
    labels=[('couloir','Couloir')] if task=='corridor' else [('rail_left','Gauche'),('rail_right','Droit')]
    for key,label in labels:
        row=metrics['per_class'][key]
        values=[f'{100*row[k]:.1f} %' if row[k] is not None else 'indisponible' for k in ['dice','iou']]
        lines.append(f'| {label} | {values[0]} | {values[1]} |')
    if task=='corridor':
        axis=evaluation['axis']
        lines+=['','## Axe en pixels','']
        for key,label in [('mean_absolute_center_error_px','Erreur moyenne absolue de l’axe (px)'),
                          ('p95_absolute_center_error_px','95e percentile de l’erreur (px)'),
                          ('coverage_pct','Couverture des portions annotées (%)')]:
            value=axis[key]
            lines.append(f'{label} : {value:.2f}.' if value is not None else f'{label} : indisponible.')
        lines+=['','Le centre est comparé à la même hauteur de l’image. Les portions absentes réduisent la couverture. La qualité des annotations limite cette évaluation.']
    lines+=['','Ces scores comparent les masques aux annotations de validation. Ils ne mesurent pas une erreur en mètres.',
            'La validation sert au choix du modèle ; une troisième capture sera nécessaire pour le test final.','',
            'Utiliser maintenant 06_CREER_LA_VIDEO.command dans 01_ACTIONS pour produire la vidéo avec ce modèle.']
    (out/'LIRE_LE_RESULTAT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Terminé : {out}',flush=True)
    if sys.platform=='darwin' and not args.no_open:subprocess.run(['open',str(out)],check=False)


if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError,subprocess.CalledProcessError) as error:
        print(f'Étape interrompue : {error}',file=sys.stderr);sys.exit(1)
