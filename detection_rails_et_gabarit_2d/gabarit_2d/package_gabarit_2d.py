#!/usr/bin/env python3
"""Package only a fully verified result, without copying data or model weights."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verified_txt_files(run, manifest, verification):
    """Freeze exactly the TXT set and contents accepted by the verifier."""
    count=int(manifest['frames']);files=[];hashes={};aggregate=hashlib.sha256()
    present=list((run/'TXT').iterdir())
    expected_names={f'frame_{index:06d}.txt' for index in range(count)}
    if len(present)!=count or {p.name for p in present}!=expected_names or not all(p.is_file() for p in present):
        raise ValueError('TXT set differs from the complete verified frame set')
    if verification.get('txt_files_checked')!=count:raise ValueError('Verified TXT count mismatch')
    for index in range(count):
        rel=f'TXT/frame_{index:06d}.txt';path=run/rel;digest=sha(path)
        aggregate.update(rel.encode('utf-8')+b'\0'+bytes.fromhex(digest))
        files.append((path,rel));hashes[rel]=digest
    if aggregate.hexdigest()!=verification.get('txt_path_and_content_aggregate_sha256'):
        raise ValueError('TXT contents changed after independent verification')
    return files,hashes


def publish_exclusive(source,destination,created):
    """Publish a staged small report without overwriting an existing file."""
    with destination.open('xb') as target:
        created.append(destination)
        with source.open('rb') as handle:shutil.copyfileobj(handle,target)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','training','evaluation','out','notice','code']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--original-coco',type=Path)
    a=p.parse_args(argv);run=a.run.resolve()
    m=json.loads((run/'MANIFESTE.json').read_text());v=json.loads((run/'VERIFICATION.json').read_text())
    train=json.loads((a.training/'TRAINING_MANIFEST.json').read_text());metrics=json.loads((a.evaluation/'METRICS.json').read_text())
    if m['status']!='complete' or m['partial_source'] or v['status']!='passed':raise ValueError('Full independently verified video required')
    if train['status']!='complete' or metrics['status']!='complete':raise ValueError('Actual complete training and evaluation required')
    em=json.loads((a.evaluation/'EVALUATION_MANIFEST.json').read_text())
    if em['status']!='complete' or sha(a.evaluation/'METRICS.json')!=em['metrics_json']['sha256']:
        raise ValueError('Evaluation report was not frozen as complete')
    if not m['parameters'].get('litter_model'):raise ValueError('Trained litter inference required for this package')
    learned=train['checkpoints']['best']['sha256']
    model_used=str(Path(m['parameters']['litter_model']).resolve())
    if m['inputs_sha256'].get(model_used)!=learned or metrics['checkpoint_sha256']!=learned:
        raise ValueError('Video, evaluation and selected training checkpoint must be identical')
    if train['dataset_sha256']!=metrics['dataset_sha256']:raise ValueError('Training and evaluation dataset differ')
    snapshot_path=a.training/'DATASET_SNAPSHOT.json'
    if sha(snapshot_path)!=train['dataset_snapshot']['sha256']:raise ValueError('Training dataset snapshot changed')
    split_info=json.loads(snapshot_path.read_text())['splits']
    if v['manifest_sha256']!=sha(run/'MANIFESTE.json'):raise ValueError('Verification does not match manifest')
    for name,digest in m['output_sha256'].items():
        if sha(run/name)!=digest:raise ValueError('Changed output: '+name)
    for name in ['STATISTIQUES.json','RESULTATS_OBTENUS.txt']:
        if (run/name).exists():raise FileExistsError(run/name)
    if a.out.exists():raise FileExistsError(a.out)
    receipt=a.out.with_suffix('.verification.json')
    if receipt.exists():raise FileExistsError(receipt)
    for path in [a.notice,a.code/'lire_contours.py',a.code/'corridor_geometry.py']:
        if not path.is_file():raise FileNotFoundError(path)
    txt_files,txt_hashes=verified_txt_files(run,m,v)
    classes=Counter();sources=Counter();frames_with_object=0;frames_without_support=0
    partial=border=unconfirmed=0
    with (run/'OBJETS.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            classes[row['class_name']]+=1;sources[row['model_source']]+=1
            partial+=row['support_status']=='partial';border+=row['touches_crop_border']=='True' or row['touches_image_border']=='True';unconfirmed+=row['track_confirmed']!='True'
    with (run/'FRAMES.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
        frames_with_object=sum(int(r['overlap_candidates'])>0 for r in rows)
        frames_without_support=sum(int(r['rail_supported_rows'])==0 for r in rows)
    rectangles_only=raw_public=raw_litter=suppressed=indeterminate=0
    with (run/'AUDIT_FUSION.jsonl').open() as f:
        for line in f:
            record=json.loads(line);suppressed+=len(record['suppressed'])
            for obj in record['merged_predictions']:
                raw_public+=obj['model_source']=='public';raw_litter+=obj['model_source']=='litter'
                rectangles_only+=obj['bbox_overlap_pixel_count']>0 and obj['relation_2d']!='overlap_observed'
                indeterminate+=obj['relation_2d']=='indeterminate'
    stats={'frames':m['frames'],'first_time_s':m['first_time_s'],'last_time_s':m['last_time_s'],
           'overlap_observations':m['overlap_observations'],'point_rows':m['point_rows'],
           'frames_with_overlap':frames_with_object,'frames_without_rail_support':frames_without_support,
           'observations_by_class':dict(classes),'observations_by_model':dict(sources),
           'partially_supported_observations':partial,'border_touching_observations':border,'unconfirmed_observations':unconfirmed,
           'raw_public_instances':raw_public,'remaining_litter_instances':raw_litter,'suppressed_cross_model_duplicates':suppressed,
           'rectangle_overlap_without_mask_overlap_observations':rectangles_only,'indeterminate_instances':indeterminate,
           'runtime_s':m['elapsed_s'],'offline_average_frames_per_second':m['frames']/m['elapsed_s'],
           'counts_are_observations_not_unique_physical_objects':True,
           'video_accuracy_measured':False,'physical_danger_assessed':False,'rails_changed':False,
           'test_metrics':metrics,'original_coco_metrics':json.loads(a.original_coco.read_text()) if a.original_coco else None}
    mask=metrics['mask_metrics'];pct=lambda key:f"{100*mask['metrics/'+key+'(M)']:.1f}"
    original_note=''
    if stats['original_coco_metrics'] is not None:
        cm=stats['original_coco_metrics']
        if cm['status']!='complete' or cm['checkpoint_sha256']!=metrics['checkpoint_sha256'] or cm['split']!=metrics['split']:
            raise ValueError('Original COCO report does not match this evaluation')
        sm=cm['segmentation_metrics']
        original_note=f"\nContrôle contre les polygones COCO originaux : AP50={100*sm['AP_IoU_0.50']:.1f} %, AP50-95={100*sm['AP_IoU_0.50_0.95']:.1f} %.\nProtocole COCO max100 objets/image, prédictions agrandies aux dimensions\nsource ; pas de ponts YOLO ajoutés à la vérité terrain. Les conventions de\ncalcul diffèrent aussi du score natif : l'écart n'est pas dû uniquement\nà la conversion des polygones. Ces scores restent propres à TACO.\n"
    note=f'''RÉSULTATS RÉELS — GABARIT 2D, VIDÉO {m['video_id']}

Source : {Path(m['source']).name}
{m['frames']} images originales, de {m['first_time_s']:.6f} à {m['last_time_s']:.6f} secondes.
Les horodatages natifs sont conservés. Vidéo sans son, {rows[0]['image_width']} × {rows[0]['image_height']} pixels.
Le CSV des rails et de l'axe est identique octet pour octet à l'entrée.

APPRENTISSAGE EFFECTUÉ
{train['completed_epochs']} passages d'apprentissage sur {split_info['train']['image_count']} photos / {split_info['train']['instance_count']} objets TACO.
{split_info['val']['image_count']} photos de validation ; {split_info['test']['image_count']} photos / {split_info['test']['instance_count']} objets réservés au test.
Le meilleur modèle est choisi avec la validation.
Évaluation sur la partition {metrics['split']} : {metrics['image_count']} photos.
mAP50 des masques : {pct('mAP50')} %.
mAP50-95 des masques : {pct('mAP50-95')} %.
Ces pourcentages sont des scores de détection/segmentation sur TACO,
pas le pourcentage de réussite dans la vidéo TSV et pas une erreur par frame.
Les chiffres ci-dessus utilisent les contours YOLO convertis et audités.
Les masques COCO originaux et leur comparaison éventuelle sont séparés.
{original_note}

CE QUI A ÉTÉ EXPORTÉ
{m['overlap_observations']} observations d'objets recouvrant le couloir,
sur {frames_with_object} images. Un même objet peut apparaître dans plusieurs
images : ce n'est pas un compte d'objets physiques distincts.
{m['point_rows']} lignes de points : contours complets et intersections séparées.
Répartition des observations : {json.dumps(dict(classes),ensure_ascii=False)}.
Par modèle : {json.dumps(dict(sources),ensure_ascii=False)}.
{partial} observations avec support de rails partiel ; {border} touchent un bord
de l'image ou du recadrage. Ces indicateurs permettent de filtrer les cas difficiles.

L'utilisation des silhouettes a écarté {rectangles_only} recouvrements qui
existeraient avec leur seul rectangle englobant. Cela démontre une différence
géométrique, pas un nombre de fausses alertes mesurées contre une vérité terrain.
{suppressed} hypothèses redondantes entre modèles ont été supprimées et tracées.
{indeterminate} relations objet/couloir sont restées indéterminées dans l'audit.

VÉRIFICATION
Le vérificateur indépendant a relu les frames source et rendues, les temps,
les CSV, tous les TXT, les points, les trous et les intersections exportés.
Il vérifie la cohérence des fichiers, pas la vérité des prédictions.
Traitement hors ligne : {m['elapsed_s']/60:.1f} minutes,
soit {m['frames']/m['elapsed_s']:.2f} images/s en moyenne pour cette exécution
de traitement (lecture, deux modèles, géométrie, écriture vidéo/CSV/TXT),
hors chargement initial des poids et premier calcul des empreintes.
Cette mesure ne démontre pas un fonctionnement en temps réel.

LIMITES
Les petits déchets, les objets flous/masqués et les catégories non apprises
peuvent manquer ; des éléments normaux peuvent être pris pour des objets.
Une silhouette dans le couloir 2D ne prouve pas un danger physique.
Les pierres, branches et obstacles inhabituels ne sont pas couverts exhaustivement.
Sans annotations d'objets de référence sur cette vidéo, l'erreur réelle par
image est inconnue. Sans calibration, les distances en mètres restent inconnues.
Les scores de confiance ne sont pas des pourcentages de précision mesurée.

À FOURNIR À L'ÉQUIPE
Le ZIP contient la vidéo, les TXT, les CSV, une notice et un lecteur Python.
Commencer par NOTICE_EQUIPE_GABARIT.txt. Pour relier les fichiers,
utiliser video_id + frame_index et time_video_s.
'''
    a.out.parent.mkdir(parents=True,exist_ok=True)
    # Nothing is added to the verified run until the entire archive succeeds.
    # A disk/reader/ZIP failure only removes this private staging directory.
    with tempfile.TemporaryDirectory(prefix='.gabarit_package_',dir=a.out.parent) as temporary:
        stage=Path(temporary)
        (stage/'STATISTIQUES.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (stage/'RESULTATS_OBTENUS.txt').write_text(note,encoding='utf-8')
        files=[(run/n,n) for n in ['VIDEO_GABARIT_2D.mp4','RAILS.csv','FRAMES.csv','OBJETS.csv','POINTS_OBJETS.csv','MANIFESTE.json','VERIFICATION.json']]
        files +=[(stage/n,n) for n in ['RESULTATS_OBTENUS.txt','STATISTIQUES.json']]
        files +=txt_files
        files +=[(a.notice,'NOTICE_EQUIPE_GABARIT.txt'),(a.code/'lire_contours.py','lire_contours.py'),(a.code/'corridor_geometry.py','corridor_geometry.py')]
        size=sum(f.stat().st_size for f,_ in files)
        if shutil.disk_usage(a.out.parent).free<size+max(256*1024**2,len(files)*2048):
            raise OSError('Insufficient conservative free-space margin for package')
        prefix='GABARIT_2D_'+re.sub(r'[^A-Za-z0-9_-]','_',m['video_id'])+'/'
        expected={prefix+name:txt_hashes.get(name) or m['output_sha256'].get(name) or sha(path) for path,name in files}
        staged_zip=stage/'package.zip'
        with zipfile.ZipFile(staged_zip,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path,name in files:
                z.write(path,prefix+name,compress_type=zipfile.ZIP_STORED if path.suffix=='.mp4' else zipfile.ZIP_DEFLATED)
        with zipfile.ZipFile(staged_zip) as z:
            if len(z.namelist())!=len(expected) or set(z.namelist())!=set(expected):raise ValueError('Package member mismatch')
            for name,digest in expected.items():
                h=hashlib.sha256()
                with z.open(name) as f:
                    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
                if h.hexdigest()!=digest:raise ValueError('Package hash mismatch: '+name)
        result={'status':'passed','zip':str(a.out),'sha256':sha(staged_zip),'entries':len(files),'bytes':staged_zip.stat().st_size,
                'all_entries_sha256_verified':True,'txt_files_checked':len(txt_files),
                'txt_path_and_content_aggregate_sha256':v['txt_path_and_content_aggregate_sha256'],
                'raw_training_images_included':False,'weights_included':False}
        staged_receipt=stage/'receipt.json';staged_receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        created=[]
        try:
            for name in ['STATISTIQUES.json','RESULTATS_OBTENUS.txt']:
                publish_exclusive(stage/name,run/name,created)
            # Same-filesystem hard link atomically publishes the finished ZIP;
            # it fails instead of overwriting if another process created it.
            os.link(staged_zip,a.out);created.append(a.out)
            publish_exclusive(staged_receipt,receipt,created)
        except BaseException:
            for path in reversed(created):path.unlink(missing_ok=True)
            raise
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
