#!/usr/bin/env python3
"""Package the verified, selected final result without overwriting any file."""
import argparse,csv,json,hashlib,os,shutil,tempfile,zipfile
from pathlib import Path
from collections import Counter
from package_gabarit_2d import sha,verified_txt_files,publish_exclusive
from selection_evidence import verify_selection_evidence

OUTPUTS={'VIDEO_GABARIT_2D.mp4','OBJETS.csv','POINTS_OBJETS.csv','A_VERIFIER.csv','FRAMES.csv','RAILS.csv','AUDIT_FUSION.jsonl'}

def require(ok,message):
    if not ok:raise ValueError(message)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','selection-report','notice','code','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(argv);r=a.run.resolve()
    m=json.loads((r/'MANIFESTE.json').read_text());v=json.loads((r/'VERIFICATION.json').read_text())
    require(m['status']=='complete' and not m['partial_source'] and m['schema_version']==3,'Verified full selective result required')
    require(v['status']=='passed' and sha(r/'MANIFESTE.json')==v['manifest_sha256'],'Final independent verification required')
    require(m['rail_csv_byte_identical'] and v['original_rail_byte_identical'] and m['source_sha256']==v['source_sha256'],'Source and rail verification differ')
    require(set(m['output_sha256'])==OUTPUTS,'Output file set differs')
    frozen={str((r/n).resolve()):d for n,d in m['output_sha256'].items()}
    frozen[str((r/'MANIFESTE.json').resolve())]=v['manifest_sha256']
    frozen[str((r/'VERIFICATION.json').resolve())]=sha(r/'VERIFICATION.json')
    for file,digest in frozen.items():require(sha(file)==digest,'Changed output '+file)
    txt,txthashes=verified_txt_files(r,m,v)
    for file,name in txt:frozen[str(file.resolve())]=txthashes[name]
    ev,evidence=verify_selection_evidence(a.selection_report,m);frozen.update(evidence)
    bp=Path(m['parameters']['baseline']);bm=json.loads((bp/'MANIFESTE.json').read_text())
    require(sha(bp/'MANIFESTE.json')==m['baseline_manifest_sha256'],'Baseline manifest changed')
    frozen[str((bp/'MANIFESTE.json').resolve())]=m['baseline_manifest_sha256']
    for key,field in [('frames_checked','frames'),('objects_checked','overlap_observations'),('quarantined_objects_checked','quarantined_observations'),('point_rows_checked','point_rows')]:
        require(v[key]==m[field],'Verification count mismatch: '+key)
    classes=Counter();origins=Counter();reviewclasses=Counter();frames=set()
    with (r/'OBJETS.csv').open(newline='') as f:
        for row in csv.DictReader(f):classes[row['class_name']]+=1;frames.add(int(row['frame_index']))
    with (r/'A_VERIFIER.csv').open(newline='') as f:
        for row in csv.DictReader(f):reviewclasses[row['class_name']]+=1
    for file,_ in txt:origins.update(o['inference_origin'] for o in json.loads(file.read_text())['objects'])
    require(set(origins)<={'baseline','tile'},'Unknown accepted prediction origin')
    require(sum(classes.values())==sum(origins.values())==m['overlap_observations'],'Summary counts differ')
    require(sum(reviewclasses.values())==m['quarantined_observations'],'Review count differs')
    require(origins['baseline']+m['quarantined_observations']==bm['overlap_observations'],'A baseline mask was lost or duplicated')
    require(origins['tile']==m['selective_additions'],'Addition count differs')
    stats=dict(video_id=m['video_id'],source=Path(m['source']).name,frames=m['frames'],first_time_s=m['first_time_s'],last_time_s=m['last_time_s'],
        baseline_observations=bm['overlap_observations'],accepted_observations=m['overlap_observations'],quarantined_observations=m['quarantined_observations'],
        selective_additions=m['selective_additions'],frames_with_accepted=len(frames),point_rows=m['point_rows'],accepted_by_class=dict(classes),
        accepted_by_origin=dict(origins),review_by_class=dict(reviewclasses),elapsed_s=m['elapsed_s'],runtime_scope=m['runtime_scope'],rails_unchanged=True,
        counts_are_predictions_not_unique_objects=True,TSV_accuracy_measured=False,new_training_performed=False,
        validation_report=ev,validation_report_sha256=sha(a.selection_report))
    b,e=ev['baseline'],ev['selected'];bo,eo=ev['baseline_operating'],ev['selected_operating'];cfg=ev['selected_config']
    note=f'''RÉSULTAT FINAL V2 — PETITS OBJETS DANS LE COULOIR

Vidéo {m['video_id']} : {Path(m['source']).name}.
{m['frames']} images natives, jusqu'à {m['last_time_s']:.3f} secondes, sans son.
Les rails et l'axe sont identiques à V1. Les coordonnées restent en pixels.

CE QUI EST RETENU
Les masques V1 restent intacts, sauf les hypothèses de train ambiguës mises
à vérifier. Les vues rapprochées servent seulement à ajouter des petits
candidats assez confiants, non tronqués et non redondants avec les silhouettes
déjà présentes. Les poids du modèle n'ont pas été réentraînés dans cette V2.
Seuil des ajouts : {cfg['min_confidence']:.2f} ; aire maximale :
{100*cfg['max_area_fraction']:.2f} % des pixels de l'image.

COMPARAISON DE DÉVELOPPEMENT — 288 PHOTOS TACO, 871 OBJETS
                                  V1              V2 sélective
AP petits masques                {100*b['AP_small']:.2f} %         {100*e['AP_small']:.2f} %
AP globale 50-95                 {100*b['AP50_95']:.2f} %         {100*e['AP50_95']:.2f} %
Précision au seuil utilisé       {100*bo['precision']:.2f} %         {100*eo['precision']:.2f} %
Rappel au seuil utilisé          {100*bo['recall']:.2f} %         {100*eo['recall']:.2f} %
Rappel des petits masques        {100*bo['small_recall']:.2f} %         {100*eo['small_recall']:.2f} %

Les deux méthodes utilisent les mêmes références et la même grille préparée.
AP résume qualité des contours et compromis précision-rappel. Le terme petit
désigne une aire COCO <=32×32 pixels dans la photo préparée, pas une taille
en mètres. Les scores du réseau ne sont pas des pourcentages de précision.

Le premier essai de tuiles non sélectives a été rejeté : il retrouvait plus
de petits objets mais produisait beaucoup trop de fausses alertes. La variante
finale a été choisie parmi une grille déclarée de 12 réglages sur la même
validation. Ces scores sont exploratoires/de développement : ils ne sont pas
un nouveau test indépendant, ni la précision certifiée sur la vidéo TSV.
Le détail des 12 configurations est dans COMPARAISON_VALIDATION.json.

COMPTAGES DU FILM — PAS UN INVENTAIRE D'OBJETS PHYSIQUES
V1 : {bm['overlap_observations']} observations.
V2 : {m['overlap_observations']} observations retenues, dont {m['selective_additions']} ajouts ;
{m['quarantined_observations']} hypothèses de train conservées à vérifier.
{m['point_rows']} lignes de points ; {len(frames)} images avec observations retenues.
Un même objet est répété sur plusieurs images. Une observation reste une prédiction.

POUR L'ÉQUIPE GABARIT
VIDEO_GABARIT_2D.mp4 : voir le résultat.
OBJETS.csv : résumé des objets retenus.
TXT/ ou POINTS_OBJETS.csv : leurs contours, en pixels, avec les temps.
RAILS.csv : les limites et l'axe médian inchangés.
FRAMES.csv : une ligne par image.
A_VERIFIER.csv : hypothèses mises à part, contours conservés dans les TXT.
Relier les fichiers avec video_id + frame_index ; utiliser time_video_s.
Lire la notice pour les contours complets et leurs intersections avec le couloir.

Les masques, intersections, temps et correspondances de fichiers ont été
vérifiés. Cela contrôle leur cohérence, pas la vérité de chaque détection.
Les mètres, erreurs réelles par image et dangers physiques restent inconnus.
Une absence de détection ou une hypothèse mise à part ne prouve pas une voie libre.
'''
    receipt=a.out.with_suffix('.verification.json');newnames=['STATISTIQUES.json','RESULTATS_OBTENUS.txt']
    for path in [a.out,receipt]+[r/n for n in newnames]:
        if path.exists():raise FileExistsError(path)
    files=[(r/n,n) for n in ['VIDEO_GABARIT_2D.mp4','RAILS.csv','FRAMES.csv','OBJETS.csv','POINTS_OBJETS.csv','A_VERIFIER.csv','AUDIT_FUSION.jsonl','MANIFESTE.json','VERIFICATION.json']]+txt
    files +=[(a.notice,'NOTICE_EQUIPE_GABARIT.txt'),(a.code/'lire_contours.py','lire_contours.py'),(a.code/'corridor_geometry.py','corridor_geometry.py'),
             (a.selection_report,'COMPARAISON_VALIDATION.json'),(a.selection_report.parent/'EVALUATION_MANIFEST.json','PROVENANCE_VALIDATION.json'),
             (a.selection_report.parent/'PROTOCOL_FROZEN.json','PROTOCOLE_VALIDATION.json')]
    require(len(files)==len({n for _,n in files}),'Duplicate archive member')
    expected={name:frozen.get(str(file.resolve())) or sha(file) for file,name in files}
    for file,name in files:frozen.setdefault(str(file.resolve()),expected[name])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    require(shutil.disk_usage(a.out.parent).free>sum(f.stat().st_size for f,_ in files)+256*1024**2,'Insufficient disk margin')
    created=[]
    try:
        with tempfile.TemporaryDirectory(prefix='.gabarit2d_selective_',dir=a.out.parent) as stage:
            stage=Path(stage);(stage/'STATISTIQUES.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n');(stage/'RESULTATS_OBTENUS.txt').write_text(note)
            files.extend((stage/n,n) for n in newnames);expected.update({n:sha(stage/n) for n in newnames})
            archive=stage/'package.zip';prefix='GABARIT_2D_V2/'
            with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for file,name in files:z.write(file,prefix+name,compress_type=zipfile.ZIP_STORED if file.suffix=='.mp4' else zipfile.ZIP_DEFLATED)
            with zipfile.ZipFile(archive) as z:
                require(len(z.namelist())==len(expected) and set(z.namelist())=={prefix+n for n in expected},'Archive members differ')
                for name,digest in expected.items():
                    h=hashlib.sha256()
                    with z.open(prefix+name) as f:
                        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
                    require(h.hexdigest()==digest,'Archive member changed: '+name)
            for path,digest in frozen.items():require(sha(path)==digest,'Input changed during packaging: '+path)
            proof=dict(status='passed',zip=str(a.out.resolve()),sha256=sha(archive),entries=len(files),bytes=archive.stat().st_size,
                txt_files_checked=len(txt),all_entries_sha256_verified=True,weights_included=False,raw_training_images_included=False,
                validation_split='val',validation_images=288,validation_selected_config=cfg,validation_eligible=True)
            (stage/'receipt.json').write_text(json.dumps(proof,indent=2)+'\n')
            for name in newnames:publish_exclusive(stage/name,r/name,created)
            publish_exclusive(stage/'receipt.json',receipt,created)
            os.link(archive,a.out);created.append(a.out)
    except BaseException:
        for file in reversed(created):file.unlink(missing_ok=True)
        raise
    print(json.dumps(proof),flush=True)

if __name__=='__main__':main()
