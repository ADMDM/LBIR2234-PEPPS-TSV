#!/usr/bin/env python3
"""Package the verified V1-preserving correction; never claim a small-object gain."""
import argparse,csv,hashlib,json,os,shutil,tempfile,zipfile
from collections import Counter
from pathlib import Path
from guard_only_common import read,require,rejection_evidence,sha
from package_gabarit_2d import verified_txt_files,publish_exclusive

OUTPUTS={'VIDEO_GABARIT_2D.mp4','OBJETS.csv','POINTS_OBJETS.csv','A_VERIFIER.csv','FRAMES.csv','RAILS.csv','AUDIT_FUSION.jsonl'}
METHOD='baseline_with_train_review'
KINDS={'exploratory_selective_tiles_v1':'02_SELECTION',
       'exploratory_multiview_confirmation_v1':'03_CONFIRMATION'}

def validation_files_and_summary(paths,checkpoint):
    require(len(paths)==3,'Exactly three rejection reports required')
    proof,frozen=rejection_evidence(paths,checkpoint)
    files=[];groups={};reports={}
    for raw in paths:
        path=Path(raw).resolve(strict=True);mp=path.parent/'EVALUATION_MANIFEST.json'
        report,manifest=read(path),read(mp)
        if path.name=='METRICS.json':
            group='01_TUILES';require(report['tile_size']==640,'Unexpected initial tile recipe')
        else:
            name=manifest['configuration']['protocol']['name']
            require(name in KINDS,'Unknown rejected selection protocol')
            group=KINDS[name]
        require(group not in groups,'Duplicate rejected experiment kind')
        require(report['reference_objects']==871,'Unexpected validation reference count')
        groups[group]=report;reports[group]=dict(report=path,manifest=mp)
        files.extend([(path,f'VALIDATION/{group}/{path.name}'),
                      (mp,f'VALIDATION/{group}/EVALUATION_MANIFEST.json')])
        protocol=path.parent/'PROTOCOL_FROZEN.json'
        if protocol.exists():
            pr=read(protocol)
            require(pr['configuration']==manifest['configuration']
                    and pr['configuration_sha256']==manifest['configuration_sha256'],
                    'Protocol file differs from completed validation')
            files.append((protocol,f'VALIDATION/{group}/PROTOCOL_FROZEN.json'))
            frozen[str(protocol)]=sha(protocol)
    require(set(groups)=={'01_TUILES','02_SELECTION','03_CONFIRMATION'},'Three distinct rejected experiments required')
    initial=groups['01_TUILES'];summary=[]
    for group in sorted(groups):
        report=groups[group]
        require(report['baseline']==initial['baseline']
                and report['baseline_operating']==initial['baseline_operating']
                and report['dataset_sha256']==initial['dataset_sha256'],
                'Experiments do not use the same frozen baseline/dataset')
        if group=='01_TUILES':
            summary.append(dict(experiment=group,configurations=1,eligible=0,
                baseline=report['baseline'],baseline_operating=report['baseline_operating'],
                attempted=report['tiled'],attempted_operating=report['tiled_operating']))
        else:
            candidates=report['candidates']
            ranges={name:[min(c[section][key] for c in candidates),max(c[section][key] for c in candidates)]
                    for name,section,key in [('AP_small','metrics','AP_small'),('AP50_95','metrics','AP50_95'),
                                            ('precision','operating','precision'),('small_recall','operating','small_recall')]}
            failed=Counter(key for c in candidates for key,value in c['eligibility_checks'].items() if value is False)
            summary.append(dict(experiment=group,configurations=len(candidates),eligible=0,
                                metric_ranges=ranges,failed_gate_counts=dict(failed)))
    return proof,frozen,files,summary

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','notice','code','out'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args(argv);r=a.run.resolve(strict=True)
    receipt=a.out.with_suffix('.verification.json');newnames=['STATISTIQUES.json','RESULTATS_OBTENUS.txt']
    for path in [a.out,receipt]+[r/n for n in newnames]:
        if path.exists():raise FileExistsError(path)
    m,v=read(r/'MANIFESTE.json'),read(r/'VERIFICATION.json')
    require(m['status']=='complete' and m['schema_version']==3 and m['method']==METHOD
            and m['partial_source'] is False,'Verified full guard-only correction required')
    require(all(m[k] is False for k in ('rails_recomputed','neural_inference_performed','new_tracking_performed','small_object_gain_validated'))
            and m['selective_additions']==0,'Correction must preserve baseline without new masks or small-object gain')
    require(v['status']=='passed' and v['manifest_sha256']==sha(r/'MANIFESTE.json'),'Independent verification differs')
    for flag in ['retained_objects_exact_v1_equality','retained_csv_and_points_byte_equality',
                 'historical_ids_and_tracking_preserved','quarantine_guard_independently_recomputed',
                 'original_rail_byte_identical']:
        require(v.get(flag) is True,'Exact V1 verification missing: '+flag)
    require(v.get('small_object_gain_validated') is False and m['rail_csv_byte_identical'] is True
            and v['source_sha256']==m['source_sha256'],'Verification scope/source differs')
    require(set(m['output_sha256'])==OUTPUTS,'Output file set differs')
    frozen=dict(m['inputs_sha256'])
    frozen.update({str((r/name).resolve()):digest for name,digest in m['output_sha256'].items()})
    frozen.update({str(r/'MANIFESTE.json'):v['manifest_sha256'],str(r/'VERIFICATION.json'):sha(r/'VERIFICATION.json')})
    for path,digest in frozen.items():require(sha(path)==digest,'Changed verified output: '+path)
    txt,txthashes=verified_txt_files(r,m,v)
    for path,name in txt:frozen[str(path.resolve())]=txthashes[name]
    bp=Path(m['parameters']['baseline']).resolve(strict=True);bm,bv=read(bp/'MANIFESTE.json'),read(bp/'VERIFICATION.json')
    require(bm['status']=='complete' and bm['schema_version']==1 and bm['partial_source'] is False
            and bv['status']=='passed','Verified V1 baseline required')
    require(sha(bp/'MANIFESTE.json')==bv['manifest_sha256']==m['baseline_manifest_sha256'],'Baseline manifest changed')
    require(m['frames']==bm['frames'] and m['source_sha256']==bm['source_sha256'],'Baseline frame count/source differ')
    baseline_txt,baseline_hashes=verified_txt_files(bp,bm,bv)
    require(bv['txt_path_and_content_aggregate_sha256']==m['baseline_txt_path_and_content_aggregate_sha256'],'Baseline TXT lineage changed')
    frozen.update({str(bp/'MANIFESTE.json'):sha(bp/'MANIFESTE.json'),str(bp/'VERIFICATION.json'):sha(bp/'VERIFICATION.json')})
    for path,name in baseline_txt:frozen[str(path.resolve())]=baseline_hashes[name]
    for name,digest in bm['output_sha256'].items():
        require(sha(bp/name)==digest,'Changed baseline output: '+name);frozen[str(bp/name)]=digest
    checkpoint=bm['inputs_sha256'][str(Path(bm['parameters']['litter_model']).resolve())]
    require(checkpoint==m['checkpoint_sha256'],'Baseline checkpoint differs')
    reject_proof,reject_frozen,evidence_files,experiments=validation_files_and_summary(m['parameters']['rejection_reports'],checkpoint)
    require(reject_proof==m['rejected_experiments'],'Rejection evidence differs from final result')
    frozen.update(reject_frozen)
    for key,field in [('frames_checked','frames'),('objects_checked','overlap_observations'),
                      ('quarantined_objects_checked','quarantined_observations'),('point_rows_checked','point_rows')]:
        require(v[key]==m[field],'Verification count differs: '+key)
    classes,reviewclasses=Counter(),Counter();accepted_ids=set();review_ids=set();frames=set()
    for name,counts,identities in [('OBJETS.csv',classes,accepted_ids),('A_VERIFIER.csv',reviewclasses,review_ids)]:
        with (r/name).open(newline='') as handle:
            for row in csv.DictReader(handle):
                require(row['observation_id'] not in identities,'Repeated observation ID')
                identities.add(row['observation_id']);counts[row['class_name']]+=1
                if name=='OBJETS.csv':frames.add(int(row['frame_index']))
                else:require(row['review_reason']=='ambiguous_train_tracklike','Unexpected quarantine reason')
    require(not accepted_ids&review_ids,'Accepted/review observation ID overlap')
    require(sum(classes.values())==m['overlap_observations'] and sum(reviewclasses.values())==m['quarantined_observations'],'Summary count differs')
    require(m['overlap_observations']+m['quarantined_observations']==bm['overlap_observations'],'Baseline observations lost or added')
    require(set(reviewclasses)<={'train'},'Non-train quarantine is not supported by this correction')
    stats=dict(video_id=m['video_id'],source=Path(m['source']).name,frames=m['frames'],first_time_s=m['first_time_s'],last_time_s=m['last_time_s'],
        method=METHOD,baseline_observations=bm['overlap_observations'],accepted_observations=m['overlap_observations'],
        quarantined_observations=m['quarantined_observations'],selective_additions=0,small_object_gain_validated=False,
        point_rows=m['point_rows'],frames_with_accepted=len(frames),accepted_by_class=dict(classes),review_by_class=dict(reviewclasses),
        rails_unchanged=True,historical_ids_and_tracking_preserved=True,retained_csv_and_points_byte_equality=True,
        counts_are_predictions_not_unique_objects=True,TSV_accuracy_measured=False,new_training_performed=False,
        neural_inference_performed=False,elapsed_s=m['elapsed_s'],runtime_scope=m['runtime_scope'],
        rejected_experiments=experiments,rejected_configurations=sum(e['configurations'] for e in experiments))
    first=experiments[0];b,t=first['baseline'],first['attempted'];bo,to=first['baseline_operating'],first['attempted_operating']
    range_lines=[]
    for label,entry in zip(['Sélection prudente','Confirmation par plusieurs vues'],experiments[1:]):
        rr=entry['metric_ranges']
        range_lines.append(f"{label} : 0/{entry['configurations']} réglages admissibles. AP petits : {100*rr['AP_small'][0]:.2f}–{100*rr['AP_small'][1]:.2f} % ; précision : {100*rr['precision'][0]:.2f}–{100*rr['precision'][1]:.2f} %.")
    note=f'''RÉSULTAT CORRIGÉ — DÉTECTION DANS LE COULOIR

Vidéo {m['video_id']} : {Path(m['source']).name}.
{m['frames']} images natives, jusqu'à {m['last_time_s']:.3f} secondes, sans son.

CE QUI CHANGE
Les {m['quarantined_observations']} hypothèses « train » ambiguës sont mises à vérifier.
Leurs contours restent disponibles dans les TXT et A_VERIFIER.csv.
Tous les autres objets, contours, identifiants et suivis sont exactement ceux de V1.
Les lignes conservées de OBJETS.csv et POINTS_OBJETS.csv sont identiques à V1.
Les rails, l'axe et les temps restent inchangés. Aucun nouveau masque n'est ajouté.

PETITS OBJETS : PAS DE GAIN VALIDÉ
Nous avons évalué 25 réglages dans trois expériences sur les mêmes 288 photos
de validation TACO et 871 objets. Aucun n'a satisfait tous les critères fixés.
Ils ne sont donc pas appliqués au résultat livré. Aucun réentraînement dans cette version.

Premier essai, vues rapprochées sans sélection :
AP petits masques : {100*b['AP_small']:.2f} % → {100*t['AP_small']:.2f} %.
AP globale 50–95 : {100*b['AP50_95']:.2f} % → {100*t['AP50_95']:.2f} %.
Précision : {100*bo['precision']:.2f} % → {100*to['precision']:.2f} %.
Faux positifs : {bo['false_positive']} → {to['false_positive']}.
Le léger gain sur les petits masques s'accompagne de trop de fausses alertes
et d'une dégradation globale : cet essai a été rejeté.

{chr(10).join(range_lines)}
Ces plages décrivent plusieurs réglages, pas les performances d'un modèle unique.
Les rapports complets et les critères sont dans VALIDATION/.
La validation a été réutilisée pour ces essais : ce n'est pas un test indépendant.
Ces scores ne mesurent pas la précision réelle sur la vidéo TSV.
AP résume la qualité des masques et le compromis précision/rappel sur plusieurs
seuils. « Petit » désigne ici une aire COCO <=32×32 pixels dans l'image préparée,
pas une taille en mètres. Le score affiché n'est pas un pourcentage de précision.

COMPTAGES — OBSERVATIONS, PAS OBJETS PHYSIQUES DISTINCTS
V1 : {bm['overlap_observations']} observations = {m['overlap_observations']} conservées
+ {m['quarantined_observations']} mises à vérifier. Ajouts : 0.
{m['point_rows']} lignes de points ; {len(frames)} images avec observations retenues.
Un objet peut apparaître sur plusieurs images.

POUR L'ÉQUIPE GABARIT
VIDEO_GABARIT_2D.mp4 : voir le résultat.
OBJETS.csv : résumé des objets conservés.
TXT/ ou POINTS_OBJETS.csv : contours complets et intersections avec le couloir.
RAILS.csv : limites et axe médian, en pixels.
FRAMES.csv : une ligne par image, même lorsqu'aucun objet n'est retenu.
A_VERIFIER.csv : hypothèses mises à part ; contours dans les TXT.
Relier les fichiers par video_id + frame_index et utiliser time_video_s.
Les identifiants historiques ont été conservés pour faciliter vos comparaisons.

La vérification démontre la cohérence et la conservation des données, pas la
vérité de chaque détection. Distances en mètres, erreurs réelles par image et
danger physique ne sont pas mesurés. Une absence de détection ou une mise à
part ne prouve pas une voie libre. Les petits objets restent une limite à améliorer.
'''
    files=[(r/name,name) for name in sorted(OUTPUTS)]+[(r/'MANIFESTE.json','MANIFESTE.json'),(r/'VERIFICATION.json','VERIFICATION.json')]+txt
    files +=[(a.notice,'NOTICE_EQUIPE_GABARIT.txt'),(a.code/'lire_contours.py','lire_contours.py'),
             (a.code/'corridor_geometry.py','corridor_geometry.py')]+evidence_files
    require(len(files)==len({name for _,name in files}),'Duplicate archive member')
    expected={name:frozen.get(str(path.resolve())) or sha(path) for path,name in files}
    for path,name in files:frozen.setdefault(str(path.resolve()),expected[name])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    require(shutil.disk_usage(a.out.parent).free>sum(p.stat().st_size for p,_ in files)+256*1024**2,'Insufficient disk margin')
    created=[]
    try:
        with tempfile.TemporaryDirectory(prefix='.gabarit2d_corrected_',dir=a.out.parent) as temp:
            stage=Path(temp);(stage/'STATISTIQUES.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n');(stage/'RESULTATS_OBTENUS.txt').write_text(note)
            files.extend((stage/name,name) for name in newnames);expected.update({name:sha(stage/name) for name in newnames})
            archive=stage/'package.zip';prefix='GABARIT_2D_V2_CORRIGEE/'
            with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for path,name in files:z.write(path,prefix+name,compress_type=zipfile.ZIP_STORED if path.suffix=='.mp4' else zipfile.ZIP_DEFLATED)
            with zipfile.ZipFile(archive) as z:
                require(len(z.namelist())==len(expected) and set(z.namelist())=={prefix+n for n in expected},'Archive members differ')
                for name,digest in expected.items():
                    h=hashlib.sha256()
                    with z.open(prefix+name) as handle:
                        for chunk in iter(lambda:handle.read(1024*1024),b''):h.update(chunk)
                    require(h.hexdigest()==digest,'Archive member changed: '+name)
            for path,digest in frozen.items():require(sha(path)==digest,'Input changed during packaging: '+path)
            result=dict(status='passed',zip=str(a.out.resolve()),sha256=sha(archive),entries=len(files),bytes=archive.stat().st_size,
                txt_files_checked=len(txt),all_entries_sha256_verified=True,method=METHOD,small_object_gain_validated=False,
                selective_additions=0,exact_baseline_preservation_verified=True,rejected_experiments=3,rejected_configurations=25,
                weights_included=False,raw_training_images_included=False)
            (stage/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
            for name in newnames:publish_exclusive(stage/name,r/name,created)
            publish_exclusive(stage/'receipt.json',receipt,created)
            os.link(archive,a.out);created.append(a.out)
    except BaseException:
        for path in reversed(created):path.unlink(missing_ok=True)
        raise
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
