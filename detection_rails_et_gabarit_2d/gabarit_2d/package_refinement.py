#!/usr/bin/env python3
"""Build an exclusive, hash-verified V2 handoff with accepted and review hypotheses."""
import argparse,csv,json,hashlib,shutil,tempfile,zipfile,os,math
from pathlib import Path
from collections import Counter
from package_gabarit_2d import sha,verified_txt_files,publish_exclusive

CHECKPOINT_SHA='85986c011c6298d0f000b12c89d9d43375269242037734261faa4170285884ba'
VALIDATION_OUTPUTS={'GROUND_TRUTH_PREPARED.json','BASELINE_PREDICTIONS.json','TILED_PREDICTIONS.json',
 'BASELINE_OPERATING_PREDICTIONS.json','TILED_OPERATING_PREDICTIONS.json','IMAGE_AUDIT.json','METRICS.json'}
RUN_OUTPUTS={'VIDEO_GABARIT_2D.mp4','OBJETS.csv','POINTS_OBJETS.csv','A_VERIFIER.csv','FRAMES.csv','RAILS.csv','AUDIT_FUSION.jsonl'}

def require(ok,message):
 if not ok:raise ValueError(message)

def read_json(path):
 return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON: '+value)))

def json_digest(data):
 return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def verified_validation(path,run_manifest):
 """Freeze exact complete VAL evidence and recalculate the declared selection rule.

 This checks provenance and consistency, not a fresh metric calculation from the
 network. It never accesses test images/annotations or runs an ML inference.
 """
 path=Path(path).resolve(strict=True);root=path.parent
 require(path.name=='METRICS.json','Expected the completed evaluation METRICS.json')
 mp=root/'EVALUATION_MANIFEST.json';em=read_json(mp);ev=read_json(path)
 frozen={str(path):sha(path),str(mp):sha(mp)}
 require(em['status']==ev['status']=='complete' and em['inputs_unchanged'] is True,'Complete unchanged validation required')
 require(em['split']==ev['split']=='val' and em['test_images_or_annotations_opened'] is False,'Only VAL evidence is allowed')
 require(ev['images']==em['image_count']==em['source_image_count']==288,'Exactly 288 completed VAL images required')
 model_path=str(Path(run_manifest['parameters']['litter_model']).resolve())
 learned=run_manifest['inputs_sha256'][model_path]
 config=em['configuration'];protocol=config['protocol'];rule=em['selection_rule_frozen_before_metrics']
 require(learned==CHECKPOINT_SHA==ev['checkpoint_sha256']==em['checkpoint_sha256']==config['model_sha256'],'Video and VAL must use the same selected best checkpoint')
 require(ev['dataset_sha256']==em['dataset_sha256']==config['dataset_sha256'],'VAL dataset identity differs')
 require(json_digest(config)==em['configuration_sha256']==ev['configuration_sha256'],'VAL configuration hash mismatch')
 require(ev['selection_rule']==rule,'Selection rule changed after metrics')
 expected={'split':'val','first_tile_size':640,'fallback_tile_size':512,'overlap':.25,'inference_imgsz':960,
  'confidence_floor':.001,'operating_confidence':.15,'max_detections_per_image':100,'test_data_used_for_selection':False}
 require(all(rule.get(k)==value and protocol.get(k)==value for k,value in expected.items()),'Unexpected validation protocol')
 params=run_manifest['parameters'];size=params['tile_size']
 require(size in (640,512) and size==ev['tile_size']==protocol['tile_size'],'Video and VAL tile sizes differ')
 require(params['conf']==protocol['operating_confidence'] and params['imgsz']==protocol['inference_imgsz'] and params['tile_overlap']==protocol['overlap'],'Video and VAL inference settings differ')
 require(ev['baseline_operating']['confidence']==ev['tiled_operating']['confidence']==params['conf'],'Operating score threshold mismatch')
 require(ev['baseline_operating']['mask_iou']==ev['tiled_operating']['mask_iou']==.5,'Operating IoU threshold mismatch')
 def score(value):
  require(not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value) and 0<=value<=1,'Invalid AP selection score');return value
 base,new=ev['baseline'],ev['tiled']
 eligible=score(new['AP_small'])>score(base['AP_small']) and score(new['AP50_95'])>=score(base['AP50_95'])-.01
 require(ev['eligible_by_frozen_rule'] is True and eligible,'Tiled method is ineligible under the frozen rule')
 outputs=em['output_files'];names=[Path(x['path']).name for x in outputs]
 require(len(names)==len(set(names)) and set(names)==VALIDATION_OUTPUTS,'Validation output set differs')
 for record in outputs:
  target=Path(record['path']).resolve(strict=True)
  require(target.parent==root,'Validation output escapes its evaluation directory')
  require(sha(target)==record['sha256'] and target.stat().st_size==record['bytes'],'Changed validation output: '+target.name)
  frozen[str(target)]=record['sha256']
 require(Path(em['metrics']['path']).resolve()==path and em['metrics']['sha256']==frozen[str(path)],'Metrics are not frozen by the completed manifest')
 gt=read_json(root/'GROUND_TRUTH_PREPARED.json')
 require(len(gt['images'])==288 and len(gt['annotations'])==ev['reference_objects'],'Prepared GT image/object count mismatch')
 for record in em['inputs']:
  target=Path(record['path']).resolve(strict=True)
  require(sha(target)==record['sha256'] and target.stat().st_size==record['bytes'],'Changed validation input: '+target.name)
  frozen[str(target)]=record['sha256']
 require(all(config[k] in frozen.values() for k in ('program_sha256','helper_sha256','baseline_predictions_sha256','validation_manifest_sha256','validation_coco_sha256')),'Configuration input hashes lack frozen evidence')
 return ev,frozen

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['run','validation-report','notice','code','out']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args(argv);r=a.run.resolve();m=read_json(r/'MANIFESTE.json');v=read_json(r/'VERIFICATION.json')
 if m['status']!='complete' or m['partial_source'] or m['schema_version']!=2 or v['status']!='passed':raise ValueError('Verified full V2 required')
 if sha(r/'MANIFESTE.json')!=v['manifest_sha256']:raise ValueError('Manifest changed')
 require(m['rail_csv_byte_identical'] is True and v['original_rail_byte_identical'] is True and v['source_sha256']==m['source_sha256'],'Full source/rail verification required')
 require(set(m['output_sha256'])==RUN_OUTPUTS,'V2 output set differs')
 for n,d in m['output_sha256'].items():
  if sha(r/n)!=d:raise ValueError('Changed export: '+n)
 txt,txthashes=verified_txt_files(r,m,v)
 frozen={str((r/n).resolve()):d for n,d in m['output_sha256'].items()}
 frozen.update({str((r/'MANIFESTE.json').resolve()):v['manifest_sha256'],str((r/'VERIFICATION.json').resolve()):sha(r/'VERIFICATION.json')})
 for file,name in txt:frozen[str(file.resolve())]=txthashes[name]
 ev,validation_frozen=verified_validation(a.validation_report,m);frozen.update(validation_frozen)
 b=Path(m['parameters']['baseline']);bm=read_json(b/'MANIFESTE.json')
 if sha(b/'MANIFESTE.json')!=m['baseline_manifest_sha256']:raise ValueError('Changed baseline')
 frozen[str((b/'MANIFESTE.json').resolve())]=m['baseline_manifest_sha256']
 require(v['frames_checked']==m['frames'] and v['objects_checked']==m['overlap_observations'] and v['quarantined_objects_checked']==m['quarantined_observations'] and v['point_rows_checked']==m['point_rows'],'Verification counts differ')
 classes=Counter();origin=Counter();frames=set();reviewclasses=Counter()
 with (r/'OBJETS.csv').open(newline='') as f:
  for row in csv.DictReader(f):classes[row['class_name']]+=1;frames.add(int(row['frame_index']))
 with (r/'A_VERIFIER.csv').open(newline='') as f:
  for row in csv.DictReader(f):reviewclasses[row['class_name']]+=1
 for file,_ in txt:
  payload=json.loads(file.read_text())
  origin.update(o['inference_origin'] for o in payload['objects'])
 require(sum(classes.values())==sum(origin.values())==m['overlap_observations'] and sum(reviewclasses.values())==m['quarantined_observations'],'Package summary counts differ from the verified manifest')
 stats=dict(video_id=m['video_id'],source=Path(m['source']).name,frames=m['frames'],first_time_s=m['first_time_s'],last_time_s=m['last_time_s'],
  baseline_observations=bm['overlap_observations'],accepted_observations=m['overlap_observations'],quarantined_observations=m['quarantined_observations'],frames_with_accepted=len(frames),point_rows=m['point_rows'],
  accepted_by_class=dict(classes),accepted_by_origin=dict(origin),review_by_class=dict(reviewclasses),elapsed_s=m['elapsed_s'],rails_unchanged=m['rail_csv_byte_identical'],
  counts_are_predictions_not_unique_objects=True,TSV_accuracy_measured=False,new_training_performed=False,validation_report=ev,validation_report_sha256=frozen[str(a.validation_report.resolve())],validation_manifest_sha256=frozen[str((a.validation_report.parent/'EVALUATION_MANIFEST.json').resolve())])
 note=f'''RÉSULTAT V2 — PETITS OBJETS DANS LE COULOIR

Vidéo {m['video_id']} : {Path(m['source']).name}.
{m['frames']} images natives, jusqu'à {m['last_time_s']:.3f} secondes, sans son.
Les rails et l'axe restent identiques. Coordonnées dans l'image originale.

CHANGEMENTS
Les masques V1 vérifiés sont réutilisés comme vue globale. Des vues rapprochées
chevauchantes ajoutent une analyse des détails du couloir. Les doublons sont
traités au niveau des silhouettes, sans fusionner leurs pixels aveuglément.
Les hypothèses faibles de train qui recouvrent presque toute la voie sont
mises à vérifier. Elles ne remplissent plus la voie en rouge ; leurs contours
restent dans TXT.quarantined_predictions et A_VERIFIER.csv.
Cette règle peut aussi demander de vérifier un vrai train peu confiant :
elle ne signifie jamais que la voie est libre.

COMPTAGES DU FILM (PAS UNE MESURE DE PRÉCISION)
V1 : {bm['overlap_observations']} observations candidates.
V2 : {m['overlap_observations']} observations retenues, sur {len(frames)} images ;
{m['quarantined_observations']} hypothèses à vérifier séparément.
{m['point_rows']} lignes de points pour les contours retenus et leurs intersections.
Origines des masques retenus : {dict(origin)}.
Catégories prédites : {dict(classes)}.
Les images successives répètent les mêmes objets ; ces comptes ne sont pas
un inventaire d'objets physiques ni un nombre de bonnes détections.

VALIDATION
Le fichier COMPARAISON_VALIDATION.json contient la comparaison contrôlée sur
les photos TACO de validation, mêmes poids et même grille pour les deux méthodes.
Les poids n'ont pas été réentraînés dans cette V2. Aucun score de précision
sur la vidéo TSV n'est inventé. Le test public d'hier n'est pas un nouveau test
de cette méthode. Les vues rapprochées agrandissent l'information disponible,
mais ne recréent pas les détails absents d'une image floue ou trop petite.

UTILISATION
OBJETS.csv et POINTS_OBJETS.csv = observations retenues.
A_VERIFIER.csv = hypothèses mises à part ; ne pas les compter comme absence.
TXT = les deux listes, clairement séparées, avec les contours complets.
RAILS.csv = limites et axe inchangés. FRAMES.csv = temps et compteurs.
Lire NOTICE_EQUIPE_GABARIT.txt et utiliser video_id + frame_index + time_video_s.

CONTRÔLE
Tous les temps, contours, CSV et masques ont été vérifiés indépendamment.
Traitement hors ligne en {m['elapsed_s']/60:.1f} minutes. Cette durée ne démontre
pas une exécution en temps réel. Un score de confiance n'est pas une erreur.
Les distances en mètres, la vérité de chaque détection et le danger physique
restent inconnus sans référence et géométrie adaptées.
'''
 receipt=a.out.with_suffix('.verification.json');newnames=['STATISTIQUES.json','RESULTATS_OBTENUS.txt']
 for q in [a.out,receipt]+[r/n for n in newnames]:
  if q.exists():raise FileExistsError(q)
 for q in [a.notice,a.code/'lire_contours.py',a.code/'corridor_geometry.py',a.validation_report]:
  if not q.is_file():raise FileNotFoundError(q)
 files=[(r/n,n) for n in ['VIDEO_GABARIT_2D.mp4','RAILS.csv','FRAMES.csv','OBJETS.csv','POINTS_OBJETS.csv','A_VERIFIER.csv','MANIFESTE.json','VERIFICATION.json']]+txt
 files +=[(a.notice,'NOTICE_EQUIPE_GABARIT.txt'),(a.code/'lire_contours.py','lire_contours.py'),(a.code/'corridor_geometry.py','corridor_geometry.py'),(a.validation_report,'COMPARAISON_VALIDATION.json'),(a.validation_report.parent/'EVALUATION_MANIFEST.json','PROVENANCE_VALIDATION.json')]
 require(len(files)==len({name for _,name in files}),'Duplicate package member')
 expected={name:frozen.get(str(file.resolve())) or sha(file) for file,name in files}
 for file,name in files:frozen.setdefault(str(file.resolve()),expected[name])
 if any(expected[n]!=d for n,d in txthashes.items()):raise ValueError('TXT changed before packaging')
 a.out.parent.mkdir(parents=True,exist_ok=True)
 if shutil.disk_usage(a.out.parent).free<sum(f.stat().st_size for f,_ in files)+max(256*1024**2,len(files)*2048):raise OSError('Insufficient space margin')
 created=[]
 try:
  with tempfile.TemporaryDirectory(prefix='.gabarit2d_v2_',dir=a.out.parent) as stage:
   stage=Path(stage);(stage/'STATISTIQUES.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n');(stage/'RESULTATS_OBTENUS.txt').write_text(note)
   files +=[(stage/n,n) for n in newnames];expected.update({n:sha(stage/n) for n in newnames})
   archive=stage/'package.zip';prefix='GABARIT_2D_V2/'
   with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for file,name in files:z.write(file,prefix+name,compress_type=zipfile.ZIP_STORED if file.suffix=='.mp4' else zipfile.ZIP_DEFLATED)
   with zipfile.ZipFile(archive) as z:
    if len(z.namelist())!=len(expected) or set(z.namelist())!={prefix+n for n in expected}:raise ValueError('Archive member mismatch')
    for name,digest in expected.items():
     h=hashlib.sha256()
     with z.open(prefix+name) as f:
      for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
     if h.hexdigest()!=digest:raise ValueError('Archive contents changed: '+name)
   for path,digest in frozen.items():
    if sha(path)!=digest:raise ValueError('Input changed during packaging: '+path)
   proof=dict(status='passed',zip=str(a.out.resolve()),sha256=sha(archive),entries=len(files),bytes=archive.stat().st_size,txt_files_checked=len(txt),all_entries_sha256_verified=True,raw_training_images_included=False,weights_included=False,
    validation_split='val',validation_images=288,validation_checkpoint_sha256=ev['checkpoint_sha256'],validation_configuration_sha256=ev['configuration_sha256'],validation_eligible=True)
   (stage/'receipt.json').write_text(json.dumps(proof,indent=2)+'\n')
   for n in newnames:publish_exclusive(stage/n,r/n,created)
   publish_exclusive(stage/'receipt.json',receipt,created)
   os.link(archive,a.out);created.append(a.out)
 except BaseException:
  for f in reversed(created):f.unlink(missing_ok=True)
  raise
 print(json.dumps(proof),flush=True)

if __name__=='__main__':main()
