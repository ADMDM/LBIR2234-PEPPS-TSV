"""Validate frozen original and exploratory VAL selection evidence; no ML imports."""
import hashlib,json,math
from pathlib import Path

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))

def finite(value):
    number=float(value)
    require(math.isfinite(number),'Nonfinite number')
    return number

def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def verify_selection_evidence(path,manifest):
    path=Path(path).resolve(strict=True);empath=path.parent/'EVALUATION_MANIFEST.json'
    s=read(path);em=read(empath)
    require(s['status']==em['status']=='complete' and s['split']==em['split']=='val','Completed VAL selection required')
    require(s['images']==em['image_count']==em['source_image_count']==288 and s['reference_objects']==871,'Wrong selection cardinality')
    require(em['test_images_or_annotations_opened'] is False and em['model_inference_performed'] is False and em['inputs_unchanged'] is True,'Invalid selective replay scope')
    require(em['selection']['sha256']==sha(path)==manifest['selection_report_sha256'],'Selection SHA mismatch')
    require(digest(em['configuration'])==em['configuration_sha256']==s['configuration_sha256'],'Selection config hash mismatch')
    rule=em['configuration']['protocol']
    require(s['frozen_rule']==rule and rule['name']=='exploratory_selective_tiles_v1','Unexpected selection protocol')
    require(rule['confidence_grid']==[.35,.5,.65,.8] and rule['area_fraction_grid']==[.001,.002,.004],'Selection grid changed')
    require(rule['operating_baseline_confidence']==.15 and rule['ap_baseline_confidence']==.001 and rule['network_imgsz']==960 and rule['tile_size']==640 and rule['nominal_overlap']==.25 and rule['test_used'] is False,'Unexpected underlying inference recipe')
    frozen={str(path):sha(path),str(empath):sha(empath)}
    for item in em['inputs']+em['output_files']:
        target=Path(item['path']).resolve(strict=True)
        require(sha(target)==item['sha256'] and target.stat().st_size==item['bytes'],'Changed selection evidence: '+str(target))
        frozen[str(target)]=item['sha256']
    original_record=em['prior_validation_manifest'];original_path=Path(original_record['path']).resolve(strict=True)
    require(sha(original_path)==original_record['sha256'],'Original validation manifest changed')
    original=read(original_path);previous=read(original_path.parent/'METRICS.json')
    require(original['status']==previous['status']=='complete' and original['split']==previous['split']=='val' and original['image_count']==previous['images']==288,'Invalid original VAL run')
    require(original['test_images_or_annotations_opened'] is False and original['inputs_unchanged'] is True,'Invalid original VAL scope')
    require(previous['checkpoint_sha256']==original['checkpoint_sha256']==s['checkpoint_sha256'] and previous['dataset_sha256']==original['dataset_sha256']==s['dataset_sha256'],'Original and selective validation lineage mismatch')
    require(previous['baseline']==s['baseline'] and previous['baseline_operating']==s['baseline_operating'],'Unchanged baseline scores were altered')
    old_eligible=previous['tiled']['AP_small']>previous['baseline']['AP_small'] and previous['tiled']['AP50_95']>=previous['baseline']['AP50_95']-.01
    require(previous['eligible_by_frozen_rule'] is old_eligible and not old_eligible,'Expected explicitly rejected original tile recipe')
    for item in original['inputs']+original['output_files']+[original_record]:
        target=Path(item['path']).resolve(strict=True)
        require(sha(target)==item['sha256'] and target.stat().st_size==item['bytes'],'Changed original VAL evidence: '+str(target))
        frozen[str(target)]=item['sha256']
    require(s['checkpoint_sha256']==em['configuration']['checkpoint_sha256'] and s['dataset_sha256']==em['configuration']['dataset_sha256'],'Selection data/model provenance mismatch')
    require(s['selected_eligible'] is True and len(s['candidates'])==12,'No eligible selective candidate')
    b,bo=s['baseline'],s['baseline_operating'];eligible=[]
    grid=[(c,a) for c in [.35,.5,.65,.8] for a in [.001,.002,.004]]
    for index,record in enumerate(s['candidates']):
        require(record['grid_index']==index and (record['min_confidence'],record['max_area_fraction'])==grid[index],'Candidate grid order/config changed')
        metrics,op=record['metrics'],record['operating']
        for d,keys in [(metrics,['AP_small','AP50_95']),(op,['precision','small_recall','F2'])]:
            require(all(0<=finite(d[k])<=1 for k in keys),'Invalid selection metric')
        checks=dict(precision=op['precision']>=bo['precision']-.03,small_recall=op['small_recall']>bo['small_recall'],F2=op['F2']>=bo['F2'],AP_small=metrics['AP_small']>b['AP_small'],AP50_95=metrics['AP50_95']>=b['AP50_95']-.01)
        require(record['eligibility_checks']==checks and record['eligible'] is all(checks.values()),'Eligibility not supported by metrics')
        if all(checks.values()):eligible.append(record)
    require(bool(eligible),'No eligible candidate after recomputation')
    best=max(eligible,key=lambda c:(c['operating']['small_recall'],c['operating']['precision'],-c['grid_index']))
    cfg={k:best[k] for k in ['min_confidence','max_area_fraction']}
    require(s['selected_config']==cfg==manifest['selection_config'] and s['selected_grid_index']==best['grid_index'],'Wrong selected winner')
    require(s['selected']==best['metrics'] and s['selected_operating']==best['operating'],'Selected metrics differ from winner')
    require(all(manifest['parameters'][k]==value for k,value in cfg.items()),'Video selection thresholds differ')
    return s,frozen
