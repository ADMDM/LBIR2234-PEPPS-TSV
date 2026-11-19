#!/usr/bin/env python3
"""Compare cached full-image litter masks to additional tiled masks on VAL only.

Original multipart COCO polygons are scaled into the prepared image grid. Both
arms use exactly this GT/grid. Small-object metrics therefore use prepared pixel
areas and must not be compared to the old original-resolution COCO size scores.
Raw tile RLE caches permit resumable inference and CPU-only rescoring. No model
is trained, no test image/annotation is opened, no existing dataset is changed.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
from PIL import Image
from faster_coco_eval import COCO, COCOeval_faster, mask as mask_api
import small_object_refinement as refinement

CHECKPOINT_SHA = '85986c011c6298d0f000b12c89d9d43375269242037734261faa4170285884ba'
METRIC_NAMES = ['AP50_95','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']
SELECTION_RULE = {
    'split':'val', 'first_tile_size':640, 'fallback_tile_size':512,
    'overlap':.25, 'inference_imgsz':960, 'confidence_floor':.001,
    'operating_confidence':.15, 'max_detections_per_image':100,
    'eligibility':'AP_small strictly improves and AP50_95 >= baseline AP50_95 - 0.01',
    'fallback':'Only if 640 is ineligible, evaluate 512; retain best eligible AP_small. If neither eligible, investigate without claiming improvement.',
    'test_data_used_for_selection':False,
    'operating_rule':'Filter raw global and tile scores before cross-view mask deduplication, then keep highest-score 100.'}


def require(ok, message):
    if not ok: raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))


def write_json(path, data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def fingerprint(path):
    path=Path(path).resolve(strict=True)
    require(path.is_file(), 'Expected file: '+str(path))
    return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}


def digest_json(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def utc(): return datetime.now(timezone.utc).isoformat()


def encode_mask(mask):
    rle=mask_api.encode(np.asfortranarray(mask,dtype=np.uint8))
    if isinstance(rle['counts'],bytes):rle['counts']=rle['counts'].decode('ascii')
    rle['size']=[int(x) for x in rle['size']]
    return rle


def decode_mask(rle, width, height):
    require(isinstance(rle,dict) and rle.get('size')==[height,width] and isinstance(rle.get('counts'),str),'Invalid RLE grid or encoding')
    value=mask_api.decode(rle)
    require(value.shape==(height,width) and np.isin(value,(0,1)).all(),'Invalid RLE mask')
    return value.astype(bool,copy=False)


def key_for(path):
    s=Path(path).stem
    return int(s) if s.isnumeric() else s


def preflight(dataset, baseline, model):
    dataset,baseline,model=[Path(x).resolve(strict=True) for x in (dataset,baseline,model)]
    ep=baseline/'EVALUATION_MANIFEST.json';mp=baseline/'METRICS.json';pp=baseline/'predictions.json';sp=baseline/'DATASET_SNAPSHOT.json'
    e,m,s=map(read_json,(ep,mp,sp))
    require(e['status']==m['status']=='complete' and e['split']==m['split']=='val','Only a complete VAL baseline is allowed; test is forbidden')
    require(e['checkpoint']['sha256']==m['checkpoint_sha256']==sha(model)==CHECKPOINT_SHA,'Wrong or changed trained checkpoint')
    require(e['dataset_sha256']==m['dataset_sha256']==s['dataset_sha256'],'Baseline dataset identity mismatch')
    require(Path(s['root']).resolve()==dataset and s['names']=={'0':'dechet'},'Wrong baseline dataset or class mapping')
    for path,record in [(mp,e['metrics_json']),(pp,e['predictions_json']),(sp,e['dataset_snapshot'])]:
        require(sha(path)==record['sha256'],'Baseline evidence changed: '+path.name)
    a=e['evaluation_arguments']
    expected={'split':'val','imgsz':960,'conf':.001,'iou':.5,'max_det':100,'nms':True,'retina_masks':True,'augment':False,'batch':1}
    require(all(a[k]==v for k,v in expected.items()),'Baseline inference settings differ from frozen comparison')
    require(m['segmentation_backend_end2end'] is False,'Conventional NMS baseline required')
    frozen=[fingerprint(p) for p in [ep,mp,pp,sp,model,Path(__file__),Path(refinement.__file__)]]
    evidence={str(Path(x['path']).resolve()):x['sha256'] for x in s['source_evidence']}
    vp,cp=dataset/'manifest_val.json',dataset/'COCO_ORIGINAL_val.json'
    for p in [vp,cp]:
        require(evidence.get(str(p))==sha(p),'VAL mapping or original polygons changed: '+p.name);frozen.append(fingerprint(p))
    rows,original=read_json(vp),read_json(cp)
    require(len(rows)==e['image_count']==m['image_count']==s['splits']['val']['image_count'],'VAL image count differs')
    byid={x['image_id']:x for x in rows};im={x['id']:x for x in original['images']}
    require(len(byid)==len(rows) and set(byid)==set(im),'VAL image IDs duplicate or differ')
    pairs={str(Path(p['image']['path']).resolve()):p for p in s['splits']['val']['pairs']}
    require(set(pairs)=={str(Path(r['prepared_image']).resolve()) for r in rows},'Exact prepared VAL file set differs')
    mapping={}
    for r in rows:
        require(r['split']=='val' and r['status']=='ok' and r['download_kind']=='original','Unverified VAL row')
        image=im[r['image_id']];pw,ph=r['prepared_width'],r['prepared_height'];sw,sh=r['source_width'],r['source_height']
        require(image['file_name']==r['file_name'] and (image['width'],image['height'])==(sw,sh),'Source dimensions/filename mismatch')
        require(r['image_transform']=='RGB_JPEG_quality95_maxside1280_no_upscale' and 0<pw<=sw and 0<ph<=sh,'Unknown transformation')
        require(abs(r['scale_x']-pw/sw)<1e-12 and abs(r['scale_y']-ph/sh)<1e-12,'Scale mismatch')
        path=Path(r['prepared_image']).resolve();proof=fingerprint(path)
        require(proof['sha256']==r['prepared_image_sha256']==pairs[str(path)]['image']['sha256'],'Prepared VAL image changed')
        with Image.open(path) as img:require(img.size==(pw,ph) and img.mode=='RGB','Prepared image metadata mismatch')
        frozen.append(proof);key=key_for(path);require(key not in mapping,'Ambiguous prediction image ID');mapping[key]=r
    pred=read_json(pp)
    require(len(pred)==e['prediction_count']==m['prediction_count'],'Baseline prediction count mismatch')
    grouped=defaultdict(list)
    for q in pred:
        require(q['image_id'] in mapping,'Prediction outside VAL');r=mapping[q['image_id']]
        require(q.get('file_name')==Path(r['prepared_image']).name and q['category_id']==1,'Baseline filename/category mismatch')
        require(math.isfinite(q['score']) and 0<=q['score']<=1,'Invalid baseline score')
        require(q['segmentation']['size']==[r['prepared_height'],r['prepared_width']],'Baseline RLE grid differs')
        grouped[r['image_id']].append(q)
    require(all(len(x)<=100 for x in grouped.values()),'Baseline exceeds max100 per image')
    require(len(original['annotations'])==m['ground_truth_instances']==s['splits']['val']['instance_count'],'VAL annotation count mismatch')
    return sorted(rows,key=lambda r:r['image_id']),original,grouped,frozen,m['dataset_sha256']


def prepared_ground_truth(original, rows):
    """Scale every original component; do not create YOLO bridges or invert EXIF."""
    result=deepcopy(original);byid={r['image_id']:r for r in rows}
    result['categories']=[{'id':1,'name':'dechet','supercategory':'dechet'}]
    for im in result['images']:
        r=byid[im['id']];im['width']=r['prepared_width'];im['height']=r['prepared_height'];im['file_name']=Path(r['prepared_image']).name
    ids=set()
    for ann in result['annotations']:
        require(ann['id'] not in ids and ann['image_id'] in byid,'Unknown or duplicate GT annotation');ids.add(ann['id'])
        require(ann.get('iscrowd',0)==0 and not ann.get('ignore',False),'Operating P/R supports non-crowd, non-ignored TACO annotations only')
        r=byid[ann['image_id']];sx,sy=r['scale_x'],r['scale_y'];polygons=ann['segmentation']
        require(isinstance(polygons,list) and polygons,'Original polygon components required')
        scaled=[]
        for p in polygons:
            require(len(p)>=6 and len(p)%2==0 and all(math.isfinite(float(x)) for x in p),'Invalid GT polygon')
            scaled.append([float(x)*(sx if i%2==0 else sy) for i,x in enumerate(p)])
        ann['segmentation']=scaled;ann['category_id']=1
    gt=COCO();gt.dataset=result;gt.createIndex()
    for ann in result['annotations']:
        rle=gt.annToRLE(ann);area=float(mask_api.area(rle));require(area>0,'Empty prepared GT')
        ann['area']=area;ann['bbox']=[float(x) for x in mask_api.toBbox(rle)]
    return gt,result


def baseline_objects(predictions, row):
    objects=[];empty=0;w,h=row['prepared_width'],row['prepared_height']
    for i,p in enumerate(predictions):
        mask=decode_mask(p['segmentation'],w,h);yy,xx=np.nonzero(mask)
        if not len(xx):empty+=1;continue
        objects.append({'model_source':'litter','class_id':0,'class_name':'dechet','confidence_score':float(p['score']),
                        'raw_detection_index':i,'mask':mask,'mask_bbox':[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1],
                        'touches_crop_border':False,'view_id':'global','view_kind':'global'})
    return objects,empty


def serialize_object(obj):
    keys=['model_source','class_id','class_name','confidence_score','raw_detection_index','mask_bbox','model_bbox_original_px','inference_crop_xyxy','touches_crop_border','view_id','view_kind','tile_index','tile_detection_index']
    result={k:obj[k] for k in keys if k in obj};result['segmentation']=encode_mask(obj['mask']);return result


def deserialize_object(record,row):
    obj=dict(record);rle=obj.pop('segmentation');obj['mask']=decode_mask(rle,row['prepared_width'],row['prepared_height'])
    require(obj['model_source']=='litter' and obj['class_id']==0 and obj['class_name']=='dechet','Unexpected cached tile class')
    return obj


def as_coco(objects,image_id):
    return [{'image_id':image_id,'category_id':1,'score':float(o['confidence_score']),'segmentation':encode_mask(o['mask'])} for o in objects]


def merge_for_threshold(global_objects,tile_objects,confidence):
    # Filtering occurs BEFORE non-truncated-mask preference, including for .15.
    glob=[o for o in global_objects if o['confidence_score']>=confidence]
    tiles=[o for o in tile_objects if o['confidence_score']>=confidence]
    merged,audit=refinement.merge_multiscale(glob,tiles)
    merged=sorted(merged,key=lambda o:-o['confidence_score'])[:100]
    return merged,audit


def coco_metrics(gt,predictions):
    if predictions:
        dt=gt.loadRes(deepcopy(predictions))
    else:
        dt=COCO();dt.dataset={'images':deepcopy(gt.dataset['images']),'categories':deepcopy(gt.dataset['categories']),'annotations':[]};dt.createIndex()
    ev=COCOeval_faster(gt,dt,iouType='segm',print_function=print,boundary_cpu_count=1,rle_iou_max_workers=1)
    ev.params.imgIds=sorted(gt.getImgIds());ev.params.catIds=[1];ev.params.maxDets=[1,10,100]
    ev.evaluate();ev.accumulate();ev.summarize()
    return {n:None if float(v)<0 else float(v) for n,v in zip(METRIC_NAMES,ev.stats)}


def operating_pr(gt,predictions,threshold=.15):
    """Exact micro counts, score-sorted greedy one-to-one mask IoU>=.5, max100."""
    byimage=defaultdict(list)
    for p in predictions:
        require(p['score']>=threshold-1e-12,'Operating predictions were not prefiltered');byimage[p['image_id']].append(p)
    tp=fp=fn=0;small_tp=small_gt=0
    for image_id in sorted(gt.getImgIds()):
        anns=gt.loadAnns(gt.getAnnIds(imgIds=[image_id],catIds=[1]));gs=[gt.annToRLE(a) for a in anns]
        ps=sorted(byimage[image_id],key=lambda p:-p['score'])[:100]
        ious=mask_api.iou([p['segmentation'] for p in ps],gs,[0]*len(gs)) if ps and gs else np.zeros((len(ps),len(gs)))
        matched=set();this_tp=0
        for i in range(len(ps)):
            choices=[(float(ious[i,j]),j) for j in range(len(gs)) if j not in matched and ious[i,j]>=.5]
            if choices:
                _,j=max(choices,key=lambda z:(z[0],z[1]));matched.add(j);this_tp+=1
        tp+=this_tp;fp+=len(ps)-this_tp;fn+=len(gs)-this_tp
        small_gt+=sum(a['area']<=1024 for a in anns);small_tp+=sum(anns[j]['area']<=1024 for j in matched)
    p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.
    return {'confidence':threshold,'mask_iou':.5,'true_positive':tp,'false_positive':fp,'false_negative':fn,'precision':p,'recall':r,'F2':5*p*r/(4*p+r) if p or r else 0.,'small_reference_count':small_gt,'small_true_positive':small_tp,'small_recall':small_tp/small_gt if small_gt else None,'scope':'micro counts on prepared-grid non-crowd TACO VAL masks, after raw-score filtering and fusion; not TSV safety accuracy'}


def eligible(base,new):
    require(base['AP_small'] is not None and new['AP_small'] is not None,'No small-object reference metric available')
    return new['AP_small']>base['AP_small'] and new['AP50_95']>=base['AP50_95']-.01


def check_inputs(frozen):
    for f in frozen:require(sha(f['path'])==f['sha256'],'Frozen input changed: '+f['path'])


def run(args):
    rows,original,baseline,frozen,dataset_sha=preflight(args.dataset,args.baseline_evaluation,args.model)
    out=args.out.resolve();helper=next(x for x in frozen if x['path']==str(Path(refinement.__file__).resolve()))
    protocol=dict(SELECTION_RULE,tile_size=args.tile_size)
    config={'protocol':protocol,'model_sha256':sha(args.model),'dataset_sha256':dataset_sha,'program_sha256':sha(__file__),'helper_sha256':helper['sha256'],
            'baseline_predictions_sha256':sha(args.baseline_evaluation/'predictions.json'),'validation_manifest_sha256':sha(args.dataset/'manifest_val.json'),
            'validation_coco_sha256':sha(args.dataset/'COCO_ORIGINAL_val.json')}
    config_sha=digest_json(config);manifest_path=out/'EVALUATION_MANIFEST.json'
    if out.exists():
        require(args.resume,'Output exists: use --resume only for this matching unfinished run');manifest=read_json(manifest_path)
        require(manifest['configuration_sha256']==config_sha,'Resume configuration or inputs changed')
        require(manifest['status']!='complete','Evaluation already complete; refusing mutation')
    else:
        require(not args.resume,'Cannot resume a nonexistent run');out.mkdir(parents=True,exist_ok=False);(out/'CACHE_TILES').mkdir()
        manifest={'status':'prepared','started_at':utc(),'configuration':config,'configuration_sha256':config_sha,'inputs':frozen,'cache':{},'inference_segments':[],
                  'split':'val','test_images_or_annotations_opened':False,'cached_snapshot_contains_other_split_metadata':True,
                  'ground_truth':'Original multipart polygons scaled into prepared grid; canonical COCO raster areas/bboxes; no YOLO bridges, no inverse EXIF',
                  'selection_rule_frozen_before_metrics':SELECTION_RULE,'source_image_count':len(rows),'checkpoint_sha256':sha(args.model),'dataset_sha256':dataset_sha,
                  'known_limitations':['Prepared-grid small-object bins differ from old source-grid bins.','Scores are validation selection evidence, not independent test or TSV performance.','Cached global scores retain native JSON rounding.','Both arms discard empty masks; those counts are reported.','COCO AP uses the .001 merged snapshot; operating P/R re-filters raw scores before fusion.']}
        write_json(manifest_path,manifest)
    gt,gtdata=prepared_ground_truth(original,rows);gtpath=out/'GROUND_TRUTH_PREPARED.json'
    if gtpath.exists():require(read_json(gtpath)==gtdata,'Prepared GT output changed')
    else:write_json(gtpath,gtdata)
    manifest['ground_truth_output']=fingerprint(gtpath);write_json(manifest_path,manifest)
    if args.prepare_only:
        print(json.dumps({'status':manifest['status'],'images':len(rows),'configuration_sha256':config_sha}),flush=True);return
    os.environ['YOLO_AUTOINSTALL']='false';os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(tempfile.gettempdir())/'gabarit2d_val_yolo'))
    import cv2,torch,ultralytics
    from ultralytics import YOLO,settings
    settings.update({'sync':False});torch.set_num_threads(4);cv2.setNumThreads(0)
    device=args.device
    if device=='auto':device='mps' if torch.backends.mps.is_available() else 'cpu'
    require(device!='mps' or torch.backends.mps.is_available(),'MPS requested but unavailable')
    if device=='mps':torch.mps.set_per_process_memory_fraction(.5)
    segment={'started_at':utc(),'device':device,'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'opencv':cv2.__version__,'numpy':np.__version__,'new_images':0,'reused_images':0}
    manifest['inference_segments'].append(segment);manifest['status']='running';write_json(manifest_path,manifest)
    started=time.perf_counter();model=None
    base_predictions=[];tiled_predictions=[];base_operating=[];tiled_operating=[];audits=[];total_empty=0
    try:
        for number,row in enumerate(rows,1):
            ident=row['image_id'];cachepath=out/'CACHE_TILES'/f'{ident:06d}.json';key=str(ident)
            if key in manifest['cache']:
                proof=manifest['cache'][key];require(cachepath.is_file() and sha(cachepath)==proof['sha256'],'Cached tile RLE changed or missing')
                cache=read_json(cachepath);require(cache['configuration_sha256']==config_sha and cache['image_sha256']==row['prepared_image_sha256'] and cache['image_id']==ident,'Cached tile identity differs')
                tiles=[deserialize_object(o,row) for o in cache['objects']];segment['reused_images']+=1
            else:
                require(not cachepath.exists(),'Orphan cache file without recorded hash; preserve it and investigate before resuming')
                if model is None:
                    model=YOLO(str(args.model));require(model.task=='segment' and model.names=={0:'dechet'},'Wrong model task/classes')
                bgr=cv2.imread(row['prepared_image']);require(bgr is not None and bgr.shape[:2]==(row['prepared_height'],row['prepared_width']),'Unreadable prepared image')
                tiles,empty=refinement.predict_tiled_litter(model,bgr,[0,0,row['prepared_width'],row['prepared_height']],imgsz=960,conf=.001,device=device,tile_size=args.tile_size,overlap=.25)
                cache={'configuration_sha256':config_sha,'image_id':ident,'image_sha256':row['prepared_image_sha256'],'empty_masks':empty,'tile_count':len(refinement.tile_windows([0,0,row['prepared_width'],row['prepared_height']],row['prepared_width'],row['prepared_height'],args.tile_size,.25)), 'objects':[serialize_object(o) for o in tiles]}
                write_json(cachepath,cache);manifest['cache'][key]=fingerprint(cachepath);segment['new_images']+=1;write_json(manifest_path,manifest);del bgr
            global_objects,empty_global=baseline_objects(baseline.get(ident,[]),row);total_empty+=empty_global
            base=sorted(global_objects,key=lambda o:-o['confidence_score'])[:100]
            fused,mergeaudit=merge_for_threshold(global_objects,tiles,.001)
            opbase=[o for o in base if o['confidence_score']>=.15];opfused,opaudit=merge_for_threshold(global_objects,tiles,.15)
            base_predictions.extend(as_coco(base,ident));tiled_predictions.extend(as_coco(fused,ident));base_operating.extend(as_coco(opbase,ident));tiled_operating.extend(as_coco(opfused,ident))
            audits.append({'image_id':ident,'global_nonempty':len(global_objects),'global_empty_dropped':empty_global,'tile_nonempty':len(tiles),'tile_empty':cache['empty_masks'],'tile_count':cache['tile_count'],'fused_capped':len(fused),'suppressed_ap':len(mergeaudit),'operating_global':len(opbase),'operating_fused':len(opfused),'suppressed_operating':len(opaudit)})
            if number%10==0 or number==len(rows):print(json.dumps({'images':number,'total':len(rows),'tile_size':args.tile_size,'elapsed_s':round(time.perf_counter()-started,1),'cache_new':segment['new_images'],'cache_reused':segment['reused_images']}),flush=True)
            del tiles,global_objects,base,fused,opbase,opfused,cache,mergeaudit,opaudit;gc.collect()
            if device=='mps':torch.mps.empty_cache()
        for name,value in [('BASELINE_PREDICTIONS.json',base_predictions),('TILED_PREDICTIONS.json',tiled_predictions),('BASELINE_OPERATING_PREDICTIONS.json',base_operating),('TILED_OPERATING_PREDICTIONS.json',tiled_operating),('IMAGE_AUDIT.json',audits)]:write_json(out/name,value)
        basem=coco_metrics(gt,base_predictions);tilem=coco_metrics(gt,tiled_predictions)
        metrics={'status':'complete','split':'val','images':len(rows),'reference_objects':len(gtdata['annotations']),'baseline':basem,'tiled':tilem,
                 'delta':{k:None if basem[k] is None or tilem[k] is None else tilem[k]-basem[k] for k in basem},
                 'baseline_operating':operating_pr(gt,base_operating),'tiled_operating':operating_pr(gt,tiled_operating),
                 'eligible_by_frozen_rule':eligible(basem,tilem),'selection_rule':SELECTION_RULE,'tile_size':args.tile_size,
                 'checkpoint_sha256':sha(args.model),'dataset_sha256':dataset_sha,'configuration_sha256':config_sha,
                 'ground_truth_grid':'prepared image pixels','AP_small_area_definition':'canonical raster area <= 32^2 prepared pixels; not source-grid area or physical size',
                 'global_empty_masks_dropped_from_both_arms':total_empty,'scope':'Validation for selecting tiled inference only. No test run or TSV accuracy claim.'}
        write_json(out/'METRICS.json',metrics);check_inputs(frozen)
        segment.update(completed_at=utc(),elapsed_s=time.perf_counter()-started)
        manifest.update(status='complete',completed_at=utc(),inputs_unchanged=True,image_count=len(rows),metrics=fingerprint(out/'METRICS.json'),
                        output_files=[fingerprint(out/n) for n in ['GROUND_TRUTH_PREPARED.json','BASELINE_PREDICTIONS.json','TILED_PREDICTIONS.json','BASELINE_OPERATING_PREDICTIONS.json','TILED_OPERATING_PREDICTIONS.json','IMAGE_AUDIT.json','METRICS.json']])
        write_json(manifest_path,manifest);print(json.dumps({'status':'complete','tile_size':args.tile_size,'eligible':metrics['eligible_by_frozen_rule'],'baseline':basem,'tiled':tilem,'baseline_operating':metrics['baseline_operating'],'tiled_operating':metrics['tiled_operating']}),flush=True)
    except BaseException as exc:
        segment.update(stopped_at=utc(),elapsed_s=time.perf_counter()-started);manifest.update(status='failed',error=repr(exc));write_json(manifest_path,manifest);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['dataset','baseline-evaluation','model','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tile-size',type=int,choices=[640,512],default=640);p.add_argument('--device',choices=['auto','mps','cpu'],default='auto')
    p.add_argument('--resume',action='store_true');p.add_argument('--prepare-only',action='store_true');run(p.parse_args())


if __name__=='__main__':main()
