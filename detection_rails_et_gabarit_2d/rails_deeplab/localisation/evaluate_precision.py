"""Recompute scoped accuracy evidence from saved references and predictions.

No new training or inference; OCR transcription, rail development errors and
public litter segmentation are separate evaluations with separate denominators.
COCO scoring uses the reference pycocotools implementation for an independent
check of the earlier faster-coco-eval result.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import numpy as np
from pyproj import Geod
from handoff_bundle import read_csv, sha256, write_json


def errors(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {'n': 0, 'mae': None, 'rmse': None, 'bias': None, 'p95_absolute': None}
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite reference error')
    return {'n': len(a), 'mae': float(np.mean(abs(a))), 'rmse': float(np.sqrt(np.mean(a*a))),
            'bias': float(a.mean()), 'p95_absolute': float(np.percentile(abs(a), 95))}


def ocr_accuracy(observation_file, reference_file):
    observations = {int(x['frame_index']): x for x in read_csv(observation_file)}
    references = read_csv(reference_file)
    if len({r['frame_index'] for r in references}) != len(references):
        raise ValueError('Repeated OCR reference frame')
    geod = Geod(ellps='WGS84'); records = []
    for ref in references:
        observed = observations.get(int(ref['frame_index']))
        distance = None
        if observed:
            if abs(float(observed['video_time_s'])-float(ref['video_time_s'])) > 1e-6:
                raise ValueError('OCR reference is not synchronized')
            distance = geod.inv(float(ref['longitude_deg']),float(ref['latitude_deg']),
                                float(observed['longitude_deg']),float(observed['latitude_deg']))[2]
        records.append(dict(frame_index=int(ref['frame_index']), accepted=observed is not None,
                            transcription_disagreement_m=distance))
    d = [r['transcription_disagreement_m'] for r in records if r['accepted']]
    return {'reference_frames':len(records), 'accepted':len(d), 'rejected':len(records)-len(d),
            'exact_to_1mm_numeric_tolerance':sum(x < .001 for x in d),
            'accepted_reading_disagreement':{'mean_m':float(np.mean(d)) if d else None,
                'rms_m':float(np.sqrt(np.mean(np.asarray(d)**2))) if d else None},
            'max_disagreement_m':max(d) if d else None,
            'scope':'Same-video development text-reading comparison; not GNSS accuracy or physical rail accuracy',
            'records':records}


def rail_accuracy(directory):
    directory = Path(directory)
    rows = read_csv(directory/'errors.csv')
    images = read_csv(directory/'images.csv')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['image']].append(row)
        expected = float(row['center_pred_px'])-float(row['center_ref_px'])
        if abs(expected-float(row['center_error_px'])) > 1e-6:
            raise ValueError('Saved axis error disagrees with coordinates')
        if abs(float(row['width_pred_px'])-float(row['width_ref_px'])-float(row['width_error_px'])) > 1e-6:
            raise ValueError('Saved width error disagrees with widths')
    metrics = {}
    per_image = []
    for label, column in [('left','left_error_px'),('right','right_error_px'),('axis','center_error_px'),('width','width_error_px')]:
        metrics[label] = errors([float(x[column]) for x in rows])
        means = [float(np.mean([abs(float(r[column])) for r in group])) for group in grouped.values()]
        metrics[label]['mean_per_image_mae_px'] = float(np.mean(means))
    for name, group in grouped.items():
        per_image.append(dict(image=name,sections=len(group),
                              **{label+'_mae_px':float(np.mean([abs(float(r[c])) for r in group])) for label,c in
                                 [('left','left_error_px'),('right','right_error_px'),('axis','center_error_px')]}))
    summary = json.loads((directory/'summary.json').read_text())
    macro = {key:float(np.mean([float(row[key]) for row in images])) for key in ['dice','iou','precision','recall']}
    n_ref = sum(int(x['reference_points']) for x in images)
    n_match = sum(int(x['matched_points']) for x in images)
    if n_match != len(rows):
        raise ValueError('Per-image matched counts disagree with error table')
    return {'images':len(images),'capture_groups':sorted({r['group'] for r in images}),
            'sections_with_predictions':len(rows), 'reference_sections':n_ref,
            'section_coverage_fraction':n_match/n_ref, 'pixel_segmentation_macro':macro, 'per_section_px':metrics,
            'per_image':per_image,'saved_summary':summary,
            'scope':'Saved development images; correlated sections, model selection already used these images; not video-wide or final independent accuracy'}


def coco_accuracy(directory, confidence=.15):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    directory = Path(directory)
    gt = COCO(str(directory/'GROUND_TRUTH_SINGLE_CLASS.json'))
    predictions = json.loads((directory/'PREDICTIONS_SOURCE_GRID.json').read_text())
    dt = gt.loadRes(predictions)
    evaluator = COCOeval(gt, dt, 'segm')
    evaluator.params.imgIds = sorted(gt.getImgIds())
    evaluator.params.catIds = [1]
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    names = ['AP50_95','AP50','AP75','AP_small','AP_medium','AP_large',
             'AR1','AR10','AR100','AR_small','AR_medium','AR_large']
    metrics = dict(zip(names, map(float, evaluator.stats)))
    tp = fp = total_gt = 0
    for e in evaluator.evalImgs:
        if e is None or e['aRng'] != evaluator.params.areaRng[0] or e['maxDet'] != 100:
            continue
        selected = np.asarray(e['dtScores']) >= confidence
        ignore = np.asarray(e['dtIgnore'])[0]
        matched = np.asarray(e['dtMatches'])[0] > 0
        tp += int(np.sum(selected & ~ignore & matched))
        fp += int(np.sum(selected & ~ignore & ~matched))
        total_gt += int(np.sum(~np.asarray(e['gtIgnore'],dtype=bool)))
    operating = {'confidence_threshold':confidence,'mask_iou':.5,'true_positive':tp,'false_positive':fp,
                 'false_negative':total_gt-tp, 'ground_truth_nonignored':total_gt,
                 'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/total_gt if total_gt else None}
    historical = json.loads((directory/'METRICS_COCO.json').read_text())['raw_COCO_stats'][:12]
    difference = max(abs(a-b) for a,b in zip(metrics.values(),historical))
    return {'images':len(gt.imgs),'reference_instances':len(gt.anns),'metrics':metrics,'operating_point':operating,
            'max_difference_vs_previous_engine':difference,'engine':'pycocotools 2.0.11',
            'scope':'Saved predictions on 173 public TACO test photos, masks on original annotation grid. Not TSV corridor or hazard accuracy. Not a new held-out dataset.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    paths={k:Path(v) if Path(v).is_absolute() else (args.config.parent/Path(v)).resolve() for k,v in cfg.items()}
    result={}
    if 'ocr_observations' in paths:
        result['ocr']=ocr_accuracy(paths['ocr_observations'],paths['ocr_references'])
    if 'rail_evaluation' in paths:
        result['rails']=rail_accuracy(paths['rail_evaluation'])
    if 'coco_evaluation' in paths:
        result['objects']=coco_accuracy(paths['coco_evaluation'])
    result['configuration_sha256']=sha256(args.config)
    result['inputs_sha256']={}
    for key,p in paths.items():
        files=[p] if p.is_file() else [p/name for name in
            (['errors.csv','images.csv','summary.json'] if key=='rail_evaluation' else
             ['GROUND_TRUTH_SINGLE_CLASS.json','PREDICTIONS_SOURCE_GRID.json','METRICS_COCO.json'])]
        result['inputs_sha256'][key]={f.name:sha256(f) for f in files}
    result['ground_truth_for_each_video_frame']=False
    result['physical_accuracy_m']=None
    write_json(args.out/'PRECISION_RECALCULEE.json',result)
    print(json.dumps({k:{x:v for x,v in value.items() if x not in ['records','per_image','saved_summary']} if isinstance(value,dict) else value for k,value in result.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
