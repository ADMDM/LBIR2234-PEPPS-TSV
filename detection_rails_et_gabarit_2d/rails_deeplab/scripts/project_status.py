"""Read actual project provenance, annotations and diagnostic values without training."""
import argparse, csv, hashlib, json, math, statistics, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.after_annotation import collect


def read(path):return json.loads(Path(path).read_text())
def rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def snapshot(root=ROOT):
    root=Path(root).resolve()
    _,annotations=collect(root)
    run=root/read(root/'configs/diagnostic_target.json')['run']
    metadata=read(run/'modele/run.json');history=read(run/'modele/history.json')
    init=read(run/'initialisation.json');pretrain=read(root/'configs/railsem_pretraining.json')
    checkpoint=root/init['checkpoint'];best=root/'runs'/Path(run).name/'modele/best.pt'
    evaluation=read(run/'evaluation_validation/metrics.json')
    per_image=rows(run/'diagnostic_validation/metriques_par_image.csv')
    sections=rows(run/'diagnostic_validation/erreurs_axe_sections.csv')
    e=[float(r['center_error_px']) for r in sections]
    c=evaluation['metrics']['confusion'];tp,fp,fn=c[1][1],c[0][1],c[1][0]
    import numpy as np
    normalized=[100*abs(float(r['center_error_px']))/float(r['width_px']) for r in sections]
    measurements=dict(images=len(per_image),matched_sections=len(e),
        reference_sections=sum(int(r['reference_points']) for r in per_image),
        dice_macro=statistics.mean(float(r['dice']) for r in per_image),
        iou_macro=statistics.mean(float(r['iou']) for r in per_image),
        dice_micro=2*tp/(2*tp+fp+fn),iou_micro=tp/(tp+fp+fn),
        precision_micro=tp/(tp+fp),recall_micro=tp/(tp+fn),
        true_positive_pixels=tp,false_positive_pixels=fp,false_negative_pixels=fn,
        axis_mae_px=statistics.mean(abs(x) for x in e),
        axis_rmse_px=math.sqrt(statistics.mean(x*x for x in e)),
        axis_bias_px=statistics.mean(e),axis_p95_absolute_px=float(np.percentile(np.abs(e),95)),
        axis_max_absolute_px=max(map(abs,e)),axis_mae_pct_image_width=statistics.mean(normalized),
        width_mae_px=statistics.mean(abs(float(r['width_error_px'])) for r in sections),
        width_bias_px=statistics.mean(float(r['width_error_px']) for r in sections),
        coverage_pct=100*len(e)/sum(int(r['reference_points']) for r in per_image),
        spatial_units='original_image_pixels',scope='12 images of one model-selection capture; human polygon reference, no independent final test')
    assert math.isclose(measurements['dice_micro'],evaluation['metrics']['per_class']['couloir']['dice'],abs_tol=1e-12)
    assert math.isclose(measurements['axis_mae_px'],evaluation['axis']['mean_absolute_center_error_px'],abs_tol=1e-12)
    batch=root/'04_ANNOTATIONS/RENFORT_2_LOTS_30_20261014'
    by_lot=defaultdict(lambda:dict(assigned=0,reviewed=0,with_shapes=0,excluded=0,missing=0))
    for r in read(batch/'MANIFESTE.json')['cases']:
        key=r.get('lot_id') or r['role'];count=by_lot[key];count['assigned']+=1
        p=root/r['json']
        if not p.exists():count['missing']+=1;continue
        a=read(p);count['reviewed']+=a.get('flags',{}).get('reviewed') is True
        count['with_shapes']+=bool(a.get('shapes'))
        count['excluded']+=bool(a.get('flags',{}).get('exclude_from_training'))
    ego=root/'data/prepared/railsem_egopath_20261014';prep=read(ego/'preparation.json')
    counts={key:len(rows(ego/(key+'.csv'))) for key in ['train','val','test','unassigned']}
    assert counts==prep['prepared']
    generic=root/'data/prepared/railsem_track_paper_v1'
    generic_counts={key:len(rows(generic/(key+'.csv'))) for key in ['train','val','test']}
    observed_ego_runs=[]
    for p in (root/'runs').glob('*/*/run.json'):
        if read(p)['arguments']['task']=='railsem_egopath':observed_ego_runs.append(str(p.relative_to(root)))
    stability=read(root/'03_RESULTATS/AMELIORATIONS_20261014/stabilite/RESULTATS_STABILITE.json')
    resolution=defaultdict(list)
    for r in rows(run/'diagnostic_validation/sensibilite_resolution.csv'):
        resolution[(int(r['network_width']),int(r['network_height']))].append(r)
    resolution_stats=[dict(width=w,height=h,images=len(rr),dice_macro=statistics.mean(float(r['dice']) for r in rr),
        coverage_macro_pct=statistics.mean(float(r['coverage_pct']) for r in rr),
        inference_median_seconds=statistics.median(float(r['inference_seconds']) for r in rr)) for (w,h),rr in resolution.items()]
    # The active video's lineage is checked separately from the current source code.
    video=read(root/'03_RESULTATS/AMELIORATIONS_20261014/video_nouvelle_capture_60s/video.json')
    return dict(captured_at=datetime.now(ZoneInfo('Europe/Brussels')).isoformat(),
        base_annotations=annotations,reinforcement=dict(by_lot),
        active_model=dict(run=str(run.relative_to(root)),architecture='DeepLabV3Plus/ResNet50',
            training_images=len(rows(run/'dataset/train.csv')),validation_images=len(rows(run/'dataset/val.csv')),
            completed_epochs=len(history),selected_epoch=max(history,key=lambda h:h['metrics']['foreground_dice'])['epoch'],
            device=metadata['device'],initialization=init,
            initialization_checkpoint_hash_matches=sha(checkpoint)==init['checkpoint_sha256'],
            checkpoint_hash_matches_evaluation=sha(best)==evaluation['checkpoint_sha256'],
            network_dimensions=[metadata['arguments']['width'],metadata['arguments']['height']],
            first_epoch={k:history[0][k] for k in ['epoch','train_loss','val_loss']},
            last_epoch={k:history[-1][k] for k in ['epoch','train_loss','val_loss']}),
        railsem_generic=dict(pretraining_status=pretrain['status'],completed_epochs=pretrain['completed_epochs'],split_counts=generic_counts,used_by_current_model=init['source']=='railsem19'),
        railsem_egopath=dict(source_annotations=prep['source_annotations'],split_counts=counts,training_runs_found=observed_ego_runs,
            video_sequence_independence_established=prep['video_sequence_independence_established'],tsv_inner_edges_equivalent=prep['tsv_inner_edges_equivalent']),
        validation=measurements,resolution_experiment=resolution_stats,stability=stability,
        new_capture_video=dict(processing_fps=video['processing_fps'],source_fps=video['fps'],device=video['device'],duration_seconds=video['duration_seconds'],checkpoint_matches_evaluation=video['checkpoint_sha256']==evaluation['checkpoint_sha256']),
        calibration=read(root/'configs/processing_measurement.json')['metric_calibration'],
        model_retrained_by_this_script=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path);args=parser.parse_args()
    result=snapshot()
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
