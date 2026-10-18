"""Evaluate the remaining public holdout once, with immutable existing models."""
from pathlib import Path
import sys, json, csv, time, argparse
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'railtsv').is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/diagnostic_axes_20261018'))
from axes_eval_20261018 import (MODELS, EGO, device, torch, np, read_manifest,
    rgb_read, mask_read, predict_image, image_metrics, build_model, load_checkpoint,
    sha256, dump, write_csv, stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    prepared = ROOT / 'data/prepared/railsem_egopath_20261014'
    prior = json.loads((ROOT / 'runs/fusion_axes_20261018/test_public_60/selection.json').read_text())
    used = {row['image_sha256'] for row in prior['images']}
    selected = []
    for row in read_manifest(prepared / 'test.csv'):
        digest = sha256(row['image'])
        if digest not in used:
            selected.append(dict(row, image_sha256=digest, mask_sha256=sha256(row['mask'])))
    assert len(selected) == 419
    assert len({r['image_sha256'] for r in selected}) == 419
    forbidden = set()
    manifests = [prepared / 'train.csv', prepared / 'val.csv', EGO / 'dataset/train.csv',
                 EGO / 'dataset/val.csv', ROOT / 'data/prepared/railsem_track_paper_v1/train.csv',
                 ROOT / 'data/prepared/railsem_track_paper_v1/val.csv']
    for p in manifests:
        forbidden.update(r['image'] for r in read_manifest(p))
    collisions = [p for p in sorted(forbidden) if sha256(p) in {r['image_sha256'] for r in selected}]
    assert not collisions, collisions
    models = ['ancien_tsv', 'ego_tsv', 'public_seul_vga']
    dump(args.out / 'PROTOCOLE_AVANT_TEST.json', dict(
        images=selected, count=419, previous_60_excluded=True,
        models={n:dict(checkpoint_sha256=sha256(MODELS[n][0]), height=480, width=640) for n in models},
        primary_comparison='Frozen ego_tsv versus frozen ancien_tsv; public checkpoint is an additional reference',
        inference_geometry='640x480 letterbox, original image coordinates for metrics',
        score_threshold=.55, profile_step_px=8, no_training=True, no_tuning_on_test=True,
        all_generic_public_and_tsv_train_validation_byte_overlap=False,
        source_video_independence_proven=False, independent_tsv_test=False,
        clinical_or_physical_accuracy=False, manual_switch_labels_available=False,
        source_script_sha256=sha256(__file__),
        note='Remaining public holdout; cannot certify field performance or physical metres.'))
    dev = device('auto'); torch.set_num_threads(4)
    all_images, all_points, summaries = [], [], []
    started = time.perf_counter()
    for name in models:
        ck = load_checkpoint(MODELS[name][0]); model = build_model(2)
        model.load_state_dict(ck['state_dict']); model.to(dev).eval()
        images, points = [], []
        for i, row in enumerate(selected, 1):
            rgb = rgb_read(row['image']); ref = mask_read(row['mask'], [0, 1, 255])
            pred, prob = predict_image(model, rgb, dev, 480, 640)
            m, pp, _ = image_metrics(pred, ref, prob, Path(row['image']).stem, 'public_reserved_419')
            images.append(dict(model=name, **m)); points.extend(dict(model=name, **p) for p in pp)
            if i % 25 == 0 or i == 419:
                print(f'{name}: {i}/419; {time.perf_counter()-started:.0f}s', flush=True)
        nref = sum(r['reference_points'] for r in images)
        summaries.append(dict(model=name, images=len(images),
            dice_macro=float(np.mean([r['dice'] for r in images])),
            iou_macro=float(np.mean([r['iou'] for r in images])),
            coverage_pct=100*len(points)/nref, reference_points=nref,
            **stats([p['center_error_px'] for p in points])))
        write_csv(args.out / f'{name}_PAR_IMAGE.csv', images)
        write_csv(args.out / f'{name}_ERREURS.csv', points)
        all_images.extend(images); all_points.extend(points)
        del model, ck
        if dev.type == 'mps': torch.mps.empty_cache()
    write_csv(args.out / 'METRIQUES.csv', summaries)
    write_csv(args.out / 'PAR_IMAGE.csv', all_images)
    by_model = {n:{(r['image'], r['y_px']):r for r in all_points if r['model']==n} for n in models}
    common = sorted(set(by_model[models[0]]) & set(by_model[models[1]]))
    comparison = [dict(model=n, **stats([by_model[n][k]['center_error_px'] for k in common])) for n in models[:2]]
    write_csv(args.out / 'COMPARAISON_AXE_SECTIONS_COMMUNES.csv', comparison)
    dump(args.out / 'TERMINE.json', dict(complete=True, images=419, models=models,
        seconds=time.perf_counter()-started, device=str(dev), metrics_verified_against_csv=False,
        public_holdout_consumed=True, final_tsv_field_test=False))
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
