#!/usr/bin/env python3
"""Exploratory VAL-only replay: preserve globals and add selective small tiles.

No model inference, training, image download, test split or mutation of V2 VAL.
The rejected initial 640 experiment stays intact. Freeze this new protocol before
the 12 configurations are scored, and retain no configuration if none qualifies.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from faster_coco_eval import COCO
import evaluate_tiled_validation as first
import small_object_refinement as coarse
import selective_refinement as selective


PROTOCOL = {
    'name': 'exploratory_selective_tiles_v1',
    'prior_640_recipe': 'rejected; original selection protocol abandoned explicitly; no 512 trial',
    'split': 'val', 'images': 288, 'reference_objects': 871,
    'tile_size': 640, 'nominal_overlap': .25, 'network_imgsz': 960,
    'operating_baseline_confidence': .15, 'ap_baseline_confidence': .001,
    'confidence_grid': [.35, .5, .65, .8],
    'area_fraction_grid': [.001, .002, .004],
    'first_stage': 'merge_multiscale(raw globals>=.15, raw tiles>=.15), keep tile survivors, no first-stage max100 truncation',
    'addition_rules': 'score>=minimum; mask area/(image_width*image_height)<=maximum; no crop-border or image-border contact; reject if mask IoU>.1 OR IoS>.2 with any operating baseline mask; process additions by descending score then stable candidate ID and reject the same overlaps with earlier selected additions',
    'baseline_policy': 'preserve every baseline mask; no replacement, union or suppression, including all low-score AP globals>=.001',
    'ap_policy': 'all unchanged global AP predictions plus selected additions; COCO maxDets100 only inside evaluation',
    'operating_policy': 'all unchanged global predictions>=.15 plus selected additions; score-sorted max100 only inside metric evaluation',
    'eligibility': {
        'precision': '>= baseline precision - .03',
        'small_recall': '> baseline small recall',
        'F2': '>= baseline F2',
        'AP_small': '> baseline AP_small',
        'AP50_95': '>= baseline AP50_95 - .01',
    },
    'ranking': 'highest operating small recall, then highest operating precision; exact ties use frozen grid order',
    'test_used': False,
    'limitation': 'Exploratory tuning on the same development validation, not an independent test or TSV obstacle-accuracy proof.',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return first.sha(path)


def fingerprint(path):
    return first.fingerprint(path)


def dump(path, value):
    return first.write_json(path, value)


def eligible(baseline, baseline_operating, candidate, operating):
    checks = dict(
        precision=operating['precision'] >= baseline_operating['precision'] - .03,
        small_recall=operating['small_recall'] > baseline_operating['small_recall'],
        F2=operating['F2'] >= baseline_operating['F2'],
        AP_small=candidate['AP_small'] > baseline['AP_small'],
        AP50_95=candidate['AP50_95'] >= baseline['AP50_95'] - .01)
    return all(checks.values()), checks


def choose(candidates):
    admissible = [c for c in candidates if c['eligible']]
    if not admissible:
        return None
    return max(admissible, key=lambda c: (c['operating']['small_recall'], c['operating']['precision'], -c['grid_index']))


def grouped(predictions):
    out = defaultdict(list)
    for p in predictions:
        out[p['image_id']].append(p)
    return out


def preflight(folder):
    folder = Path(folder).resolve(strict=True)
    m = read(folder / 'EVALUATION_MANIFEST.json')
    require(m['status'] == 'complete' and m['split'] == 'val', 'Completed VAL required; test forbidden')
    require(m['source_image_count'] == m['image_count'] == 288 and m['inputs_unchanged'], 'Wrong or mutable prior validation')
    require(m['configuration']['protocol']['tile_size'] == 640, 'Only the frozen 640 cache is supported')
    require(not m['test_images_or_annotations_opened'], 'Prior evaluation must not use test images or annotations')
    require(sha(first.__file__) == m['configuration']['program_sha256'], 'Prior evaluation program changed')
    require(sha(coarse.__file__) == m['configuration']['helper_sha256'], 'Prior fusion helper changed')
    first.check_inputs(m['inputs'])
    for item in m['output_files']:
        require(sha(item['path']) == item['sha256'], 'Prior output changed: ' + item['path'])
    frozen = [fingerprint(folder / 'EVALUATION_MANIFEST.json')]
    frozen += [fingerprint(folder / n) for n in ['GROUND_TRUTH_PREPARED.json', 'BASELINE_PREDICTIONS.json', 'BASELINE_OPERATING_PREDICTIONS.json', 'METRICS.json']]
    for ident, proof in m['cache'].items():
        p = folder / 'CACHE_TILES' / f'{int(ident):06d}.json'
        require(sha(p) == proof['sha256'], 'Raw tile cache changed')
        frozen.append(fingerprint(p))
    require(len(m['cache']) == 288, 'Incomplete raw tile cache')
    gt = COCO(str(folder / 'GROUND_TRUTH_PREPARED.json'))
    require(len(gt.getImgIds()) == 288 and len(gt.getAnnIds()) == 871, 'Wrong GT cardinality')
    require({str(i) for i in gt.getImgIds()} == set(m['cache']), 'Cache/GT image IDs differ')
    baseline = read(folder / 'BASELINE_PREDICTIONS.json')
    operating = read(folder / 'BASELINE_OPERATING_PREDICTIONS.json')
    def canonical(ps):
        return sorted((p['image_id'], p['score'], p['segmentation']['counts']) for p in ps)
    require(canonical(operating) == canonical([p for p in baseline if p['score'] >= .15]), 'Baseline confidence subsets differ')
    return folder, m, frozen, gt, baseline, operating


def truncation_counts(predictions):
    counts = [len(v) for v in grouped(predictions).values()]
    return {'images_above_max100': sum(n > 100 for n in counts), 'predictions_beyond_max100': sum(max(0, n-100) for n in counts)}


def run(args):
    folder, prior, frozen, gt, baseline, baseline_op = preflight(args.prior_validation)
    out = args.out.resolve()
    require(not out.exists(), 'Output exists; refusing overwrite')
    out.mkdir(parents=True)
    code = [fingerprint(p) for p in [Path(__file__), Path(selective.__file__), Path(first.__file__), Path(coarse.__file__)]]
    configuration = {'protocol': PROTOCOL, 'programs': code, 'prior_manifest_sha256': sha(folder / 'EVALUATION_MANIFEST.json'),
                     'checkpoint_sha256': prior['checkpoint_sha256'], 'dataset_sha256': prior['dataset_sha256']}
    config_sha = first.digest_json(configuration)
    manifest = dict(status='frozen_before_scoring', started_at=utc(), split='val', source_image_count=288,
                    test_images_or_annotations_opened=False, model_inference_performed=False, configuration=configuration,
                    configuration_sha256=config_sha, inputs=frozen+code, prior_validation_manifest=fingerprint(folder/'EVALUATION_MANIFEST.json'))
    dump(out / 'EVALUATION_MANIFEST.json', manifest)
    dump(out / 'PROTOCOL_FROZEN.json', {'frozen_at': utc(), 'configuration_sha256': config_sha, 'configuration': configuration})
    started = time.perf_counter()
    configs = [dict(grid_index=i, min_confidence=c, max_area_fraction=f) for i, (c, f) in enumerate((c, f) for c in PROTOCOL['confidence_grid'] for f in PROTOCOL['area_fraction_grid'])]
    additions = [[] for _ in configs]
    audits = []
    op_by_image = grouped(baseline_op)
    try:
        manifest['status'] = 'running'; dump(out/'EVALUATION_MANIFEST.json', manifest)
        for number, image_id in enumerate(sorted(gt.getImgIds()), 1):
            im = gt.imgs[image_id]
            row = dict(prepared_width=im['width'], prepared_height=im['height'])
            globals_, empty = first.baseline_objects(op_by_image[image_id], row)
            require(empty == 0, 'Operating baseline unexpectedly contains empty masks')
            for i, obj in enumerate(globals_):
                obj['candidate_id'] = f'b{i:04d}'
            cache = read(folder/'CACHE_TILES'/f'{image_id:06d}.json')
            require(cache['configuration_sha256'] == prior['configuration_sha256'] and cache['image_id'] == image_id, 'Cache configuration/identity differs')
            tiles = [first.deserialize_object(o, row) for o in cache['objects'] if o['confidence_score'] >= .15]
            for obj in tiles:
                obj['candidate_id'] = f"t{int(obj['raw_detection_index']):04d}"
            merged, _ = coarse.merge_multiscale(globals_, tiles)
            candidates = [o for o in merged if o['view_kind'] == 'tile']
            per_config = []
            for cfg in configs:
                kept, audit = selective.select_additions(globals_, candidates, cfg['min_confidence'], cfg['max_area_fraction'], image_width=im['width'], image_height=im['height'])
                additions[cfg['grid_index']].extend(first.as_coco(kept, image_id))
                per_config.append({'grid_index': cfg['grid_index'], 'kept': len(kept), 'audit': audit})
            audits.append({'image_id': image_id, 'raw_operating_tiles': len(tiles), 'first_stage_tile_candidates': len(candidates), 'configurations': per_config})
            del globals_, tiles, merged, candidates
            gc.collect()
            if number % 30 == 0:
                print(json.dumps({'images_replayed': number, 'total': 288, 'elapsed_s': round(time.perf_counter()-started, 1)}), flush=True)
        # Recompute the unchanged baseline once on exactly the same canonical GT.
        base_metrics = first.coco_metrics(gt, baseline)
        base_op_metrics = first.operating_pr(gt, baseline_op)
        previous = read(folder/'METRICS.json')
        for key, value in base_metrics.items():
            require(value == previous['baseline'][key], 'Baseline AP changed: '+key)
        require(base_op_metrics == previous['baseline_operating'], 'Baseline operating metrics changed')
        results = []; output_paths = []
        for cfg, extra in zip(configs, additions):
            key = f"config_{cfg['grid_index']:02d}"
            addition_path = out / (key + '_ADDITIONS.json'); dump(addition_path, extra); output_paths.append(addition_path)
            ap_predictions = baseline + extra
            op_predictions = baseline_op + extra
            metrics = first.coco_metrics(gt, ap_predictions)
            op = first.operating_pr(gt, op_predictions)
            ok, checks = eligible(base_metrics, base_op_metrics, metrics, op)
            result = dict(cfg, metrics=metrics, operating=op, eligible=ok, eligibility_checks=checks, additions=len(extra),
                          ap_truncation=truncation_counts(ap_predictions), operating_truncation=truncation_counts(op_predictions),
                          additions_file=fingerprint(addition_path))
            results.append(result)
            print(json.dumps({'configuration_scored': cfg, 'eligible': ok, 'checks': checks, 'AP_small': metrics['AP_small'], 'precision': op['precision'], 'small_recall': op['small_recall']}), flush=True)
        winner = choose(results)
        selection = dict(status='complete', split='val', images=288, reference_objects=871,
                         checkpoint_sha256=prior['checkpoint_sha256'], dataset_sha256=prior['dataset_sha256'],
                         selected_config=None if winner is None else {k:winner[k] for k in ['min_confidence','max_area_fraction']},
                         selected_eligible=winner is not None, baseline=base_metrics, baseline_operating=base_op_metrics,
                         selected=None if winner is None else winner['metrics'], selected_operating=None if winner is None else winner['operating'],
                         selected_grid_index=None if winner is None else winner['grid_index'], candidates=results,
                         frozen_rule=PROTOCOL, configuration_sha256=config_sha,
                         ground_truth_grid='prepared image pixels; canonical original COCO polygon masks',
                         scope='Exploratory development selection on the same VAL; no independent test or TSV accuracy claim.')
        dump(out/'SELECTION.json', selection); output_paths.append(out/'SELECTION.json')
        dump(out/'IMAGE_REPLAY_AUDIT.json', audits); output_paths.append(out/'IMAGE_REPLAY_AUDIT.json')
        if winner is not None:
            extra = additions[winner['grid_index']]
            for filename, predictions in [('SELECTED_AP_PREDICTIONS.json', baseline+extra), ('SELECTED_OPERATING_PREDICTIONS.json', baseline_op+extra)]:
                dump(out/filename, predictions); output_paths.append(out/filename)
        first.check_inputs(frozen+code)
        first.check_inputs(prior['inputs'])
        manifest.update(status='complete', completed_at=utc(), inputs_unchanged=True, image_count=288,
                        elapsed_s=time.perf_counter()-started, selection=fingerprint(out/'SELECTION.json'),
                        output_files=[fingerprint(p) for p in output_paths])
        dump(out/'EVALUATION_MANIFEST.json', manifest)
        print(json.dumps({'status':'complete','selected_config':selection['selected_config'],'selected_eligible':selection['selected_eligible']}), flush=True)
    except BaseException as e:
        manifest.update(status='failed', error=repr(e), stopped_at=utc(), elapsed_s=time.perf_counter()-started)
        dump(out/'EVALUATION_MANIFEST.json', manifest)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prior-validation', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
