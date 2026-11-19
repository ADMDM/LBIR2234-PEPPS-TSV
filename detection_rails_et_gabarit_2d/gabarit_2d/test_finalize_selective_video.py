"""CPU-only provenance and geometry contracts; no model or video replay."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import finalize_selective_video as render
from corridor_geometry import extract_contours, analyze_object_mask
from selective_refinement import select_additions


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def tile_record(key='t0042', *, origin='tile', score=.8):
    mask = np.zeros((100, 100), bool)
    mask[20:28, 30:38] = True
    mask[22:25, 32:35] = False  # Preserve a real hole, not a filled polygon.
    old = dict(inference_origin=origin, candidate_id=key,
               full_predicted_mask_contours=extract_contours(mask),
               full_mask_sha256=render.mask_sha(mask), model_source='litter',
               class_id=7, class_name='custom_litter', confidence_score=score,
               raw_detection_index=42, mask_bbox=[30, 20, 38, 28],
               model_bbox_original_px=[29.5, 19.5, 38.2, 28.2],
               inference_crop_xyxy=[10, 10, 90, 90], touches_crop_border=False,
               view_id='tile_0002', view_kind='tile', tile_index=2,
               observation_id='video:v2:f000012:o003')
    return old, mask


class TileCandidateTests(unittest.TestCase):
    def test_reuse_exact_mask_holes_provenance_and_ids(self):
        tile, mask = tile_record()
        baseline, _ = tile_record('b0001', origin='baseline')
        payload = {'objects': [baseline, tile], 'quarantined_predictions': [tile]}
        before = copy.deepcopy(payload)
        candidates = render.tile_candidates(payload, 100, 100)
        self.assertEqual(len(candidates), 1)
        obj = candidates[0]
        np.testing.assert_array_equal(obj['mask'], mask)
        self.assertEqual(obj['candidate_id'], 't0042')
        self.assertEqual(obj['candidate_observation_id'], tile['observation_id'])
        self.assertEqual(obj['candidate_mask_sha256'], tile['full_mask_sha256'])
        self.assertEqual(obj['class_id'], 7)
        self.assertEqual(obj['class_name'], 'custom_litter')
        self.assertEqual(obj['view_id'], 'tile_0002')
        self.assertEqual(payload, before)

    def test_rejects_changed_contour_or_mask_hash(self):
        tile, _ = tile_record()
        tile['full_mask_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            render.tile_candidates({'objects': [tile]}, 100, 100)

    def test_empty_and_global_only_pool(self):
        baseline, _ = tile_record(origin='baseline')
        self.assertEqual(render.tile_candidates({'objects': []}, 100, 100), [])
        self.assertEqual(render.tile_candidates({'objects': [baseline]}, 100, 100), [])

    def test_reconstructed_candidate_uses_same_selective_policy(self):
        tile, mask = tile_record()
        pool = render.tile_candidates({'objects': [tile]}, 100, 100)
        kept, audit = select_additions([], pool, .65, .01,
                                      image_width=100, image_height=100)
        self.assertEqual(len(kept), 1)
        self.assertTrue(audit[0]['accepted'])
        self.assertEqual(audit[0]['candidate_id'], 't0042')
        self.assertEqual(audit[0]['mask_area_pixels'], int(mask.sum()))
        self.assertFalse(select_additions([], pool, .9, .01)[0])

    def test_same_mask_as_baseline_is_not_added(self):
        tile, _ = tile_record()
        pool = render.tile_candidates({'objects': [tile]}, 100, 100)
        baseline = dict(pool[0], candidate_id='b0010', view_kind='global',
                        inference_origin='baseline')
        kept, audit = select_additions([baseline], pool, .5, .01)
        self.assertFalse(kept)
        self.assertEqual(audit[0]['baseline_ids'], ['b0010'])
        self.assertEqual(render.mask_sha(baseline['mask']), tile['full_mask_sha256'])

    def test_serialized_geometry_does_not_invent_precision_or_metres(self):
        tile, mask = tile_record()
        obj = render.tile_candidates({'objects': [tile]}, 100, 100)[0]
        support = np.ones(mask.shape, bool)
        analysis = analyze_object_mask(mask, support, support)
        render.decorate(obj, support, analysis, 'video:v2s:f000012:o000')
        record = render.serial_record(obj)
        for field in ['distance_m', 'actual_error_px', 'mask_quality_score',
                      'physical_danger_confirmed']:
            self.assertIsNone(record[field])
        self.assertEqual(record['confidence_score'], .8)
        self.assertEqual(record['full_mask_sha256'], tile['full_mask_sha256'])
        json.dumps(record, allow_nan=False)


class SelectedConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.path = self.folder/'SELECTION.json'
        protocol = {
            'name':'exploratory_selective_tiles_v1', 'split':'val', 'images':288,
            'reference_objects':871, 'tile_size':640, 'nominal_overlap':.25,
            'network_imgsz':960, 'operating_baseline_confidence':.15,
            'ap_baseline_confidence':.001, 'confidence_grid':[.35,.5,.65,.8],
            'area_fraction_grid':[.001,.002,.004], 'test_used':False,
            'first_stage':'merge_multiscale(raw globals>=.15, raw tiles>=.15), keep tile survivors, no first-stage max100 truncation',
            'baseline_policy':'preserve every baseline mask; no replacement, union or suppression, including all low-score AP globals>=.001'}
        code = []
        for name in ['small_object_refinement.py', 'selective_refinement.py']:
            path = Path(render.__file__).with_name(name)
            code.append(dict(path=str(path), sha256=render.sha(path), bytes=path.stat().st_size))
        configuration = dict(protocol=protocol, programs=code,
                             checkpoint_sha256='a'*64, dataset_sha256='b'*64)
        candidate = dict(grid_index=6, min_confidence=.65, max_area_fraction=.001,
                         eligible=True, eligibility_checks={'precision':True},
                         metrics={'AP_small':.12}, operating={'precision':.7})
        self.report = dict(status='complete', split='val', images=288,
                           reference_objects=871, selected_eligible=True,
                           selected_config={'min_confidence':.65, 'max_area_fraction':.001},
                           checkpoint_sha256='a'*64, dataset_sha256='b'*64,
                           selected_grid_index=6, candidates=[candidate],
                           selected=candidate['metrics'], selected_operating=candidate['operating'],
                           frozen_rule=copy.deepcopy(protocol), configuration_sha256=digest(configuration))
        self.manifest = dict(status='complete', split='val', image_count=288,
                             inputs_unchanged=True, test_images_or_annotations_opened=False,
                             model_inference_performed=False, configuration=configuration,
                             configuration_sha256=digest(configuration))

    def write(self, *, recalc_config=True):
        if recalc_config:
            self.manifest['configuration_sha256'] = digest(self.manifest['configuration'])
            self.report['configuration_sha256'] = self.manifest['configuration_sha256']
        self.path.write_text(json.dumps(self.report))
        self.manifest['selection'] = dict(path=str(self.path), sha256=render.sha(self.path),
                                          bytes=self.path.stat().st_size)
        (self.folder/'EVALUATION_MANIFEST.json').write_text(json.dumps(self.manifest))

    def test_valid_linked_report(self):
        self.write()
        cfg, report = render.selected_config(self.path)
        self.assertEqual(cfg, self.report['selected_config'])
        self.assertEqual(report, self.report)

    def test_modified_report_is_rejected_even_with_valid_grid_value(self):
        self.write()
        self.report['selected_config']['min_confidence'] = .8
        self.path.write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, 'differs from its evaluation'):
            render.selected_config(self.path)

    def test_unfinished_or_ineligible_validation_rejected(self):
        for key, value in [('status','running'), ('selected_eligible',False),
                           ('selected_eligible','true'), ('split','test'), ('images',287)]:
            old = self.report[key]
            self.report[key] = value
            self.write()
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                render.selected_config(self.path)
            self.report[key] = old

    def test_incompatible_protocol_rejected_despite_self_consistent_hash(self):
        self.manifest['configuration']['protocol']['tile_size'] = 512
        self.report['frozen_rule']['tile_size'] = 512
        self.write()
        with self.assertRaisesRegex(ValueError, 'incompatible first-stage'):
            render.selected_config(self.path)

    def test_changed_helper_rejected_without_touching_source_file(self):
        self.manifest['configuration']['programs'][0]['sha256'] = '0'*64
        self.write()
        with self.assertRaisesRegex(ValueError, 'helper differs'):
            render.selected_config(self.path)

    def test_duplicate_helper_evidence_rejected(self):
        self.manifest['configuration']['programs'].append(self.manifest['configuration']['programs'][0])
        self.write()
        with self.assertRaisesRegex(ValueError, 'helper differs'):
            render.selected_config(self.path)

    def test_configuration_digest_mismatch_rejected(self):
        self.manifest['configuration_sha256'] = '0'*64
        self.write(recalc_config=False)
        with self.assertRaisesRegex(ValueError, 'configuration evidence'):
            render.selected_config(self.path)

    def test_wrong_grid_index_or_rejected_selected_row(self):
        self.report['selected_grid_index'] = 7
        self.write()
        with self.assertRaises(ValueError):
            render.selected_config(self.path)
        self.report['selected_grid_index'] = 6
        self.report['candidates'][0]['eligible'] = False
        self.write()
        with self.assertRaises(ValueError):
            render.selected_config(self.path)

    def test_test_exposure_or_changed_inputs_rejected(self):
        for key, value in [('test_images_or_annotations_opened',True),
                           ('inputs_unchanged',False), ('model_inference_performed',True)]:
            original = self.manifest[key]
            self.manifest[key] = value
            self.write()
            with self.subTest(key=key), self.assertRaises(ValueError):
                render.selected_config(self.path)
            self.manifest[key] = original

    def enable_consensus(self):
        for rule in [self.manifest['configuration']['protocol'], self.report['frozen_rule']]:
            rule.update(name=render.CONSENSUS_PROTOCOL, confirmation_rule=render.CONFIRMATION_RULE)
        path=Path(render.__file__).with_name('multiview_confirmation.py')
        self.manifest['configuration']['programs'].append(
            dict(path=str(path),sha256=render.sha(path),bytes=path.stat().st_size))

    def test_explicit_consensus_protocol_and_helper_evidence_accepted(self):
        self.enable_consensus();self.write()
        cfg,report=render.selected_config(self.path)
        self.assertTrue(render.confirmation_mode(report))
        self.assertEqual(cfg,self.report['selected_config'])

    def test_consensus_missing_or_changed_helper_is_rejected(self):
        self.enable_consensus()
        proof=self.manifest['configuration']['programs'].pop()
        self.write()
        with self.assertRaisesRegex(ValueError,'helper differs'):
            render.selected_config(self.path)
        proof['sha256']='0'*64
        self.manifest['configuration']['programs'].append(proof);self.write()
        with self.assertRaisesRegex(ValueError,'helper differs'):
            render.selected_config(self.path)

    def test_consensus_rule_cannot_be_relaxed(self):
        self.enable_consensus()
        for rule in [self.manifest['configuration']['protocol'],self.report['frozen_rule']]:
            rule['confirmation_rule']=render.CONFIRMATION_RULE.replace('>.5','>=.5')
        self.write()
        with self.assertRaisesRegex(ValueError,'confirmation rule'):
            render.selected_config(self.path)

    def test_unmarked_confirmation_or_unknown_protocol_is_rejected(self):
        self.report['frozen_rule']['confirmation_rule']=render.CONFIRMATION_RULE
        self.manifest['configuration']['protocol']['confirmation_rule']=render.CONFIRMATION_RULE
        self.write()
        with self.assertRaisesRegex(ValueError,'contradictory selection protocol'):
            render.selected_config(self.path)
        self.report['frozen_rule']['name']='unregistered_experiment'
        self.manifest['configuration']['protocol']['name']='unregistered_experiment'
        self.write()
        with self.assertRaisesRegex(ValueError,'Unknown'):
            render.selected_config(self.path)

    def test_consensus_ineligible_report_stays_blocked(self):
        self.enable_consensus();self.report['selected_eligible']=False;self.write()
        with self.assertRaisesRegex(ValueError,'eligible selective validation'):
            render.selected_config(self.path)


class ConfirmationFrameTests(unittest.TestCase):
    def setUp(self):
        old,_=tile_record()
        self.pool=render.tile_candidates({'objects':[old]},100,100)
        self.row=dict(video_id='video',frame_index=12,time_video_s=.4,image_width=100,image_height=100,
            suppressed=[dict(rule='cross_view_mask_iou',mask_iou=.7,
                kept_candidate_id='t0042',kept_view_id='tile_0002',kept_raw_detection_index=42,
                suppressed_candidate_id='t0043',suppressed_view_id='tile_0003',suppressed_raw_detection_index=43)])

    def test_same_frame_audit_confirms_original_candidate(self):
        kept,audit=render.confirmed_pool_for_frame(self.pool,self.row,12,.4,100,100,'video')
        self.assertIs(kept[0],self.pool[0]);self.assertTrue(audit[0]['confirmed'])

    def test_adjacent_frame_or_wrong_time_cannot_supply_confirmation(self):
        for key,value in [('frame_index',13),('time_video_s',.5),('video_id','other'),('image_width',99)]:
            row=dict(self.row);row[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                render.confirmed_pool_for_frame(self.pool,row,12,.4,100,100,'video')

    def test_no_suppression_is_not_confirmation(self):
        row=dict(self.row,suppressed=[])
        kept,audit=render.confirmed_pool_for_frame(self.pool,row,12,.4,100,100,'video')
        self.assertFalse(kept);self.assertFalse(audit[0]['confirmed'])

    def test_missing_audit_fails_instead_of_silently_accepting_pool(self):
        for row in [None,{},dict(self.row,suppressed=None)]:
            with self.assertRaisesRegex(ValueError,'Missing candidate suppression audit'):
                render.confirmed_pool_for_frame(self.pool,row,12,.4,100,100,'video')


class ConfirmationRendererIntegrationTests(unittest.TestCase):
    def test_three_frame_cached_render_records_explicit_confirmation_and_counts(self):
        import test_verify_selective_video as fixture_module
        original=fixture_module.evidence_fixture
        def consensus_evidence(root,checkpoint='synthetic-model'):
            path,final=original(root,checkpoint)
            report=json.loads(path.read_text());manifest_path=path.parent/'EVALUATION_MANIFEST.json'
            manifest=json.loads(manifest_path.read_text())
            for rule in [report['frozen_rule'],manifest['configuration']['protocol']]:
                rule.update(name=render.CONSENSUS_PROTOCOL,confirmation_rule=render.CONFIRMATION_RULE)
            report['configuration_sha256']=digest(manifest['configuration'])
            manifest['configuration_sha256']=report['configuration_sha256']
            fixture_module.dump(path,report)
            manifest['selection']=fixture_module.record(path);manifest['output_files']=[fixture_module.record(path)]
            fixture_module.dump(manifest_path,manifest)
            final['selection_report_sha256']=render.sha(path)
            return path,final
        fixture=fixture_module.SelectiveIntegrationTests('test_real_renderer_fixture_passes_independent_verification')
        with patch.object(fixture_module,'evidence_fixture',consensus_evidence):
            fixture.setUp()
        self.addCleanup(fixture.tearDown)
        # The fixture bypasses only selection preflight, tested above. Here the
        # actual 3-frame renderer streams the independently verified V2 audit.
        manifest=json.loads((fixture.out/'MANIFESTE.json').read_text())
        self.assertEqual(manifest['method'],'multiview_confirmed_selective_additions_preserving_global_masks')
        self.assertTrue(manifest['multiview_confirmation_required'])
        self.assertEqual(manifest['selection_protocol'],render.CONSENSUS_PROTOCOL)
        self.assertEqual(manifest['tile_candidate_pool_count'],1)
        self.assertEqual(manifest['confirmation_survivors'],0)
        self.assertEqual(manifest['confirmation_rejections'],1)
        self.assertEqual(manifest['selective_additions'],0)
        self.assertEqual(manifest['quarantined_observations'],1)
        self.assertTrue(any(Path(p).name=='multiview_confirmation.py' for p in manifest['inputs_sha256']))
        audits=[json.loads(line) for line in (fixture.out/'AUDIT_FUSION.jsonl').read_text().splitlines()]
        self.assertEqual(len(audits[1]['tile_candidate_pool']),1)
        self.assertEqual(audits[1]['confirmed_tile_candidate_ids'],[])
        self.assertFalse(audits[1]['confirmation_decisions'][0]['confirmed'])
        self.assertEqual(audits[1]['selective_decisions'],[])


if __name__ == '__main__':
    unittest.main()
