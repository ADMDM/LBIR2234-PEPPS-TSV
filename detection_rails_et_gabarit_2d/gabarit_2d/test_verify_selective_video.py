"""Independent rule and frozen-evidence adversarial tests; no network/model."""
import copy,json,tempfile,unittest,sys,contextlib,io
from unittest.mock import patch
from pathlib import Path
import numpy as np
import verify_selective_video as v
import selection_evidence as se
from selective_refinement import select_additions

def dump(p,data):p.write_text(json.dumps(data,indent=2)+'\n')
def record(p):return dict(path=str(p.resolve()),sha256=se.sha(p),bytes=p.stat().st_size)

def obj(mask,ident,score=.8,source='litter',crop=None):
    yy,xx=np.nonzero(mask)
    return dict(candidate_id=ident,model_source=source,class_id=0,class_name='dechet',confidence_score=score,mask=mask,
                mask_bbox=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1],inference_origin='tile',view_kind='tile',
                view_id='tile_0000',raw_detection_index=int(ident[1:]),inference_crop_xyxy=crop,touches_crop_border=False)

def evidence_fixture(root,checkpoint='synthetic-model'):
    old=root/'old';new=root/'selection';old.mkdir();new.mkdir()
    baseline=dict(AP_small=.1,AP50_95=.3);operating=dict(precision=.5,small_recall=.2,F2=.5)
    previous=dict(status='complete',split='val',images=288,checkpoint_sha256=checkpoint,dataset_sha256='synthetic-data',baseline=baseline,baseline_operating=operating,tiled=dict(AP_small=.12,AP50_95=.28),eligible_by_frozen_rule=False)
    dump(old/'METRICS.json',previous)
    om=dict(status='complete',split='val',image_count=288,test_images_or_annotations_opened=False,inputs_unchanged=True,checkpoint_sha256=checkpoint,dataset_sha256='synthetic-data',inputs=[],output_files=[record(old/'METRICS.json')])
    dump(old/'EVALUATION_MANIFEST.json',om)
    rule=dict(name='exploratory_selective_tiles_v1',confidence_grid=[.35,.5,.65,.8],area_fraction_grid=[.001,.002,.004],operating_baseline_confidence=.15,ap_baseline_confidence=.001,network_imgsz=960,tile_size=640,nominal_overlap=.25,test_used=False)
    config=dict(protocol=rule,checkpoint_sha256=checkpoint,dataset_sha256='synthetic-data')
    candidates=[]
    for i,(c,a) in enumerate((c,a) for c in rule['confidence_grid'] for a in rule['area_fraction_grid']):
        metrics=dict(AP_small=.12 if i in (5,6) else .1,AP50_95=.3)
        op=dict(precision=.49 if i in (5,6) else .5,small_recall=.25 if i in (5,6) else .2,F2=.52 if i in (5,6) else .5)
        checks=dict(precision=True,small_recall=i in (5,6),F2=True,AP_small=i in (5,6),AP50_95=True)
        candidates.append(dict(grid_index=i,min_confidence=c,max_area_fraction=a,metrics=metrics,operating=op,eligible=i in (5,6),eligibility_checks=checks))
    winner=candidates[5];cfg={k:winner[k] for k in ['min_confidence','max_area_fraction']}
    s=dict(status='complete',split='val',images=288,reference_objects=871,checkpoint_sha256=checkpoint,dataset_sha256='synthetic-data',configuration_sha256=se.digest(config),frozen_rule=rule,
           selected_eligible=True,candidates=candidates,baseline=baseline,baseline_operating=operating,selected_config=cfg,selected_grid_index=5,selected=winner['metrics'],selected_operating=winner['operating'])
    dump(new/'SELECTION.json',s)
    m=dict(status='complete',split='val',image_count=288,source_image_count=288,test_images_or_annotations_opened=False,model_inference_performed=False,inputs_unchanged=True,selection=record(new/'SELECTION.json'),configuration=config,configuration_sha256=se.digest(config),inputs=[record(old/'EVALUATION_MANIFEST.json')],output_files=[record(new/'SELECTION.json')],prior_validation_manifest=record(old/'EVALUATION_MANIFEST.json'))
    dump(new/'EVALUATION_MANIFEST.json',m)
    final=dict(selection_report_sha256=se.sha(new/'SELECTION.json'),selection_config=cfg,parameters=cfg.copy())
    return new/'SELECTION.json',final

class IndependentRuleTests(unittest.TestCase):
    def compare(self,baseline,pool,cfg):
        w=h=100
        actual,audit=v.independent_select(baseline,pool,cfg,w,h)
        expected,reference=select_additions(baseline,pool,**cfg,image_width=w,image_height=h)
        self.assertEqual([o['candidate_id'] for o in actual],[o['candidate_id'] for o in expected]);self.assertEqual(audit,reference)
        return actual,audit
    def test_addition_and_exact_threshold_equalities(self):
        mask=np.zeros((100,100),bool);mask[40:45,40:44]=1
        kept,a=self.compare([], [obj(mask,'t0000',.5)],dict(min_confidence=.5,max_area_fraction=.002))
        self.assertEqual(len(kept),1);self.assertEqual(a[0]['reason'],'accepted_selective_addition')
    def test_low_score_large_and_border_are_all_audited(self):
        mask=np.zeros((100,100),bool);mask[0:10,20:30]=1
        kept,a=self.compare([], [obj(mask,'t0000',.3)],dict(min_confidence=.5,max_area_fraction=.002))
        self.assertFalse(kept);self.assertEqual(a[0]['rejection_reasons'],['below_min_confidence','exceeds_max_area_fraction','touches_image_border'])
    def test_crop_contact_is_recomputed(self):
        mask=np.zeros((100,100),bool);mask[20:22,20:22]=1;o=obj(mask,'t0000',crop=[20,20,60,60]);o['touches_crop_border']=True
        kept,a=self.compare([], [o],dict(min_confidence=.5,max_area_fraction=.002));self.assertFalse(kept);self.assertIn('touches_crop_border',a[0]['rejection_reasons'])
    def test_overlap_with_other_baseline_class_rejects(self):
        mask=np.zeros((100,100),bool);mask[40:44,40:44]=1;b=obj(mask.copy(),'b0000',source='public');b['class_id']=99
        kept,a=self.compare([b],[obj(mask,'t0000')],dict(min_confidence=.5,max_area_fraction=.002))
        self.assertFalse(kept);self.assertEqual(a[0]['baseline_ids'],['b0000'])
    def test_two_additions_conflict_and_ties_use_id(self):
        mask=np.zeros((100,100),bool);mask[40:44,40:44]=1
        kept,a=self.compare([], [obj(mask.copy(),'t0001'),obj(mask.copy(),'t0000')],dict(min_confidence=.5,max_area_fraction=.002))
        self.assertEqual([x['candidate_id'] for x in kept],['t0000']);self.assertEqual(a[1]['rejection_reasons'],['overlap_with_selected_addition'])

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.path,self.final=evidence_fixture(self.root)
    def tearDown(self):self.tmp.cleanup()
    def rewrite(self,change):
        s=se.read(self.path);change(s);dump(self.path,s);m=se.read(self.path.parent/'EVALUATION_MANIFEST.json');m['selection']=record(self.path);m['output_files']=[record(self.path)];dump(self.path.parent/'EVALUATION_MANIFEST.json',m);self.final['selection_report_sha256']=se.sha(self.path)
    def test_complete_chain_and_stable_tie_winner(self):
        s,frozen=se.verify_selection_evidence(self.path,self.final);self.assertEqual(s['selected_grid_index'],5);self.assertEqual(len(frozen),4)
    def test_wrong_winner_even_with_updated_hashes_rejected(self):
        self.rewrite(lambda s:s.update(selected_grid_index=6))
        with self.assertRaisesRegex(ValueError,'Wrong selected winner'):se.verify_selection_evidence(self.path,self.final)
    def test_forged_eligibility_rejected(self):
        self.rewrite(lambda s:s['candidates'][0].update(eligible=True))
        with self.assertRaisesRegex(ValueError,'Eligibility not supported'):se.verify_selection_evidence(self.path,self.final)
    def test_original_report_tamper_rejected(self):
        p=self.root/'old/METRICS.json';d=se.read(p);d['tiled']['AP_small']=.11;dump(p,d)
        with self.assertRaisesRegex(ValueError,'Changed original VAL evidence'):se.verify_selection_evidence(self.path,self.final)
    def test_test_split_forbidden(self):
        self.rewrite(lambda s:s.update(split='test'))
        with self.assertRaisesRegex(ValueError,'Completed VAL'):se.verify_selection_evidence(self.path,self.final)
    def test_video_thresholds_must_match(self):
        self.final['parameters']['min_confidence']=.8
        with self.assertRaisesRegex(ValueError,'thresholds differ'):se.verify_selection_evidence(self.path,self.final)

class SelectiveIntegrationTests(unittest.TestCase):
    def setUp(self):
        from test_verify_refinement import make_v2,read_rows,table
        import verify_refinement as v2
        import finalize_selective_video as renderer
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.candidates,self.baseline=make_v2(self.root)
        m=v.read(self.candidates/'MANIFESTE.json');m['parameters'].update(imgsz=960,tile_size=640)
        rows=read_rows(self.candidates/'RAILS.csv');frames=read_rows(self.candidates/'FRAMES.csv')
        audits=[json.loads(line) for line in (self.candidates/'AUDIT_FUSION.jsonl').read_text().splitlines()]
        for i in range(3):
            c,_,_=v2.independent_corridor([r for r in rows if int(r['frame_index'])==i],32,24);views=v2.expected_tile_views(c,640,.25)
            audits[i]['tile_views']=views;frames[i]['tile_views']=len(views)
            p=self.candidates/f'TXT/frame_{i:06d}.txt';d=v.read(p)
            for o in d['objects']:
                if o['inference_origin']=='tile':o['inference_crop_xyxy']=views[o['tile_index']]['crop_xyxy']
            dump(p,d)
        table(self.candidates/'FRAMES.csv',list(frames[0]),frames)
        (self.candidates/'AUDIT_FUSION.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in audits))
        m['guard_rule']='class=train AND confidence<0.5 AND corridor_coverage>=0.9 AND height_fraction>=0.75 AND bottom_fraction>=0.95';m['quarantine_storage']='synthetic full contours retained'
        m['output_sha256']={n:v.sha(self.candidates/n) for n in m['output_sha256']};dump(self.candidates/'MANIFESTE.json',m);v2.verify_delivery(self.candidates)
        self.selection,_=evidence_fixture(self.root,checkpoint=v.sha(Path(m['parameters']['litter_model'])));s=v.read(self.selection)
        self.out=self.root/'final'
        argv=['finalize_selective_video.py','--baseline',str(self.baseline),'--candidates',str(self.candidates),'--selection-report',str(self.selection),'--out',str(self.out)]
        # The renderer selection preflight has its own test suite. Isolate it
        # here; the independent verifier still checks all synthetic evidence.
        with patch.object(sys,'argv',argv),patch.object(renderer,'selected_config',return_value=(s['selected_config'],s)),contextlib.redirect_stdout(io.StringIO()):renderer.main()
    def tearDown(self):self.tmp.cleanup()
    def update_output_hashes(self):
        m=v.read(self.out/'MANIFESTE.json');m['output_sha256']={n:v.sha(self.out/n) for n in m['output_sha256']};dump(self.out/'MANIFESTE.json',m)
    def test_real_renderer_fixture_passes_independent_verification(self):
        result=v.verify_delivery(self.out)
        self.assertEqual((result['frames_checked'],result['objects_checked'],result['quarantined_objects_checked']), (3,1,1));self.assertTrue(result['baseline_preservation_verified'])
    def test_silently_removing_a_baseline_object_rejected(self):
        p=self.out/'TXT/frame_000000.txt';d=v.read(p);d['objects']=[];dump(p,d)
        with self.assertRaisesRegex(ValueError,'Baseline lost'):v.verify_delivery(self.out)
    def test_forged_selective_audit_rejected(self):
        p=self.out/'AUDIT_FUSION.jsonl';rows=[json.loads(line) for line in p.read_text().splitlines()];rows[1]['selective_decisions'][0]['accepted']=True;p.write_text(''.join(json.dumps(row)+'\n' for row in rows));self.update_output_hashes()
        with self.assertRaisesRegex(ValueError,'Selective rule audit differs'):v.verify_delivery(self.out)
    def test_quarantine_cannot_disappear_rejected(self):
        p=self.out/'TXT/frame_000002.txt';d=v.read(p);d['quarantined_predictions']=[];dump(p,d)
        with self.assertRaisesRegex(ValueError,'Wrong baseline quarantine set'):v.verify_delivery(self.out)

if __name__=='__main__':unittest.main()
