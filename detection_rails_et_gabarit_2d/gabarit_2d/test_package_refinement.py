"""CPU synthetic packaging and tamper tests; never load an ML model."""
import copy,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import package_refinement as pkg
import verify_refinement as verifier
from test_verify_refinement import make_v2,dump,read_rows,table


def proof(path):
    return dict(path=str(path.resolve()),sha256=pkg.sha(path),bytes=path.stat().st_size)


def make_validation(root,model):
    folder=root/'val';folder.mkdir()
    inputs=[]
    for name in ['program.py','helper.py','baseline.json','manifest_val.json','COCO_ORIGINAL_val.json']:
        path=folder/name;path.write_text(name);inputs.append(proof(path))
    inputs.append(proof(model));key=pkg.sha(model)
    rule=dict(split='val',first_tile_size=640,fallback_tile_size=512,overlap=.25,inference_imgsz=960,
              confidence_floor=.001,operating_confidence=.15,max_detections_per_image=100,test_data_used_for_selection=False,
              eligibility='AP_small strictly improves and AP50_95 >= baseline AP50_95 - 0.01')
    config=dict(protocol=dict(rule,tile_size=640),model_sha256=key,dataset_sha256='synthetic-dataset')
    for field,record in zip(['program_sha256','helper_sha256','baseline_predictions_sha256','validation_manifest_sha256','validation_coco_sha256'],inputs):config[field]=record['sha256']
    metrics=dict(status='complete',split='val',images=288,reference_objects=1,checkpoint_sha256=key,
                 dataset_sha256='synthetic-dataset',configuration_sha256=pkg.json_digest(config),selection_rule=rule,tile_size=640,
                 baseline=dict(AP_small=.1,AP50_95=.2),tiled=dict(AP_small=.15,AP50_95=.21),eligible_by_frozen_rule=True,
                 baseline_operating=dict(confidence=.15,mask_iou=.5),tiled_operating=dict(confidence=.15,mask_iou=.5))
    for name in pkg.VALIDATION_OUTPUTS:
        dump(folder/name,metrics if name=='METRICS.json' else dict(images=[dict(id=i) for i in range(288)],annotations=[dict(id=1)]) if name=='GROUND_TRUTH_PREPARED.json' else [])
    manifest=dict(status='complete',split='val',inputs_unchanged=True,test_images_or_annotations_opened=False,
                  image_count=288,source_image_count=288,checkpoint_sha256=key,dataset_sha256='synthetic-dataset',
                  configuration=config,configuration_sha256=pkg.json_digest(config),selection_rule_frozen_before_metrics=rule,
                  metrics=proof(folder/'METRICS.json'),output_files=[proof(folder/n) for n in sorted(pkg.VALIDATION_OUTPUTS)],inputs=inputs)
    dump(folder/'EVALUATION_MANIFEST.json',manifest)
    return folder/'METRICS.json'


class PackageRefinementTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.run,self.baseline=make_v2(self.root)
        m=pkg.read_json(self.run/'MANIFESTE.json');m['elapsed_s']=2.;m['parameters'].update(imgsz=960,tile_size=640)
        # Adapt the tiny fixture to the actual selected window recipe, then run
        # the real independent verifier. No invented verifier result is used.
        rails=read_rows(self.run/'RAILS.csv');frames=read_rows(self.run/'FRAMES.csv')
        audits=[json.loads(line) for line in (self.run/'AUDIT_FUSION.jsonl').read_text().splitlines()]
        for index in range(3):
            c,_,_=verifier.independent_corridor([r for r in rails if int(r['frame_index'])==index],32,24)
            views=verifier.expected_tile_views(c,640,.25);audits[index]['tile_views']=views;frames[index]['tile_views']=len(views)
            path=self.run/f'TXT/frame_{index:06d}.txt';data=pkg.read_json(path)
            for obj in data['objects']:
                if obj['inference_origin']=='tile':obj['inference_crop_xyxy']=views[obj['tile_index']]['crop_xyxy']
            dump(path,data)
        table(self.run/'FRAMES.csv',list(frames[0]),frames)
        (self.run/'AUDIT_FUSION.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in audits))
        m['output_sha256']={n:pkg.sha(self.run/n) for n in m['output_sha256']};dump(self.run/'MANIFESTE.json',m)
        verifier.verify_delivery(self.run)
        self.model=Path(m['parameters']['litter_model']);self.metrics=make_validation(self.root,self.model)
        self.pin=patch.object(pkg,'CHECKPOINT_SHA',pkg.sha(self.model));self.pin.start();self.addCleanup(self.pin.stop)
        self.notice=self.root/'notice.txt';self.notice.write_text('Synthetic notice')
        self.code=self.root/'reader';self.code.mkdir()
        for n in ['lire_contours.py','corridor_geometry.py']:(self.code/n).write_text('# synthetic reader\n')
        self.out=self.root/'delivery.zip'
        self.args=['--run',str(self.run),'--validation-report',str(self.metrics),'--notice',str(self.notice),'--code',str(self.code),'--out',str(self.out)]
    def tearDown(self):self.tmp.cleanup()
    def rewrite_validation(self,mutate_metrics=None,mutate_manifest=None):
        ev=pkg.read_json(self.metrics);em=pkg.read_json(self.metrics.parent/'EVALUATION_MANIFEST.json')
        if mutate_metrics:mutate_metrics(ev)
        dump(self.metrics,ev)
        em['metrics']=proof(self.metrics)
        em['output_files']=[proof(self.metrics.parent/Path(x['path']).name) for x in em['output_files']]
        if mutate_manifest:mutate_manifest(em)
        dump(self.metrics.parent/'EVALUATION_MANIFEST.json',em)
    def no_publication(self):
        self.assertFalse(self.out.exists());self.assertFalse(self.out.with_suffix('.verification.json').exists())
        self.assertFalse((self.run/'STATISTIQUES.json').exists());self.assertFalse((self.run/'RESULTATS_OBTENUS.txt').exists())
    def test_valid_archive_hashes_safe_names_and_minimal_content(self):
        pkg.main(self.args)
        receipt=pkg.read_json(self.out.with_suffix('.verification.json'));self.assertTrue(receipt['validation_eligible']);self.assertEqual(receipt['validation_images'],288)
        self.assertEqual(receipt['sha256'],pkg.sha(self.out))
        with zipfile.ZipFile(self.out) as z:
            names=z.namelist();self.assertEqual(len(names),len(set(names)))
            self.assertTrue(all(n.startswith('GABARIT_2D_V2/') and '..' not in Path(n).parts and not Path(n).is_absolute() for n in names))
            self.assertIn('GABARIT_2D_V2/A_VERIFIER.csv',names);self.assertIn('GABARIT_2D_V2/PROVENANCE_VALIDATION.json',names)
            self.assertFalse(any(n.endswith('.pt') for n in names))
    def test_changed_metrics_rejected(self):
        ev=pkg.read_json(self.metrics);ev['reference_objects']=2;dump(self.metrics,ev)
        with self.assertRaisesRegex(ValueError,'Changed validation output'):pkg.main(self.args)
        self.no_publication()
    def test_wrong_checkpoint_rejected(self):
        self.rewrite_validation(lambda e:e.update(checkpoint_sha256='0'*64))
        with self.assertRaisesRegex(ValueError,'same selected best checkpoint'):pkg.main(self.args)
    def test_incomplete_image_count_rejected(self):
        self.rewrite_validation(lambda e:e.update(images=287))
        with self.assertRaisesRegex(ValueError,'288'):pkg.main(self.args)
    def test_test_split_is_forbidden(self):
        self.rewrite_validation(lambda e:e.update(split='test'))
        with self.assertRaisesRegex(ValueError,'Only VAL'):pkg.main(self.args)
    def test_claimed_eligible_but_scores_fail_rejected(self):
        self.rewrite_validation(lambda e:e['tiled'].update(AP_small=.09))
        with self.assertRaisesRegex(ValueError,'ineligible'):pkg.main(self.args)
    def test_material_ap_regression_rejected(self):
        self.rewrite_validation(lambda e:e['tiled'].update(AP50_95=.18))
        with self.assertRaisesRegex(ValueError,'ineligible'):pkg.main(self.args)
    def test_video_tile_recipe_must_match_validation(self):
        self.rewrite_validation(lambda e:e.update(tile_size=512))
        with self.assertRaisesRegex(ValueError,'tile sizes differ'):pkg.main(self.args)
    def test_operating_confidence_mismatch_rejected(self):
        self.rewrite_validation(lambda e:e['tiled_operating'].update(confidence=.2))
        with self.assertRaisesRegex(ValueError,'score threshold'):pkg.main(self.args)
    def test_configuration_tamper_rejected(self):
        self.rewrite_validation(mutate_manifest=lambda e:e['configuration']['protocol'].update(overlap=.5))
        with self.assertRaisesRegex(ValueError,'configuration hash'):pkg.main(self.args)
    def test_validation_input_changed_rejected(self):
        (self.metrics.parent/'helper.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'Changed validation input'):pkg.main(self.args)
    def test_no_overwrite_existing_zip(self):
        self.out.write_bytes(b'existing user zip')
        with self.assertRaises(FileExistsError):pkg.main(self.args)
        self.assertEqual(self.out.read_bytes(),b'existing user zip')
        self.assertFalse((self.run/'STATISTIQUES.json').exists())
    def test_no_overwrite_existing_result_note(self):
        p=self.run/'RESULTATS_OBTENUS.txt';p.write_text('existing user note')
        with self.assertRaises(FileExistsError):pkg.main(self.args)
        self.assertEqual(p.read_text(),'existing user note');self.assertFalse(self.out.exists())
    def test_publication_failure_rolls_back_only_new_files(self):
        with patch.object(pkg.os,'link',side_effect=OSError('synthetic disk error')):
            with self.assertRaisesRegex(OSError,'synthetic disk'):pkg.main(self.args)
        self.no_publication();self.assertTrue(self.notice.exists())
    def test_input_mutated_after_zip_copy_rejected(self):
        original=zipfile.ZipFile.write
        def mutating_write(z,file,arcname,*args,**kwargs):
            result=original(z,file,arcname,*args,**kwargs)
            if arcname.endswith('/NOTICE_EQUIPE_GABARIT.txt'):self.notice.write_text('changed after copying')
            return result
        with patch.object(zipfile.ZipFile,'write',mutating_write):
            with self.assertRaisesRegex(ValueError,'Input changed during packaging'):pkg.main(self.args)
        self.no_publication()

if __name__=='__main__':unittest.main()
