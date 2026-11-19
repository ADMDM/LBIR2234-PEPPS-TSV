"""Small synthetic packaging tests; no real VAL cache, GPU, or model execution."""
import contextlib,hashlib,io,json,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import package_guard_video as pkg
import test_guard_only as fixture_module
from test_guard_only import dump,record,digest


_original_rejections=fixture_module.rejection_fixture


def package_rejections(root,checkpoint):
    paths=_original_rejections(root,checkpoint)
    op=dict(precision=.5,recall=.4,small_recall=.2,F2=.5,false_positive=4)
    for index,path in enumerate(paths):
        report=pkg.read(path);mp=path.parent/'EVALUATION_MANIFEST.json';manifest=pkg.read(mp)
        report.update(reference_objects=871,dataset_sha256='synthetic-data',baseline_operating=op)
        if index==0:
            report.update(tile_size=640,tiled_operating=dict(precision=.3,recall=.5,small_recall=.3,F2=.4,false_positive=10))
            manifest['configuration']['protocol']=dict(tile_size=640)
        else:
            manifest['configuration']['protocol']=dict(name=['','exploratory_selective_tiles_v1','exploratory_multiview_confirmation_v1'][index])
        manifest['configuration_sha256']=digest(manifest['configuration'])
        report['configuration_sha256']=manifest['configuration_sha256']
        dump(path,report)
        manifest['metrics' if index==0 else 'selection']=record(path)
        manifest['output_files']=[record(path)]
        if index:
            protocol=path.parent/'PROTOCOL_FROZEN.json'
            dump(protocol,dict(configuration=manifest['configuration'],configuration_sha256=manifest['configuration_sha256']))
            manifest['output_files'].append(record(protocol))
        dump(mp,manifest)
    return paths


class GuardPackageTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixture_module.FallbackTests('test_actual_renderer_full_proof_and_historical_tracking')
        with patch.object(fixture_module,'rejection_fixture',package_rejections):self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root,self.run=self.fixture.root,self.fixture.out
        fixture_module.v.verify_delivery(self.run)
        self.notice=self.root/'notice.txt';self.notice.write_text('Synthetic notice')
        self.code=self.root/'code';self.code.mkdir()
        for name in ('lire_contours.py','corridor_geometry.py'):(self.code/name).write_text('# Synthetic reader\n')
        self.out=self.root/'delivery.zip'
        self.args=['--run',str(self.run),'--notice',str(self.notice),'--code',str(self.code),'--out',str(self.out)]

    def invoke(self):
        with contextlib.redirect_stdout(io.StringIO()):pkg.main(self.args)

    def no_publication(self):
        self.assertFalse(self.out.exists());self.assertFalse(self.out.with_suffix('.verification.json').exists())
        self.assertFalse((self.run/'STATISTIQUES.json').exists());self.assertFalse((self.run/'RESULTATS_OBTENUS.txt').exists())

    def test_complete_zip_hashes_no_added_masks_and_three_rejection_reports(self):
        self.invoke();receipt=pkg.read(self.out.with_suffix('.verification.json'));stats=pkg.read(self.run/'STATISTIQUES.json')
        self.assertEqual(receipt['sha256'],pkg.sha(self.out));self.assertFalse(receipt['small_object_gain_validated'])
        self.assertEqual(stats['baseline_observations'],2)
        self.assertEqual((stats['accepted_observations'],stats['quarantined_observations'],stats['selective_additions']),(1,1,0))
        self.assertTrue(stats['historical_ids_and_tracking_preserved']);self.assertEqual(stats['rejected_configurations'],25)
        self.assertNotIn('accepted_by_origin',stats)
        with zipfile.ZipFile(self.out) as archive:
            names=archive.namelist();self.assertEqual(len(names),len(set(names)))
            prefix='GABARIT_2D_V2_CORRIGEE/'
            self.assertTrue(all(n.startswith(prefix) and '..' not in Path(n).parts for n in names))
            manifest=json.loads(archive.read(prefix+'MANIFESTE.json'))
            for name,h in manifest['output_sha256'].items():self.assertEqual(hashlib.sha256(archive.read(prefix+name)).hexdigest(),h)
            self.assertIn(prefix+'VALIDATION/01_TUILES/METRICS.json',names)
            self.assertIn(prefix+'VALIDATION/02_SELECTION/SELECTION.json',names)
            self.assertIn(prefix+'VALIDATION/03_CONFIRMATION/SELECTION.json',names)
            self.assertIn(prefix+'VALIDATION/03_CONFIRMATION/PROTOCOL_FROZEN.json',names)
            self.assertFalse(any(n.endswith('.pt') for n in names))
            note=archive.read(prefix+'RESULTATS_OBTENUS.txt').decode()
            self.assertIn('PAS DE GAIN VALIDÉ',note);self.assertIn('Ajouts : 0',note)

    def test_existing_archive_or_note_or_receipt_not_overwritten(self):
        for target in [self.out,self.run/'RESULTATS_OBTENUS.txt',self.out.with_suffix('.verification.json')]:
            target.write_text('user file')
            with self.subTest(path=target),self.assertRaises(FileExistsError):self.invoke()
            self.assertEqual(target.read_text(),'user file');target.unlink()
            self.assertFalse((self.run/'STATISTIQUES.json').exists())

    def test_changed_verified_csv_is_refused(self):
        with (self.run/'OBJETS.csv').open('a') as handle:handle.write('\n')
        with self.assertRaisesRegex(ValueError,'Changed verified output'):self.invoke()
        self.no_publication()

    def test_changed_txt_is_refused(self):
        path=self.run/'TXT/frame_000000.txt';path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'TXT contents changed'):self.invoke()
        self.no_publication()

    def test_exact_v1_proof_required(self):
        path=self.run/'VERIFICATION.json';proof=pkg.read(path);proof['historical_ids_and_tracking_preserved']=False;dump(path,proof)
        with self.assertRaisesRegex(ValueError,'Exact V1 verification missing'):self.invoke()
        self.no_publication()

    def test_claimed_small_gain_is_refused(self):
        path=self.run/'MANIFESTE.json';manifest=pkg.read(path);manifest['small_object_gain_validated']=True;dump(path,manifest)
        with self.assertRaisesRegex(ValueError,'without new masks or small-object gain'):self.invoke()
        self.no_publication()

    def test_three_reports_are_required_not_two(self):
        checkpoint=pkg.read(self.run/'MANIFESTE.json')['checkpoint_sha256']
        with self.assertRaisesRegex(ValueError,'Exactly three'):
            pkg.validation_files_and_summary(self.fixture.reports[:2],checkpoint)

    def test_rejected_report_cannot_be_changed_after_render(self):
        path=self.fixture.reports[-1];path.write_text(path.read_text()+'\n')
        with self.assertRaises(ValueError):self.invoke()
        self.no_publication()

    def test_input_changed_after_archive_copy_is_refused(self):
        original=zipfile.ZipFile.write
        def mutate(z,path,name,*args,**kwargs):
            result=original(z,path,name,*args,**kwargs)
            if name.endswith('/NOTICE_EQUIPE_GABARIT.txt'):self.notice.write_text('changed after copy')
            return result
        with patch.object(zipfile.ZipFile,'write',mutate):
            with self.assertRaisesRegex(ValueError,'Input changed during packaging'):self.invoke()
        self.no_publication()

    def test_disk_failure_rolls_back_created_files(self):
        with patch.object(pkg.os,'link',side_effect=OSError('synthetic failure')):
            with self.assertRaisesRegex(OSError,'synthetic failure'):self.invoke()
        self.no_publication();self.assertTrue(self.notice.exists())

    def test_concurrent_archive_is_not_deleted_by_rollback(self):
        original=pkg.os.link
        def race(source,destination):
            Path(destination).write_bytes(b'concurrent user file');return original(source,destination)
        with patch.object(pkg.os,'link',side_effect=race):
            with self.assertRaises(FileExistsError):self.invoke()
        self.assertEqual(self.out.read_bytes(),b'concurrent user file')
        self.assertFalse((self.run/'STATISTIQUES.json').exists())
        self.assertFalse(self.out.with_suffix('.verification.json').exists())


if __name__=='__main__':unittest.main()
