"""Synthetic three-frame packaging checks with real independent verification.

No detector is loaded and no production output is touched. The existing tiny
renderer fixture isolates renderer preflight, which has its own dedicated tests;
the selection-evidence and final-output verifiers run normally here.
"""
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

import package_selective_video as pkg
import test_verify_selective_video as fixture_module
from test_verify_selective_video import dump, record


_original_evidence_fixture = fixture_module.evidence_fixture


def complete_package_evidence(root, checkpoint='synthetic-model'):
    path, final = _original_evidence_fixture(root, checkpoint)
    selection = json.loads(path.read_text())
    selection['baseline_operating']['recall'] = .4
    for candidate in selection['candidates']:
        candidate['operating']['recall'] = .45
    selection['selected_operating'] = selection['candidates'][selection['selected_grid_index']]['operating']
    old_report = root/'old/METRICS.json'
    old = json.loads(old_report.read_text())
    old['baseline_operating'] = selection['baseline_operating']
    dump(old_report, old)
    old_manifest = root/'old/EVALUATION_MANIFEST.json'
    om = json.loads(old_manifest.read_text())
    om['output_files'] = [record(old_report)]
    dump(old_manifest, om)
    dump(path, selection)
    manifest_path = path.parent/'EVALUATION_MANIFEST.json'
    em = json.loads(manifest_path.read_text())
    protocol_path = path.parent/'PROTOCOL_FROZEN.json'
    dump(protocol_path, dict(configuration=em['configuration'],
                             configuration_sha256=em['configuration_sha256']))
    em['selection'] = record(path)
    em['output_files'] = [record(path), record(protocol_path)]
    em['inputs'] = [record(old_manifest)]
    em['prior_validation_manifest'] = record(old_manifest)
    dump(manifest_path, em)
    final['selection_report_sha256'] = pkg.sha(path)
    return path, final


class SelectivePackageTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.SelectiveIntegrationTests(
            'test_real_renderer_fixture_passes_independent_verification')
        with patch.object(fixture_module, 'evidence_fixture', complete_package_evidence):
            self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root, self.run = self.fixture.root, self.fixture.out
        fixture_module.v.verify_delivery(self.run)
        self.selection = self.fixture.selection
        self.notice = self.root/'notice.txt'; self.notice.write_text('Synthetic notice')
        self.code = self.root/'reader'; self.code.mkdir()
        for name in ['lire_contours.py', 'corridor_geometry.py']:
            (self.code/name).write_text('# Synthetic reader\n')
        self.out = self.root/'delivery.zip'
        self.args = ['--run', str(self.run), '--selection-report', str(self.selection),
                     '--notice', str(self.notice), '--code', str(self.code), '--out', str(self.out)]

    def invoke(self):
        with contextlib.redirect_stdout(io.StringIO()):
            pkg.main(self.args)

    def assert_unpublished(self):
        self.assertFalse(self.out.exists())
        self.assertFalse(self.out.with_suffix('.verification.json').exists())
        for name in ['STATISTIQUES.json', 'RESULTATS_OBTENUS.txt']:
            self.assertFalse((self.run/name).exists())

    def test_valid_archive_has_full_manifest_set_and_preservation_stats(self):
        self.invoke()
        receipt = json.loads(self.out.with_suffix('.verification.json').read_text())
        self.assertEqual(receipt['sha256'], pkg.sha(self.out))
        self.assertTrue(receipt['all_entries_sha256_verified'])
        self.assertEqual(receipt['txt_files_checked'], 3)
        stats = json.loads((self.run/'STATISTIQUES.json').read_text())
        self.assertEqual(stats['baseline_observations'], 2)
        self.assertEqual(stats['accepted_by_origin'], {'baseline': 1})
        self.assertEqual(stats['quarantined_observations'], 1)
        self.assertEqual(stats['selective_additions'], 0)
        self.assertFalse(stats['TSV_accuracy_measured'])
        self.assertFalse(stats['new_training_performed'])
        self.assertEqual(stats['accepted_observations'] + stats['quarantined_observations'],
                         stats['baseline_observations'] + stats['selective_additions'])
        with zipfile.ZipFile(self.out) as archive:
            names = archive.namelist()
            self.assertEqual(len(names), len(set(names)))
            self.assertTrue(all(n.startswith('GABARIT_2D_V2/') and '..' not in Path(n).parts for n in names))
            manifest = json.loads(archive.read('GABARIT_2D_V2/MANIFESTE.json'))
            for name, digest in manifest['output_sha256'].items():
                import hashlib
                self.assertEqual(hashlib.sha256(archive.read('GABARIT_2D_V2/'+name)).hexdigest(), digest)
            self.assertIn('GABARIT_2D_V2/AUDIT_FUSION.jsonl', names)
            self.assertIn('GABARIT_2D_V2/PROTOCOLE_VALIDATION.json', names)
            self.assertFalse(any(n.endswith('.pt') for n in names))

    def test_existing_zip_is_not_overwritten(self):
        self.out.write_bytes(b'user archive')
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual(self.out.read_bytes(), b'user archive')
        self.assertFalse((self.run/'STATISTIQUES.json').exists())

    def test_existing_note_and_receipt_are_not_overwritten(self):
        for target in [self.run/'RESULTATS_OBTENUS.txt', self.out.with_suffix('.verification.json')]:
            target.write_text('user file')
            with self.subTest(path=target), self.assertRaises(FileExistsError):
                self.invoke()
            self.assertEqual(target.read_text(), 'user file')
            self.assertFalse(self.out.exists())
            target.unlink()

    def test_changed_output_after_verification_is_refused(self):
        with (self.run/'OBJETS.csv').open('a') as f:
            f.write('\n')
        with self.assertRaisesRegex(ValueError, 'Changed output'):
            self.invoke()
        self.assert_unpublished()

    def test_changed_txt_after_verification_is_refused(self):
        path = self.run/'TXT/frame_000000.txt'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'TXT contents changed'):
            self.invoke()
        self.assert_unpublished()

    def test_schema_two_is_not_published_as_selected_final(self):
        path = self.run/'MANIFESTE.json'; m=json.loads(path.read_text())
        m['schema_version'] = 2; dump(path,m)
        with self.assertRaisesRegex(ValueError, 'selective result required'):
            self.invoke()
        self.assert_unpublished()

    def test_selection_report_tampering_is_refused(self):
        self.selection.write_text(self.selection.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'Selection SHA mismatch'):
            self.invoke()
        self.assert_unpublished()

    def test_input_mutated_during_zip_creation_is_refused(self):
        original = zipfile.ZipFile.write
        def mutate(archive, file, arcname, *args, **kwargs):
            result = original(archive, file, arcname, *args, **kwargs)
            if arcname.endswith('/NOTICE_EQUIPE_GABARIT.txt'):
                self.notice.write_text('changed after copying')
            return result
        with patch.object(zipfile.ZipFile,'write',mutate):
            with self.assertRaisesRegex(ValueError, 'Input changed during packaging'):
                self.invoke()
        self.assert_unpublished()

    def test_failed_final_publication_rolls_back_only_created_files(self):
        with patch.object(pkg.os, 'link', side_effect=OSError('synthetic disk failure')):
            with self.assertRaisesRegex(OSError, 'synthetic disk failure'):
                self.invoke()
        self.assert_unpublished()
        self.assertTrue(self.notice.exists())

    def test_archive_created_by_another_process_is_preserved(self):
        original = pkg.os.link
        def race(source, destination):
            Path(destination).write_bytes(b'concurrent user file')
            return original(source, destination)
        with patch.object(pkg.os, 'link', side_effect=race):
            with self.assertRaises(FileExistsError):
                self.invoke()
        self.assertEqual(self.out.read_bytes(), b'concurrent user file')
        self.assertFalse((self.run/'STATISTIQUES.json').exists())
        self.assertFalse(self.out.with_suffix('.verification.json').exists())


if __name__=='__main__':
    unittest.main()
