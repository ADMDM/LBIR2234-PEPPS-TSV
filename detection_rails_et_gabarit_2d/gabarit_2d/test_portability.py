"""Relocated-source contract tests using only synthetic temporary video."""
import json
from pathlib import Path
import tempfile
import unittest

from test_verify_gabarit_2d import make_delivery, dump
import verify_gabarit_2d as verifier
from rail_io import sha


class RelocatedSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.delivery = make_delivery(self.root)
        self.manifest_path = self.delivery / 'MANIFESTE.json'
        self.manifest = json.loads(self.manifest_path.read_text())
        original = Path(self.manifest['source'])
        self.moved = self.root / 'copied_native_source.mkv'
        original.rename(self.moved)
        digest = self.manifest['inputs_sha256'].pop(str(original))
        self.manifest['inputs_sha256'][str(self.moved)] = digest
        self.manifest['source'] = str(self.moved)
        self.manifest['parameters']['source'] = str(self.moved)
        dump(self.manifest_path, self.manifest)

    def tearDown(self):
        self.temp.cleanup()

    def test_identical_source_at_new_path_preserves_native_timing(self):
        report = verifier.verify_delivery(self.delivery)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['source_and_rendered_pts_checked'], 3)
        self.assertEqual(sha(self.moved), self.manifest['source_sha256'])

    def test_relocation_does_not_accept_modified_video_bytes(self):
        with self.moved.open('ab') as handle:
            handle.write(b'changed after source proof')
        with self.assertRaisesRegex(ValueError, 'Input changed'):
            verifier.verify_delivery(self.delivery)

    def test_override_must_reference_the_actual_manifest_source(self):
        self.manifest['parameters']['source'] = str(self.root / 'other.mkv')
        dump(self.manifest_path, self.manifest)
        with self.assertRaisesRegex(ValueError, 'Wrong rail/video pairing'):
            verifier.verify_delivery(self.delivery)


if __name__ == '__main__':
    unittest.main()
