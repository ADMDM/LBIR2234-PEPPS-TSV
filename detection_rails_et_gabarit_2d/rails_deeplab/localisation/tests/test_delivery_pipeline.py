"""Small synthetic failures that must prevent a misleading combined delivery."""
from pathlib import Path
import csv
import json
import tempfile
import unittest
import zipfile
from handoff_bundle import timeline, check_frame, validate_geometry, join_observations, package_delivery, run_identity
from render_connected_video import along_path, pane_transform
from run_pipeline import run


def csv_file(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.rail = dict(video_id='synthetic', source_video_sha256='a'*64, frame_index='0',
                         video_time_s='0.033333333333', image_width_px='640', image_height_px='480',
                         gps_latitude_deg='', gps_longitude_deg='', route_cumulative_estimate_m='12.5')
        self.obj = dict(video_id='synthetic', frame_index='0', time_video_s='0.033333333333',
                        image_width='640', image_height='480', observation_id='synthetic:0:0',
                        distance_m='', confidence_score='0.6')

    def tearDown(self):
        self.tmp.cleanup()

    def test_duplicate_frames_fail(self):
        with self.assertRaises(ValueError): timeline([self.rail, self.rail])

    def test_wrong_video_fails(self):
        with self.assertRaises(ValueError): check_frame(dict(self.obj,video_id='another'),timeline([self.rail]))

    def test_wrong_timestamp_fails(self):
        with self.assertRaises(ValueError): check_frame(dict(self.obj,time_video_s='0.2'),timeline([self.rail]))

    def test_nan_timestamp_fails(self):
        with self.assertRaises(ValueError): check_frame(dict(self.obj,time_video_s='nan'),timeline([self.rail]))

    def test_resized_geometry_fails(self):
        with self.assertRaises(ValueError): check_frame(dict(self.obj,image_width='1920'),timeline([self.rail]))

    def test_camera_gps_not_object_distance(self):
        source,dest=self.root/'objects.csv',self.root/'joined.csv'
        csv_file(source,[self.obj]);join_observations(source,dest,timeline([self.rail]))
        with dest.open() as f: row=next(csv.DictReader(f))
        self.assertEqual(row['distance_m'],'');self.assertEqual(row['gps_latitude_deg'],'')
        self.assertEqual(row['gps_context_meaning'],'camera_position_not_object_geolocation')
        self.assertEqual(row['confidence_score'],'0.6');self.assertEqual(row['route_cumulative_estimate_m'],'12.5')

    def test_geometry_mismatch_fails(self):
        a,b=self.root/'a.csv',self.root/'b.csv'
        x=dict(video_id='synthetic',frame_index='0',segment_index='0',time_video_s='1',
               y_px='300',left_display_px='200',right_display_px='400',display_valid='True')
        y=dict(video_id='synthetic',frame_index='0',segment_index='0',video_time_s='1',
               y_px='300',left_x_px='201',right_x_px='400')
        csv_file(a,[x]);csv_file(b,[y])
        with self.assertRaises(ValueError):validate_geometry(a,b)

    def test_curve_interpolation_follows_bend(self):
        self.assertEqual(along_path([(0,0),(10,0),(10,10)],.75),(10,5))

    def test_stationary_interpolation(self):
        self.assertEqual(along_path([(1,2),(1,2)],.5),(1,2))

    def test_vga_is_not_stretched(self):
        w,h,x,y=pane_transform(640,480)
        self.assertAlmostEqual(w/h,640/480)
        self.assertEqual((w,h,x,y),(1080,810,180,70))

    def test_zip_checks_all_files_and_refuses_overwrite(self):
        d=self.root/'delivery';d.mkdir();(d/'a.txt').write_text('hello')
        z=self.root/'out.zip';report=package_delivery(d,z)
        self.assertEqual(report['files_verified'],1)
        with zipfile.ZipFile(z) as archive:self.assertEqual(archive.read('delivery/a.txt'),b'hello')
        with self.assertRaises(FileExistsError):package_delivery(d,z)

    def test_configuration_is_not_a_shell_command(self):
        p=self.root/'bad.json';p.write_text(json.dumps(dict(schema_version=1,video_id='../escape')))
        with self.assertRaises(ValueError):run(p,True)

    def test_older_verified_marker_uses_full_provenance(self):
        (self.root/'RUN_COMPLETED.json').write_text(json.dumps(dict(status='complete_and_independently_verified',video_id='synthetic')))
        (self.root/'RESULTATS_LOCALISATION.json').write_text(json.dumps(dict(video_id='synthetic',source_video_sha256='a'*64)))
        (self.root/'VERIFICATION_INDEPENDANTE.json').write_text(json.dumps(dict(status='passed',sha256={})))
        self.assertEqual(run_identity(self.root)['source_video_sha256'],'a'*64)

    def test_failed_verification_is_not_accepted(self):
        (self.root/'RUN_COMPLETED.json').write_text(json.dumps(dict(status='complete_and_independently_verified')))
        (self.root/'RESULTATS_LOCALISATION.json').write_text(json.dumps(dict(video_id='synthetic',source_video_sha256='a'*64)))
        (self.root/'VERIFICATION_INDEPENDANTE.json').write_text(json.dumps(dict(status='failed')))
        with self.assertRaises(ValueError):run_identity(self.root)


if __name__=='__main__':unittest.main()
