import csv,json,tempfile,unittest
from pathlib import Path
from gps_inputs import parse_gps_text,extract_gps_candidate,reject_spatial_outliers,normalize_csv
from build_handoff import build_handoff


class ParsingTests(unittest.TestCase):
    def test_explicit_dms_profile_restores_missing_separators(self):
        a=parse_gps_text('N 12.3456.7 E 45.0 067.89','tsv_dms')
        self.assertAlmostEqual(a[0],12+34/60+56.7/3600)
        self.assertAlmostEqual(a[1],45+6/60+7.89/3600)
        with self.assertRaises(ValueError):parse_gps_text('N 7.42 E 3.8','tsv_dms')

    def test_isolated_short_excursion_is_rejected_without_guessing(self):
        rows=[dict(video_time_s=i,latitude_deg=7.,longitude_deg=x,status='parsed') for i,x in enumerate([3,3.0001,3.00002,3.0002])]
        reject_spatial_outliers(rows,isolated_excess_m=10)
        self.assertEqual(rows[2]['status'],'rejected_isolated_short_gps_excursion')
        self.assertEqual(rows[2]['longitude_deg'],3.00002)
    def test_dms_and_decimal(self):
        self.assertEqual(parse_gps_text('N 12 34\'56.7 " E 45.0 06\'7.89 "'),(12+34/60+56.7/3600,45+6/60+7.89/3600))
        self.assertEqual(parse_gps_text('S 7,5 W 3,25'),(-7.5,-3.25))
        self.assertAlmostEqual(parse_gps_text('N 7 25.5 E 3 54.5')[0],7.425)

    def test_terminal_seconds_mark_does_not_change_digits(self):
        self.assertAlmostEqual(parse_gps_text('N 12 34\'56.7 " E 45.0 06\'7.89 I')[1],45+6/60+7.89/3600)

    def test_bad_components_are_never_guessed(self):
        for text in ['N O7 25 9 E 3 53 34','N 7 25 9 E 3 53 3I','N 7 61 0 E 3 0 0','N 91 E 3','N 7 0 60 E 3 0 0','7.4 3.8','N -7 E 3']:
            with self.subTest(text=text),self.assertRaises(ValueError):parse_gps_text(text)

    def test_low_confidence_and_conflicts(self):
        self.assertTrue(extract_gps_candidate([{'text':'N 7 E 3','confidence':.2}])['status'].startswith('rejected'))
        lines=[{'text':'N 7 E 3','confidence':1},{'text':'N 8 E 3','confidence':1}]
        self.assertEqual(extract_gps_candidate(lines)['status'],'rejected_conflicting_coordinates')

    def test_gps_jump_quarantine_and_recovery(self):
        rows=[dict(video_time_s=i,latitude_deg=lat,longitude_deg=3.,status='parsed') for i,lat in enumerate([7.,7.00001,50.,7.00003])]
        reject_spatial_outliers(rows)
        self.assertEqual(rows[2]['status'],'rejected_implausible_gps_step')
        self.assertEqual(rows[3]['status'],'parsed')


class CsvTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.frames=[{'frame_index':i,'video_time_s':t} for i,t in enumerate([.0333333333,1.0333333333,2.0333333333])]

    def normalize(self,rows,config):
        path=self.root/'source.csv'
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter=config.get('delimiter',','));w.writeheader();w.writerows(rows)
        return normalize_csv(path,self.frames,config,'fixture','testhash')

    def test_renamed_columns_decimal_commas_elapsed_clock_and_speed(self):
        obs,audit=self.normalize([{'lat':'7,4','lon':'3,8','temps':'0','kmh':'36'},{'lat':'7,4','lon':'3,8','temps':'1','kmh':'0'}],
                                {'latitude_column':'lat','longitude_column':'lon','time_column':'temps','time_reference':'video_elapsed',
                                 'speed_column':'kmh','speed_unit':'km/h','delimiter':';'})
        self.assertEqual(len(obs),2);self.assertEqual(obs[0]['gps_speed_provided_m_s'],10)
        self.assertEqual(obs[1]['gps_speed_provided_m_s'],0)
        self.assertEqual(obs[0]['video_time_s'],self.frames[0]['video_time_s'])

    def test_utc_time_acceleration_is_explicit(self):
        obs,_=self.normalize([{'latitude_deg':'7','longitude_deg':'3','timestamp':'2026-01-01T00:00:10Z'}],
                             {'time_column':'timestamp','time_reference':'utc','video_start_utc':'2026-01-01T00:00:00Z',
                              'real_seconds_per_video_second':10})
        self.assertEqual(obs[0]['frame_index'],1)

    def test_utc_without_timezone_is_rejected(self):
        with self.assertRaises(ValueError):
            self.normalize([{'latitude_deg':7,'longitude_deg':3,'video_time_s':0}],{'time_reference':'utc','video_start_utc':'2026-01-01T00:00:00'})

    def test_wrong_hash_and_frame_rejected(self):
        for extra,config in [({'source_video_sha256':'wrong'},{}),({'f':'2'},{'frame_column':'f'}),({'video_id':'other'}, {})]:
            obs,audit=self.normalize([dict(latitude_deg=7,longitude_deg=3,video_time_s=.0333333333,**extra)],config)
            self.assertFalse(obs);self.assertTrue(audit[0]['status'].startswith('rejected'))

    def test_bad_coords_and_off_timeline_rejected(self):
        rows=[{'latitude_deg':'nan','longitude_deg':3,'video_time_s':0},{'latitude_deg':7,'longitude_deg':3,'video_time_s':30}]
        obs,audit=self.normalize(rows,{})
        self.assertFalse(obs);self.assertEqual(len(audit),2)

    def test_missing_columns_fail(self):
        with self.assertRaises(ValueError):self.normalize([{'lat':7,'lon':3}],{})

    def test_time_order_and_duplicate_frames_audited(self):
        rows=[dict(latitude_deg=7,longitude_deg=3,video_time_s=t) for t in [.0333333,.05,1.0333333,1.0]]
        obs,audit=self.normalize(rows,{})
        self.assertEqual(len(obs),2);self.assertEqual(sum(r['status'].startswith('rejected') for r in audit),2)


class ExportTests(unittest.TestCase):
    def read(self,path):
        with path.open(newline='') as f:return list(csv.DictReader(f))
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.rows=[dict(video_id='other_video',frame_index=i,video_time_s=t,source_video_sha256='other_hash',image_width_px=640,
                        left_segments_json='[[[200,400]]]',right_segments_json='[[[440,400]]]',train_speed_m_s=12.5) for i,t in enumerate([0,1,2,3])]
        self.csv=self.root/'rails.csv'
        with self.csv.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(self.rows[0]));w.writeheader();w.writerows(self.rows)
        self.matches=[dict(video_id='other_video',source_video_sha256='other_hash',frame_index=i,video_time_s=t,
                           latitude_deg=7.,longitude_deg=3.,route_cumulative_estimate_m=distance,gps_source='provided_csv')
                      for i,t,distance in [(0,0,0),(2,2,20)]]
        self.route=dict(route_status='mapped_route_estimate_with_observation_checks',gauge_selected_nominal_m=1.435,matches=self.matches,
                        route_edges=[{'line':'synthetic'}],network={'dataset_doi':'synthetic'},route_distance_estimate_m=20)

    def test_generic_video_and_gauge_preserve_existing_speed(self):
        out=self.root/'result';counts=build_handoff(self.csv,self.route,out)
        self.assertEqual(counts['nominal_gauge_m'],1.435)
        rows=self.read(out/'RAILS_GAUCHE_DROIT_GPS.csv')
        self.assertEqual(float(rows[1]['route_cumulative_estimate_m']),10)
        self.assertEqual(rows[1]['train_speed_m_s'],'12.5')
        self.assertEqual(rows[1]['gps_latitude_deg'],'')
        self.assertEqual(rows[3]['route_cumulative_estimate_m'],'')
        points=self.read(out/'POINTS_RAILS_METRES.csv')
        self.assertAlmostEqual(float(points[0]['scale_m_per_px_nominal']),1.435/240)
        self.assertEqual(points[0]['actual_error_m'],'')

    def test_unknown_gauge_keeps_pixels_and_design_only(self):
        self.route.update(route_status='no_network_coverage',gauge_selected_nominal_m=None)
        out=self.root/'unknown';build_handoff(self.csv,self.route,out)
        points=self.read(out/'POINTS_RAILS_METRES.csv')
        self.assertEqual(points[0]['scale_m_per_px_nominal'],'')
        self.assertNotEqual(points[0]['scale_m_per_px_design'],'')

    def test_long_gap_not_interpolated(self):
        out=self.root/'gap';build_handoff(self.csv,self.route,out,max_gap_s=1)
        rows=self.read(out/'RAILS_GAUCHE_DROIT_GPS.csv')
        self.assertEqual(rows[1]['route_cumulative_estimate_m'],'')
        self.assertEqual(rows[1]['route_distance_status'],'withheld_gap_too_long')

    def test_mixed_video_refused(self):
        self.route['source_video_sha256']='wrong'
        with self.assertRaises(ValueError):build_handoff(self.csv,self.route,self.root/'bad')

    def test_frame_timestamp_mismatch_refused(self):
        self.matches[0]['video_time_s']=.1
        with self.assertRaises(ValueError):build_handoff(self.csv,self.route,self.root/'bad_time')


if __name__=='__main__':unittest.main()
