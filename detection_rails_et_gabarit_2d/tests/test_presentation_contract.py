"""Tests for failure-prone coordinate, timing and uncertainty contracts."""
from pathlib import Path
from copy import deepcopy
import json
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rails_deeplab/presentation'))
from build_video import geometry_transform, compact_rail_row, smooth_existing, format_video_time
from map_view import gps_quality
from inspect_source import location


def rail(t=0,left=100,right=200):
    return dict(video_id='example',frame_index=0,video_time_s=t,image_width_px=640,image_height_px=480,
                left_segments_json=json.dumps([[[left,200],[left,208]]]),right_segments_json=json.dumps([[[right,200],[right,208]]]),
                source_video_sha256='a'*64,train_speed_m_s='',source_pts_s=t)


class PresentationContract(unittest.TestCase):
    def test_clock_carries_rounded_seconds_into_minutes(self):
        self.assertEqual(format_video_time(59.9666667), '01:00.0')
        self.assertEqual(format_video_time(179.9666667), '03:00.0')
        self.assertEqual(format_video_time(1521.4), '25:21.4')

    def test_metadata_position_does_not_claim_time(self):
        p=location('+01.2500+002.5000/')  # Synthetic parsing example.
        self.assertFalse(p['coordinate_time_known'])
        self.assertEqual(p['position_source'],'MP4 location tag')
        self.assertIsNone(location('+00.0000+000.0000/'))

    def test_missing_speed_stays_missing(self):
        r=compact_rail_row(rail(),geometry_transform(640,480))
        self.assertEqual(r['train_speed_m_s'],'');self.assertEqual(r['train_speed_km_h'],'')
        self.assertEqual(r['object_processing_status'],'not_analysed')

    def test_zero_speed_is_observed_zero(self):
        source=rail();source['gps_speed_provided_m_s']='0'
        r=compact_rail_row(source,geometry_transform(640,480),0)
        self.assertEqual(r['train_speed_km_h'],0);self.assertEqual(r['objects_in_corridor'],0)

    def test_vga_display_keeps_aspect(self):
        tr=geometry_transform(640,480)
        self.assertAlmostEqual(tr['scale_x'],tr['scale_y'])
        self.assertLessEqual(tr['pane_y']+tr['display_height'],924)

    def test_portrait_does_not_cover_footer(self):
        tr=geometry_transform(1080,1920,portrait=True)
        self.assertLess(tr['pane_y']+tr['display_height'],940)

    def test_crop_transform_roundtrip(self):
        tr=geometry_transform(1280,720,True,(400,0,480,720))
        x=500;y=600
        ox=tr['pane_x']+(x-tr['crop_x'])*tr['scale_x']
        oy=tr['pane_y']+(y-tr['crop_y'])*tr['scale_y']
        self.assertAlmostEqual((ox-tr['pane_x'])/tr['scale_x']+tr['crop_x'],x)
        self.assertAlmostEqual((oy-tr['pane_y'])/tr['scale_y']+tr['crop_y'],y)

    def test_smoothing_at_five_fps_and_gap_reset(self):
        rows=[rail(0),rail(.2,110,210),rail(2,120,220)]
        report=smooth_existing(rows,dict(min_cutoff_hz=.8,beta=15,max_gap_seconds=.35))
        self.assertGreater(report['states']['filtered'],0)
        self.assertGreater(report['states']['reset_gap'],0)
        self.assertEqual(json.loads(rows[1]['left_raw_segments_json'])[0][0][0],110)
        filtered=json.loads(rows[1]['left_segments_json'])[0][0][0]
        self.assertLess(filtered,110);self.assertGreater(filtered,100)
        self.assertEqual(json.loads(rows[2]['left_segments_json'])[0][0][0],120)

    def test_gps_spike_uses_real_utc(self):
        observations=[dict(utc_time=f'2026-01-01T00:00:0{i*2}+00:00',longitude_deg=x,latitude_deg=50.0) for i,x in enumerate([4.0,4.001,4.1,4.003,4.004])]
        old=deepcopy(observations)
        status,connections,_,_=gps_quality(observations)
        self.assertEqual(status[2],'withheld_isolated_gps_jump')
        self.assertFalse(connections[2]);self.assertFalse(connections[3])
        self.assertEqual(observations,old)

    def test_long_real_time_gap_breaks_trail(self):
        obs=[dict(utc_time='2026-01-01T00:00:00+00:00',longitude_deg=4,latitude_deg=50),dict(utc_time='2026-01-01T00:02:00+00:00',longitude_deg=4.01,latitude_deg=50)]
        _,connections,_,_=gps_quality(obs)
        self.assertFalse(connections[1])


if __name__=='__main__':unittest.main()
