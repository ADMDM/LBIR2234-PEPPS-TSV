import math,unittest
from aligner_vitesse import SpeedTimeline,align

class TestSpeedAlignment(unittest.TestCase):
    def test_linear_speed_and_piecewise_distance(self):
        s=SpeedTimeline([(0,10),(1,20),(2,10)],1.1)
        self.assertEqual(s.at(.5),(15,'interpolated'))
        self.assertEqual(s.integral(0,2),30)
        self.assertEqual(s.integral(.5,1.5),17.5)
    def test_gap_and_no_extrapolation(self):
        s=SpeedTimeline([(0,10),(1,10),(5,20),(6,20)],1.1)
        self.assertIsNone(s.at(2)[0]);self.assertIsNone(s.at(-1)[0]);self.assertIsNone(s.at(7)[0])
        self.assertIsNone(s.integral(1,5));self.assertEqual(s.integral(5,6),20)
    def test_repeated_image_keeps_real_time_speed(self):
        frames=[dict(time_s=t,repeated_source_frame=str(i==1)) for i,t in enumerate([0,.5,1])]
        r=align(frames,SpeedTimeline([(0,10),(1,20)],1.1),'GPS',0)
        self.assertEqual(r[1]['train_speed_m_s'],15)
        self.assertEqual(r[-1]['distance_from_video_start_m'],15)
    def test_missing_start_or_gap_does_not_invent_global_distance(self):
        s=SpeedTimeline([(1,10),(2,10),(5,10),(6,10)],1.1)
        rows=align([dict(time_s=t) for t in [0,1,2,3,5,6]],s,'GPS',0)
        self.assertTrue(all(r['distance_from_video_start_m'] is None for r in rows))
        self.assertEqual(rows[-1]['segment_distance_m'],10)
    def test_reject_duplicate_times(self):
        with self.assertRaises(ValueError):SpeedTimeline([(0,1),(0,2)],1)

if __name__=='__main__':unittest.main()
