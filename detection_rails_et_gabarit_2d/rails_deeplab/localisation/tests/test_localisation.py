import math
import unittest
from pyproj import Transformer
from shapely.geometry import LineString
from rail_localisation import Edge, RailNetwork, analyze_trace, shortest_path, single_gauge_m, transverse_proxy


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.forward = Transformer.from_crs(4326, 32631, always_xy=True)
        self.inverse = Transformer.from_crs(32631, 4326, always_xy=True)

    def edge(self, eid, a, b, coords, gauge="1067"):
        return Edge(eid, a, b, LineString(coords), {"line": "synthetic", "gauge": gauge, "status": "open"})

    def network(self, edges):
        return RailNetwork(edges, self.forward, self.inverse, 32631)

    def observation(self, x, y, t):
        lon, lat = self.inverse.transform(x, y)
        return {"frame_index": int(t*30), "video_time_s": t, "longitude_deg": lon, "latitude_deg": lat}

    def test_same_edge_routes_only_between_observations(self):
        net = self.network([self.edge("e", "a", "b", [(500000, 820000), (501000, 820000)])])
        report, _ = analyze_trace(net, [self.observation(500100, 820000, 0), self.observation(500300, 820000, 10)])
        self.assertAlmostEqual(report["route_distance_estimate_m"], 200.08, delta=0.02)
        self.assertEqual(report["gauge_selected_nominal_m"], 1.067)

    def test_reverse_travel_is_positive_distance(self):
        net = self.network([self.edge("e", "a", "b", [(500000, 820000), (501000, 820000)])])
        report, _ = analyze_trace(net, [self.observation(500300, 820000, 0), self.observation(500100, 820000, 10)])
        self.assertAlmostEqual(report["route_distance_estimate_m"], 200.08, delta=0.02)

    def test_return_trip_is_not_endpoint_displacement(self):
        net = self.network([self.edge("e", "a", "b", [(500000, 820000), (501000, 820000)])])
        observations = [self.observation(x, 820000, i*10) for i, x in enumerate([500100, 500300, 500100])]
        report, _ = analyze_trace(net, observations)
        self.assertAlmostEqual(report["route_distance_estimate_m"], 400.16, delta=0.03)
        self.assertAlmostEqual(report["endpoint_shortest_route_estimate_m"], 0, places=6)

    def test_disconnected_components_are_not_connected_by_a_straight_line(self):
        net = self.network([self.edge("e1", "a", "b", [(500000,820000),(500100,820000)]),
                            self.edge("e2", "c", "d", [(500200,820000),(500300,820000)])])
        report, geo = analyze_trace(net, [self.observation(500010,820000,0),self.observation(500290,820000,10)])
        self.assertEqual(report["route_status"], "disconnected_in_regional_network")
        self.assertNotIn("route_distance_estimate_m", report)
        self.assertFalse(geo["features"])

    def test_parallel_lines_are_ambiguous(self):
        net = self.network([self.edge("a", "a1", "a2", [(500000,820000),(501000,820000)]),
                            self.edge("b", "b1", "b2", [(500000,820010),(501000,820010)], "1435")])
        report, _ = analyze_trace(net, [self.observation(500100,820002,0), self.observation(500300,820002,10)])
        self.assertEqual(report["matches"][0]["match_status"], "ambiguous")
        self.assertIsNone(report["gauge_selected_nominal_m"])
        self.assertNotIn("route_distance_estimate_m", report)

    def test_degree_two_line_segmentation_is_not_a_branch(self):
        net=self.network([self.edge('e1','a','b',[(500000,820000),(500100,820000)]),
                          self.edge('e2','b','c',[(500100,820000),(500200,820000)])])
        self.assertEqual(net.match(self.observation(500099,820003,0))['match_status'],'matched_candidate')

    def test_actual_branch_stays_ambiguous(self):
        net=self.network([self.edge('e1','a','b',[(500000,820000),(500100,820000)]),
                          self.edge('e2','b','c',[(500100,820000),(500200,820000)]),
                          self.edge('e3','b','d',[(500100,820000),(500200,820050)])])
        self.assertEqual(net.match(self.observation(500099,820003,0))['match_status'],'ambiguous')

    def test_far_position_is_rejected(self):
        net = self.network([self.edge("e", "a", "b", [(500000,820000),(501000,820000)])])
        self.assertEqual(net.match(self.observation(500100,820100,0))["match_status"], "too_far")

    def test_unknown_and_dual_gauges_are_not_assumed(self):
        for value in [None, "", "unknown", "1067;1435", "1067?", "1435/1067"]:
            self.assertIsNone(single_gauge_m(value))
        self.assertEqual(single_gauge_m("1435"), 1.435)

    def test_unknown_gauge_still_allows_route_length(self):
        net = self.network([self.edge("e", "a", "b", [(500000,820000),(501000,820000)], "unknown")])
        report, _ = analyze_trace(net, [self.observation(500100,820000,0), self.observation(500200,820000,10)])
        self.assertGreater(report["route_distance_estimate_m"], 100)
        self.assertIsNone(report["gauge_selected_nominal_m"])

    def test_invalid_time_order(self):
        net = self.network([self.edge("e", "a", "b", [(500000,820000),(501000,820000)])])
        with self.assertRaises(ValueError):
            analyze_trace(net, [self.observation(500100,820000,10), self.observation(500200,820000,10)])

    def test_alternative_routing(self):
        graph = {"a": [("b",1,"ab"),("c",1,"ac")], "b": [("d",1,"bd")], "c": [("d",3,"cd")], "d": []}
        self.assertEqual(shortest_path(graph,"a","d")[0],2)
        self.assertEqual(shortest_path(graph,"a","d","ab")[0],4)

    def test_metre_proxy_and_narrow_rejection(self):
        self.assertIsNone(transverse_proxy(830,840,1920,1.067))
        proxy = transverse_proxy(577,941,1920,1.067)
        self.assertAlmostEqual(proxy["scale_m_per_px"],1.067/364)
        self.assertAlmostEqual(proxy["right_offset_from_image_midline_m"]-proxy["left_offset_from_image_midline_m"],1.067)
        with self.assertRaises(ValueError):
            transverse_proxy(950,940,1920,1.067)


if __name__ == "__main__":
    unittest.main()
