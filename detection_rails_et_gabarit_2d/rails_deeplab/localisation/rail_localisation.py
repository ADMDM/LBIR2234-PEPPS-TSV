"""Offline, auditable railway matching and route-length estimates (not navigation safety).

Input GPS positions are observations, not a trajectory reconstructed from images.
Native video timestamps must be supplied by the calling program.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import sqlite3

from pyproj import Geod, Transformer
from shapely import wkb
from shapely.geometry import LineString, Point, mapping
from shapely.ops import substring, transform

GEOD = Geod(ellps="WGS84")
DATASET_DOI = "10.5281/zenodo.17861120"
DATASET_MD5 = "f243222394e1db6a5f719f34570a92a9"


def unpack_geometry(blob):
    """Read a standard GeoPackage geometry header then its WKB geometry."""
    if blob[:2] != b"GP":
        raise ValueError("Not a GeoPackage geometry")
    envelope_code = (blob[3] >> 1) & 7
    if envelope_code not in {0, 1, 2, 3, 4}:
        raise ValueError("Unsupported geometry envelope")
    offset = 8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[envelope_code]
    return wkb.loads(blob[offset:])


def single_gauge_m(value):
    """Dual, uncertain or absent gauges never become a silently chosen number."""
    text = str(value or "").strip()
    if not re.fullmatch(r"\d{3,4}", text):
        return None
    mm = int(text)
    return mm / 1000 if 500 <= mm <= 2000 else None


def transverse_proxy(x_left, x_right, width, gauge_m, min_width_fraction=0.02):
    """Row-local approximation, in metres conditional on gauge and perspective.

    Origin: the image vertical midline, not the vehicle/optical/track centre.
    No longitudinal distance or 3D point is inferred.
    """
    if not all(math.isfinite(v) for v in [x_left, x_right, width, gauge_m]):
        raise ValueError("Non-finite geometry")
    if width <= 1 or gauge_m <= 0 or not 0 <= x_left < x_right < width:
        raise ValueError("Invalid boundaries or image size")
    w = x_right - x_left
    if w < min_width_fraction * width:
        return None
    scale = gauge_m / w
    origin = (width - 1) / 2
    return {"scale_m_per_px": scale, "left_offset_from_image_midline_m": (x_left - origin) * scale,
            "right_offset_from_image_midline_m": (x_right - origin) * scale}


@dataclass
class Edge:
    id: str
    start: str
    end: str
    xy: LineString
    properties: dict


class RailNetwork:
    def __init__(self, edges, forward, inverse, epsg, metadata=None):
        self.edges = {e.id: e for e in edges}
        self.forward, self.inverse, self.epsg = forward, inverse, epsg
        self.metadata = metadata or {}
        self.node_edges = {}
        for e in self.edges.values():
            for node in (e.start,e.end): self.node_edges.setdefault(node,set()).add(e.id)

    def same_degree_two_continuation(self, a, b):
        shared = {a.start,a.end} & {b.start,b.end}
        return bool(shared) and all(len(self.node_edges[n]) == 2 for n in shared) and bool(a.properties.get("line")) and all(
            a.properties.get(k) == b.properties.get(k) for k in ("line","gauge","status"))

    @classmethod
    def from_geopackage(cls, path, observations, margin_m=20000, statuses=("open",)):
        coords = [(float(o["longitude_deg"]), float(o["latitude_deg"])) for o in observations]
        if not coords or any(not (-180 <= lon <= 180 and -80 <= lat <= 84) for lon, lat in coords):
            raise ValueError("Missing/invalid GPS positions for UTM")
        lon = sum(p[0] for p in coords) / len(coords)
        lat = sum(p[1] for p in coords) / len(coords)
        zone = min(60, int((lon + 180) // 6) + 1)
        epsg = (32600 if lat >= 0 else 32700) + zone
        forward = Transformer.from_crs(4326, epsg, always_xy=True)
        inverse = Transformer.from_crs(epsg, 4326, always_xy=True)
        # Regional prototype: refuse long multi-zone journeys instead of distorting them.
        if max(GEOD.inv(lon, lat, x, y)[2] for x, y in coords) > 100000:
            raise ValueError("Use regional windows for journeys extending over 100 km from the centre")
        dlat = margin_m / 110000
        dlon = dlat / math.cos(math.radians(lat))
        bounds = [min(p[0] for p in coords)-dlon, min(p[1] for p in coords)-dlat,
                  max(p[0] for p in coords)+dlon, max(p[1] for p in coords)+dlat]
        con = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        with con:
            crs = con.execute("select srs_id from gpkg_geometry_columns where table_name='edges'").fetchone()
            if not crs or crs[0] != 4326:
                raise ValueError("Expected EPSG:4326 edges")
            rows = con.execute("""select e.* from edges e join rtree_edges_geom r on e.fid=r.id
                where r.maxx>=? and r.minx<=? and r.maxy>=? and r.miny<=?""",
                               (bounds[0], bounds[2], bounds[1], bounds[3])).fetchall()
            node_ids = {r[key] for r in rows for key in ("from_id", "to_id")}
            nodes = {}
            # Read node coordinates to verify direction and avoid connecting geometric gaps.
            for row in con.execute("select id, geom from nodes"):
                if row["id"] in node_ids:
                    nodes[row["id"]] = transform(forward.transform, unpack_geometry(row["geom"]))
        con.close()
        edges, rejected = [], []
        for row in rows:
            p = dict(row)
            geom = unpack_geometry(p.pop("geom"))
            if p["status"] not in statuses:
                rejected.append({"id": p["id"], "reason": "status_excluded", "status": p["status"]})
                continue
            xy = transform(forward.transform, geom)
            a, b = nodes.get(p["from_id"]), nodes.get(p["to_id"])
            if a is None or b is None or geom.geom_type != "LineString" or xy.length <= 0:
                rejected.append({"id": p["id"], "reason": "invalid_geometry_or_nodes"})
                continue
            if Point(xy.coords[0]).distance(b) + Point(xy.coords[-1]).distance(a) < Point(xy.coords[0]).distance(a) + Point(xy.coords[-1]).distance(b):
                xy = LineString(list(xy.coords)[::-1])
            mismatch = max(Point(xy.coords[0]).distance(a), Point(xy.coords[-1]).distance(b))
            if mismatch > 1.0:
                rejected.append({"id": p["id"], "reason": "endpoint_mismatch", "mismatch_m": mismatch})
                continue
            edges.append(Edge(p["id"], p["from_id"], p["to_id"], xy, p))
        return cls(edges, forward, inverse, epsg,
                   {"bbox_wgs84": bounds, "margin_m": margin_m, "included_statuses": list(statuses),
                    "loaded_edges": len(edges), "excluded_edges": rejected,
                    "dataset_doi": DATASET_DOI, "length_method": "WGS84 geodesic sum of geometry segments; not dataset length_m"})

    def to_wgs84(self, geom):
        return transform(self.inverse.transform, geom)

    def length_m(self, geom):
        return abs(GEOD.geometry_length(self.to_wgs84(geom)))

    def match(self, observation, max_distance_m=50.0, ambiguity_gap_m=15.0):
        lon, lat = float(observation["longitude_deg"]), float(observation["latitude_deg"])
        if not math.isfinite(lon + lat) or not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("Invalid GPS observation")
        point = Point(self.forward.transform(lon, lat))
        candidates = sorted(((e.xy.distance(point), e.id) for e in self.edges.values()))[:5]
        if not candidates:
            return dict(observation, match_status="no_network", gauge_mapped_nominal_m=None)
        distance, eid = candidates[0]
        edge = self.edges[eid]
        position = edge.xy.project(point)
        snapped = edge.xy.interpolate(position)
        snapped_wgs = self.to_wgs84(snapped)
        # Two adjacent pieces of the same line at a degree-two node are not two
        # competing tracks. A branch (degree >= 3) or parallel line stays ambiguous.
        ambiguous = any(d-distance < ambiguity_gap_m and not self.same_degree_two_continuation(edge,self.edges[eid2])
                        for d,eid2 in candidates[1:])
        status = "too_far" if distance > max_distance_m else "ambiguous" if ambiguous else "matched_candidate"
        return dict(observation, match_status=status, matched_edge_id=eid,
                    line_name=edge.properties.get("line"), gauge_raw=edge.properties.get("gauge"),
                    gauge_mapped_nominal_m=single_gauge_m(edge.properties.get("gauge")),
                    mapped_status=edge.properties.get("status"), distance_gps_to_map_m=distance,
                    snapped_latitude_deg=snapped_wgs.y, snapped_longitude_deg=snapped_wgs.x,
                    edge_position_utm_m=position, edge_length_utm_m=edge.xy.length,
                    alternative_candidates=[{"edge_id": eid2, "distance_m": d,
                                             "line": self.edges[eid2].properties.get("line"),
                                             "gauge": self.edges[eid2].properties.get("gauge")}
                                            for d, eid2 in candidates[1:]])

    def graph_with_observations(self, matches):
        """Insert exact projection points, including two observations on the same edge."""
        cuts = {}
        for i, match in enumerate(matches):
            if match["match_status"] != "matched_candidate":
                raise ValueError(f"Observation {i} is not unambiguous: {match['match_status']}")
            cuts.setdefault(match["matched_edge_id"], []).append((match["edge_position_utm_m"], f"gps_{i}"))
        graph, pieces = {}, {}
        for edge in self.edges.values():
            points = sorted([(0.0, edge.start), (edge.xy.length, edge.end)] + cuts.get(edge.id, []))
            for part, ((s, a), (t, b)) in enumerate(zip(points, points[1:])):
                pid = f"{edge.id}:{part}"
                geo = substring(edge.xy, s, t)
                cost = self.length_m(geo) if geo.geom_type == "LineString" else 0.0
                pieces[pid] = {"edge_id": edge.id, "geometry": geo, "from": a, "to": b, "length_m": cost}
                graph.setdefault(a, []).append((b, cost, pid))
                graph.setdefault(b, []).append((a, cost, pid))
        return graph, pieces


def shortest_path(graph, start, end, excluded_piece=None):
    queue = [(0.0, start)]
    distances, parents = {start: 0.0}, {}
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distances[node]:
            continue
        if node == end:
            path = []
            while node != start:
                previous, pid = parents[node]
                path.append((previous, node, pid))
                node = previous
            return cost, path[::-1]
        for neighbor, length, pid in graph.get(node, []):
            if pid == excluded_piece:
                continue
            new = cost + length
            if new < distances.get(neighbor, math.inf):
                distances[neighbor] = new
                parents[neighbor] = (node, pid)
                heapq.heappush(queue, (new, neighbor))
    return None


def analyze_trace(network, observations, max_distance_m=50, ambiguity_gap_m=15):
    times = [float(o["video_time_s"]) for o in observations]
    if len(times) < 2 or any(not math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("At least two observations with strictly increasing timestamps are required")
    matches = [network.match(o, max_distance_m, ambiguity_gap_m) for o in observations]
    report = {"network": network.metadata, "projection_epsg": network.epsg, "matches": matches,
              "thresholds": {"max_distance_m": max_distance_m, "ambiguity_gap_m": ambiguity_gap_m,
                             "note": "Engineering controls, not a measured uncertainty or confidence interval"},
              "gauge_design_default_m": 1.676, "gauge_selected_nominal_m": None,
              "gauge_selection_status": "unresolved", "route_status": "unresolved",
              "limitations": ["Shortest mapped route is a hypothesis, not odometry or proof of no reversal/loop",
                              "Dataset represents corridors, not each physical track or live switch state",
                              "Historical video and map publication dates differ; open does not prove operating status",
                              "GPS errors, latency and camera calibration are unmeasured",
                              "Map residuals are not rail-detection error; local metre scales remain conditional"]}
    if any(m["match_status"] != "matched_candidate" for m in matches):
        report["route_status"] = "no_network_coverage" if all(m["match_status"] == "no_network" for m in matches) else "unresolved_observations"
        return report, {"type": "FeatureCollection", "features": []}
    graph, pieces = network.graph_with_observations(matches)
    intervals, features = [], []
    cumulative = 0.0
    matches[0]["route_cumulative_estimate_m"] = 0.0
    used_edges = set()
    for i, (a, b) in enumerate(zip(matches, matches[1:])):
        found = shortest_path(graph, f"gps_{i}", f"gps_{i+1}")
        if found is None:
            report["route_status"] = "disconnected_in_regional_network"
            return report, {"type": "FeatureCollection", "features": []}
        length, path = found
        alternatives = [v[0] for _, _, pid in path if (v := shortest_path(graph, f"gps_{i}", f"gps_{i+1}", pid))]
        alternative_length = min(alternatives) if alternatives else None
        dt = float(b["video_time_s"]) - float(a["video_time_s"])
        interval = {"start_frame_index": a["frame_index"], "end_frame_index": b["frame_index"],
                    "start_video_time_s": float(a["video_time_s"]), "end_video_time_s": float(b["video_time_s"]),
                    "route_distance_estimate_m": length, "interval_mean_speed_estimate_m_s": length / dt,
                    "alternative_simple_route_length_m": alternative_length,
                    "alternative_status": "exists" if alternatives else "none_in_loaded_graph_not_proof_of_unique_real_trip"}
        intervals.append(interval)
        cumulative += length
        b["route_cumulative_estimate_m"] = cumulative
        for origin, destination, pid in path:
            piece = pieces[pid]
            used_edges.add(piece["edge_id"])
            geom = piece["geometry"]
            if geom.geom_type != "LineString":
                continue
            if piece["from"] != origin:
                geom = LineString(list(geom.coords)[::-1])
            features.append({"type": "Feature", "geometry": mapping(network.to_wgs84(geom)),
                             "properties": {"kind": "route_piece", "interval": i,
                                            "edge_id": piece["edge_id"], "length_estimate_m": piece["length_m"]}})
    end_to_end = shortest_path(graph, "gps_0", f"gps_{len(matches)-1}")
    first, last = observations[0], observations[-1]
    direct = GEOD.inv(float(first["longitude_deg"]), float(first["latitude_deg"]),
                      float(last["longitude_deg"]), float(last["latitude_deg"]))[2]
    gauges = [single_gauge_m(network.edges[eid].properties.get("gauge")) for eid in used_edges]
    selected_gauge = gauges[0] if gauges and all(g is not None and g == gauges[0] for g in gauges) else None
    report.update(route_status="mapped_route_estimate_with_observation_checks", intervals=intervals,
                  route_distance_estimate_m=cumulative, endpoint_shortest_route_estimate_m=end_to_end[0],
                  endpoint_displacement_wgs84_m=direct, observed_duration_video_s=times[-1]-times[0],
                  whole_clip_mean_speed_estimate_km_h=cumulative/(times[-1]-times[0])*3.6,
                  gauge_selected_nominal_m=selected_gauge,
                  gauge_selection_status="consistent_mapped_nominal_not_field_measured" if selected_gauge else "unknown_or_mixed_gauge",
                  route_edges=[network.edges[k].properties for k in sorted(used_edges)])
    for i, m in enumerate(matches):
        for kind, x, y in [("gps_observation", float(m["longitude_deg"]), float(m["latitude_deg"])),
                           ("snapped_to_network", m["snapped_longitude_deg"], m["snapped_latitude_deg"])]:
            features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [x, y]},
                             "properties": {"kind": kind, "sample_index": i, "frame_index": m["frame_index"],
                                            "video_time_s": float(m["video_time_s"]), "residual_m": m["distance_gps_to_map_m"]}})
    return report, {"type": "FeatureCollection", "features": features}


def write_csv(path, records):
    if not records:
        return
    keys = list(dict.fromkeys(key for rec in records for key in rec))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--gps-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--margin-m", type=float, default=20000)
    parser.add_argument("--max-distance-m", type=float, default=50)
    parser.add_argument("--ambiguity-gap-m", type=float, default=15)
    args = parser.parse_args()
    if hashlib.md5(args.network.read_bytes()).hexdigest() != DATASET_MD5:
        raise ValueError("This reproducible run requires AfTS-Db v3 with the recorded checksum")
    with args.gps_csv.open(newline="", encoding="utf-8-sig") as f:
        observations = list(csv.DictReader(f))
    network = RailNetwork.from_geopackage(args.network, observations, args.margin_m)
    report, geojson = analyze_trace(network, observations, args.max_distance_m, args.ambiguity_gap_m)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "RESULTATS_LOCALISATION.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (args.out / "TRAJET_CARTOGRAPHIQUE.geojson").write_text(json.dumps(geojson, ensure_ascii=False))
    write_csv(args.out / "POSITIONS_RACCORDEES.csv", report["matches"])
    write_csv(args.out / "DISTANCES_ENTRE_OBSERVATIONS.csv", report.get("intervals", []))
    context = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(network.to_wgs84(e.xy)), "properties": e.properties}
        for e in network.edges.values()]}
    (args.out / "RESEAU_LOCAL.geojson").write_text(json.dumps(context, ensure_ascii=False))
    print(json.dumps({k: report.get(k) for k in ["route_status", "route_distance_estimate_m", "gauge_selected_nominal_m"]}, indent=2))


if __name__ == "__main__":
    main()
