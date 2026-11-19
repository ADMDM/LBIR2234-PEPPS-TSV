#!/usr/bin/env python3
"""Independent, streaming verification of a V2 corridor refinement delivery.

No model is loaded. All decoded source/output frames, rail rows, candidate masks,
holes, intersections, object rows and point rows are checked. This validates
export consistency and geometry, not segmentation accuracy or physical danger.
The inference program and its geometry helpers are deliberately not imported.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import av
import cv2
import numpy as np

COMMON = ("video_id", "frame_index", "time_video_s", "source_pts", "time_base_num",
          "time_base_den", "image_width", "image_height")
OUTPUTS = ("VIDEO_GABARIT_2D.mp4", "OBJETS.csv", "POINTS_OBJETS.csv", "FRAMES.csv",
           "RAILS.csv", "AUDIT_FUSION.jsonl", "A_VERIFIER.csv")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mask_sha(mask):
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def integer(value, label="integer", minimum=0):
    require(not isinstance(value, (bool, np.bool_)), f"{label}: boolean is not an integer")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}: invalid integer") from exc
    require(math.isfinite(number) and number.is_integer() and number >= minimum,
            f"{label}: invalid integer {value!r}")
    return int(number)


def finite(value, label="number"):
    require(not isinstance(value, (bool, np.bool_)), f"{label}: boolean is not a number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}: invalid number") from exc
    require(math.isfinite(number), f"{label}: nonfinite number")
    return number


def boolean(value):
    require(value in (True, False, "True", "False") and not isinstance(value, (int, float))
            or isinstance(value, bool), f"Invalid boolean {value!r}")
    return value is True or value == "True"


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def json_text(text):
    return json.loads(text, object_pairs_hook=_no_duplicate_keys,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant {value}")))


def read_json(path):
    # Strict UTF-8: malformed or locale-specific text is rejected.
    return json_text(Path(path).read_text(encoding="utf-8"))


def common_check(row, base, label, source_only=False):
    keys = ("video_id", "frame_index", "time_video_s", "image_width", "image_height") if source_only else COMMON
    for key in keys:
        require(key in row, f"{label}: missing {key}")
        if key == "video_id":
            require(row[key] == base[key], f"{label}: wrong video ID")
        elif key == "time_video_s":
            require(abs(finite(row[key], key) - base[key]) <= 1e-9, f"{label}: source PTS mismatch")
        else:
            require(integer(row[key], key) == base[key], f"{label}: wrong {key}")


class CSVStream:
    """One row of lookahead; never retains an entire CSV or a point group."""
    def __init__(self, handle, label):
        self.label = label
        self.reader = csv.DictReader(handle)
        require(self.reader.fieldnames is not None, f"{label}: missing header")
        require(len(set(self.reader.fieldnames)) == len(self.reader.fieldnames), f"{label}: duplicate columns")
        self.fields = self.reader.fieldnames
        self.pending = next(self.reader, None)
        self.count = 0

    def pop(self):
        require(self.pending is not None, f"{self.label}: ended early")
        row = self.pending
        require(None not in row and None not in row.values(), f"{self.label}: malformed CSV row")
        self.pending = next(self.reader, None)
        self.count += 1
        return row

    def frame(self, index):
        result = []
        while self.pending is not None:
            current = integer(self.pending.get("frame_index"), f"{self.label} frame_index")
            require(current >= index, f"{self.label}: leftover/out-of-order frame {current}")
            if current != index:
                break
            result.append(self.pop())
        return result


def independent_corridor(rows, width, height):
    """Independent reconstruction of the documented display-profile convention."""
    parsed, spacings, seen_ids = [], [], set()
    prior = None
    for row in rows:
        y, sid = integer(row["y_px"]), integer(row["segment_index"])
        require(y < height, "Rail y outside original image")
        valid = boolean(row["display_valid"])
        if prior is not None:
            require(y > prior["y"], "Rail rows overlap or are not strictly increasing")
            if sid == prior["sid"]:
                spacings.append(y - prior["y"])
        if prior is None or sid != prior["sid"]:
            require(sid not in seen_ids, "Rail segment reappears")
            seen_ids.add(sid)
        item = dict(y=y, sid=sid, valid=valid)
        # Raw coordinates remain source evidence even for rejected display rows.
        left_raw, right_raw, axis_raw = (finite(row[k]) for k in ("left_raw_px", "right_raw_px", "axis_raw_px"))
        require(0 <= left_raw < right_raw < width, "Invalid raw rail limits")
        require(abs(axis_raw - (left_raw + right_raw) / 2) <= 1e-6, "Raw rail axis is not midpoint")
        if valid:
            left, right, axis = (finite(row[k]) for k in ("left_display_px", "right_display_px", "axis_display_px"))
            require(0 <= left < right < width, "Invalid displayed rail limits")
            require(abs(axis - (left + right) / 2) <= 1e-6, "Displayed rail axis is not midpoint")
            item.update(left=left, right=right)
        parsed.append(item)
        prior = item
    characteristic = None
    if spacings:
        small = sorted(spacings)[: (len(spacings) + 1) // 2]
        middle = len(small) // 2
        characteristic = float(small[middle] if len(small) % 2 else (small[middle - 1] + small[middle]) / 2)
    limit = min(16.0, 2 * characteristic) if characteristic is not None else 16.0
    rail = np.zeros((height, width), bool)
    supported_rows = np.zeros(height, bool)
    spans, breaks = [], []
    previous = None
    for row in parsed:
        if not row["valid"]:
            previous = None
            continue
        connect = previous is not None and previous["sid"] == row["sid"]
        if connect and row["y"] - previous["y"] > limit:
            breaks.append(dict(segment_index=row["sid"], y_before=previous["y"], y_after=row["y"], gap_px=row["y"] - previous["y"]))
            connect = False
        if connect:
            ys = range(previous["y"] + 1, row["y"] + 1)
            spans[-1]["y_last"] = row["y"]
            spans[-1]["sample_count"] += 1
        else:
            ys = [row["y"]]
            spans.append(dict(source_segment_index=row["sid"], y_first=row["y"], y_last=row["y"], sample_count=1))
        for y in ys:
            if connect and y != row["y"]:
                alpha = (y - previous["y"]) / (row["y"] - previous["y"])
                left = previous["left"] + alpha * (row["left"] - previous["left"])
                right = previous["right"] + alpha * (row["right"] - previous["right"])
            else:
                left, right = row["left"], row["right"]
            supported_rows[y] = True
            rail[y, math.ceil(left):math.floor(right) + 1] = True
        previous = row
    support = np.broadcast_to(supported_rows[:, None], (height, width))
    meta = dict(coordinates="original_image_pixel_centres", units="original_image_pixels", profile_kind="display",
                boundary_inclusion="left <= integer_x <= right",
                interpolation="linear_between_adjacent_valid_samples_in_same_segment", extrapolation=False,
                gap_limit_method="2x_median_lower_half_spacing_capped_at_16px", characteristic_spacing_px=characteristic,
                max_vertical_gap_px=limit, gap_breaks=breaks, input_rows=len(parsed),
                valid_sample_rows=sum(r["valid"] for r in parsed), invalid_sample_rows=sum(not r["valid"] for r in parsed),
                input_segments=len(seen_ids), observed_spans=spans, supported_rows=int(supported_rows.sum()),
                corridor_pixel_count=int(rail.sum()), physical_intrusion_assessed=False)
    return rail, support, meta


def decode_contours(contours, width, height):
    """Validate hierarchy and raster boundary topology before drawing all rings."""
    require(isinstance(contours, list), "Contours must be a list")
    raster = np.zeros((height, width), np.uint8)
    if not contours:
        return raster.astype(bool)
    arrays, siblings, depths, components = [], {}, [], []
    tree = np.full((1, len(contours), 4), -1, np.int32)
    component_count = 0
    for index, contour in enumerate(contours):
        require(integer(contour["contour_id"]) == index, "Contour IDs must be sequential")
        parent = contour["parent_contour_id"]
        if parent is not None:
            parent = integer(parent)
            require(parent < index, "Contour parent must precede its child")
        depth = 0 if parent is None else depths[parent] + 1
        require(integer(contour["depth"]) == depth and contour["is_hole"] is bool(depth % 2), "Wrong contour depth/hole flag")
        component = components[parent] if depth % 2 else component_count
        if not depth % 2:
            component_count += 1
        require(integer(contour["component_id"]) == component, "Wrong contour component ID")
        require(contour["closed"] is True, "Raster contours must be implicitly closed")
        points = contour["points"]
        require(isinstance(points, list) and points, "Empty contour")
        parsed = []
        for point in points:
            require(isinstance(point, list) and len(point) == 2, "Contour point must be an xy pair")
            x, y = integer(point[0], "contour x"), integer(point[1], "contour y")
            require(x < width and y < height, "Contour point outside original image")
            parsed.append((x, y))
        array = np.asarray(parsed, np.int32).reshape(-1, 1, 2)
        if len(parsed) > 1:
            adjacent = np.diff(np.vstack([array[:, 0, :], array[0, 0, :]]), axis=0)
            require(np.max(np.abs(adjacent)) <= 1, "Contour omits raster steps; CHAIN_APPROX_NONE required")
        kind = "point" if len(parsed) == 1 else ("polyline" if cv2.contourArea(array) == 0 else "polygon")
        require(contour["boundary_kind"] == kind, "Wrong contour boundary kind")
        arrays.append(array)
        depths.append(depth)
        components.append(component)
        tree[0, index, 3] = -1 if parent is None else parent
        siblings.setdefault(parent, []).append(index)
    for parent, children in siblings.items():
        if parent is not None:
            tree[0, parent, 2] = children[0]
        for pos, index in enumerate(children):
            tree[0, index, 0] = children[pos + 1] if pos + 1 < len(children) else -1
            tree[0, index, 1] = children[pos - 1] if pos else -1
    cv2.drawContours(raster, arrays, -1, 1, cv2.FILLED, lineType=cv2.LINE_8, hierarchy=tree)
    extracted, hierarchy = cv2.findContours(raster, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    require(len(extracted) == len(arrays), "Contour topology changed after reconstruction")
    for old, new in zip(arrays, extracted):
        require(np.array_equal(old, new), "Contours are not exact canonical raster boundaries")
    require(np.array_equal(tree, hierarchy), "Contour hierarchy disagrees with reconstructed raster")
    return raster.astype(bool)


def independent_analysis(mask, corridor, support):
    total = int(mask.sum())
    known = int(np.count_nonzero(mask & support))
    overlap = int(np.count_nonzero(mask & corridor))
    require(total > 0 and known > 0 and overlap > 0, "Exported candidate must have a nonempty real mask and observed overlap")
    complete = known == total
    return dict(mask_valid=True, mask_pixel_count=total, support_pixel_count=known,
                unsupported_pixel_count=total - known, support_fraction=known / total,
                support_status="full" if complete else "partial", overlap_pixel_count=overlap,
                overlap_fraction=overlap / total, overlap_fraction_supported=overlap / known,
                relation_2d="overlap_observed", relation_reason="overlap_on_full_support" if complete else "overlap_on_partial_support",
                complete_relation_assessment=complete,
                touches_image_border=bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
                physical_intrusion_assessed=False)


def same_value(actual, expected, label, csv_value=False):
    if expected is None:
        require(actual == "" if csv_value else actual is None, f"{label}: unknown value must be null/empty, never zero")
    elif isinstance(expected, bool):
        require(boolean(actual) == expected if csv_value else actual is expected, f"{label}: boolean mismatch")
    elif isinstance(expected, int):
        require(integer(actual, label) == expected, f"{label}: integer mismatch")
    elif isinstance(expected, float):
        require(abs(finite(actual, label) - expected) <= max(1e-12, abs(expected) * 1e-12), f"{label}: numeric mismatch")
    else:
        require(actual == expected, f"{label}: mismatch")


def expected_crop(corridor, padding):
    ys, xs = np.nonzero(corridor)
    if not len(xs):
        return None
    h, w = corridor.shape
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    px, py = max(32, math.ceil((right - left) * padding)), max(16, math.ceil((bottom - top) * padding))
    return [max(0, left - px), max(0, top - py), min(w, right + px), min(h, bottom + py)]


def independent_train_review(obj, mask, corridor):
    """Recompute the frozen V2 ambiguity gate; not a physical-absence claim."""
    yy, xx = np.nonzero(mask)
    require(len(xx) > 0, 'Cannot review empty mask')
    height = mask.shape[0]
    c_area = int(corridor.sum())
    inter = int(np.count_nonzero(mask & corridor))
    coverage = inter / c_area if c_area else 0.0
    h_frac = (int(yy.max()) + 1 - int(yy.min())) / height
    bottom = (int(yy.max()) + 1) / height
    score = finite(obj['confidence_score'])
    is_train = str(obj['class_name']).strip().casefold() == 'train'
    flagged = bool(is_train and c_area and score < .5 and coverage >= .9 and h_frac >= .75 and bottom >= .95)
    features = dict(quarantine=flagged, reason='ambiguous_train_tracklike' if flagged else 'train_not_matching_tracklike_guard',
                    class_name=obj['class_name'], confidence_score=score,
                    thresholds=dict(confidence_ceiling=.5, corridor_coverage_min=.9, height_fraction_min=.75, bottom_fraction_min=.95),
                    corridor_coverage=coverage, mask_height_fraction=h_frac, mask_bottom_fraction=bottom,
                    corridor_pixel_count=c_area, corridor_intersection_pixel_count=inter)
    return flagged, features


def expected_tile_views(corridor, tile_size, overlap):
    """Separate implementation of row-major crop coverage, no inference import."""
    height, width = corridor.shape
    crop = expected_crop(corridor, .15)
    if crop is None:
        return []
    size = integer(tile_size, 'tile size', 1)
    overlap = finite(overlap, 'tile overlap')
    require(0 <= overlap < 1, 'Invalid tile overlap')
    step = max(1, math.floor(size * (1 - overlap)))
    left, top, right, bottom = crop
    def positions(low, high):
        result = [low]
        if high - low <= size:
            return result
        end = high - size
        while result[-1] + step <= end:
            result.append(result[-1] + step)
        if result[-1] < end:
            result.append(end)
        return result
    views = []
    for y in positions(top, bottom):
        for x in positions(left, right):
            i = len(views)
            views.append(dict(view_id=f'tile_{i:04d}', view_kind='tile', tile_index=i,
                              crop_xyxy=[x, y, min(right, x + size), min(bottom, y + size)]))
    return views


def baseline_txt_proof(folder, baseline_manifest, baseline_verification):
    count = integer(baseline_manifest['frames'], 'baseline frame count', 1)
    expected = {f'frame_{i:06d}.txt' for i in range(count)}
    present = list((folder / 'TXT').iterdir())
    require(len(present) == count and {x.name for x in present} == expected and all(x.is_file() for x in present), 'Baseline TXT set changed')
    digest = hashlib.sha256()
    for index in range(count):
        rel = f'TXT/frame_{index:06d}.txt'
        digest.update(rel.encode() + b'\0' + bytes.fromhex(sha(folder / rel)))
    value = digest.hexdigest()
    require(value == baseline_verification['txt_path_and_content_aggregate_sha256'], 'Baseline TXT contents changed')
    require(integer(baseline_verification['txt_files_checked']) == count, 'Baseline proof did not cover all TXT')
    return value


def crop_border(mask, crop):
    if crop is None:
        return False
    height, width = mask.shape
    require(isinstance(crop, list) and len(crop) == 4, 'Invalid inference crop')
    x0, y0, x1, y1 = [integer(v, 'crop coordinate') for v in crop]
    require(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, 'Crop outside original image')
    require(not mask[:y0].any() and not mask[y1:].any() and not mask[:, :x0].any() and not mask[:, x1:].any(),
            'Mask extends outside inference crop')
    return bool((x0 > 0 and mask[y0:y1, x0].any()) or (x1 < width and mask[y0:y1, x1-1].any())
                or (y0 > 0 and mask[y0, x0:x1].any()) or (y1 < height and mask[y1-1, x0:x1].any()))


def check_object(obj, csv_row, base, corridor, support, manifest, ordinal, generation, track_history,
                 baseline_manifest, baseline_objects, tile_views, quarantined=False):
    width, height = base['image_width'], base['image_height']
    prefix = 'q' if quarantined else 'o'
    require(obj['observation_id'] == f"{base['video_id']}:v2:f{base['frame_index']:06d}:{prefix}{ordinal:03d}", 'Invalid V2 observation ID')
    require(integer(obj['display_number']) == ordinal + 1, 'Wrong display number')
    source, cid = obj['model_source'], integer(obj['class_id'])
    names = manifest['model_names'].get(source)
    require(isinstance(names, dict) and names.get(str(cid)) == obj['class_name'], 'Class does not match frozen model dictionary')
    score = finite(obj['confidence_score'])
    origin = obj['inference_origin']
    require(origin in ('baseline', 'tile'), 'Unknown inference origin')
    candidate_id = obj['candidate_id']
    require(isinstance(candidate_id, str) and re.fullmatch(('b' if origin == 'baseline' else 't') + r'\d+', candidate_id), 'Invalid candidate provenance ID')
    minimum = finite(baseline_manifest['parameters'][f'{source}_conf']) if origin == 'baseline' else finite(manifest['parameters']['conf'])
    require(minimum <= score <= 1, 'Candidate confidence outside inference threshold')
    for key in ('mask_quality_score', 'distance_m', 'actual_error_px', 'physical_danger_confirmed'):
        require(obj[key] is None and csv_row[key] == '', f'{key}: unknown must remain null/empty')
    full = decode_contours(obj['full_predicted_mask_contours'], width, height)
    inside = decode_contours(obj['corridor_intersection_contours'], width, height)
    require(np.array_equal(inside, full & corridor), 'Intersection differs from actual mask AND corridor')
    require(mask_sha(full) == obj['full_mask_sha256'] and mask_sha(inside) == obj['intersection_mask_sha256'], 'Mask hash mismatch')
    analysis = independent_analysis(full, corridor, support)
    require(set(obj['analysis']) == set(analysis), 'Unexpected analysis schema')
    for key, value in analysis.items():
        same_value(obj['analysis'][key], value, 'analysis ' + key)
        if key not in ('mask_valid', 'physical_intrusion_assessed'):
            same_value(csv_row[key], value, 'OBJETS ' + key, True)
    yy, xx = np.nonzero(full)
    bbox = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    require(obj['mask_bbox'] == bbox, 'Mask extent mismatch')
    for key, value in zip(('mask_x_min_px', 'mask_y_min_px', 'mask_x_max_px_exclusive', 'mask_y_max_px_exclusive'), bbox):
        same_value(csv_row[key], value, key, True)
    if origin == 'baseline':
        old_id = obj['baseline_observation_id']
        require(old_id in baseline_objects, 'Baseline observation is missing')
        require(candidate_id == f'b{list(baseline_objects).index(old_id):04d}', 'Baseline candidate ID does not identify its original observation')
        old = baseline_objects[old_id]
        for key in ('model_source', 'class_id', 'class_name', 'confidence_score', 'full_mask_sha256',
                    'full_predicted_mask_contours', 'mask_bbox', 'model_bbox_original_px', 'inference_crop_xyxy', 'touches_crop_border', 'raw_detection_index'):
            require(obj[key] == old[key], 'Baseline prediction changed: ' + key)
        require(obj['tile_index'] is None and obj['view_id'] == 'baseline' and obj['view_kind'] == 'global', 'Baseline has wrong view provenance')
    else:
        require(source == 'litter' and obj['baseline_observation_id'] is None, 'Tile has wrong model or baseline reference')
        require(candidate_id == f"t{integer(obj['raw_detection_index']):04d}", 'Tile candidate ID differs from its raw index')
        require(integer(obj['tile_detection_index']) < 100, 'Tile detection index exceeds per-view limit')
        view_id = obj['view_id']
        require(view_id in tile_views, 'Tile view missing from frame audit')
        view = tile_views[view_id]
        for key in ('view_kind', 'tile_index'):
            require(obj[key] == view[key], 'Tile provenance mismatch: ' + key)
        require(obj['inference_crop_xyxy'] == view['crop_xyxy'], 'Object crop differs from its declared tile')
    require(obj['touches_crop_border'] is crop_border(full, obj['inference_crop_xyxy']), 'Crop-border flag wrong')
    model_box = [finite(x) for x in obj['model_bbox_original_px']]
    require(len(model_box) == 4 and 0 <= model_box[0] < model_box[2] <= width and 0 <= model_box[1] < model_box[3] <= height,
            'Model box outside image')
    flagged, geometry = independent_train_review(obj, full, corridor)
    require(flagged is quarantined, 'Ambiguous train gate incorrectly applied')
    if quarantined:
        require(obj['review_reason'] == csv_row['review_reason'] == 'ambiguous_train_tracklike', 'Wrong quarantine reason')
        require(set(obj['review_geometry']) == set(geometry), 'Wrong quarantine geometry schema')
        for key, value in geometry.items():
            same_value(obj['review_geometry'][key], value, 'review geometry ' + key)
        require(obj['track_id'] is None and obj['observed_hits'] == 0 and obj['track_confirmed'] is False, 'Quarantine must not create accepted-object tracks')
    else:
        require(obj.get('review_reason') in (None, ''), 'Accepted object has rejection reason')
    hits = integer(obj['observed_hits'])
    require(obj['track_confirmed'] is (hits >= 2), 'Wrong tracker confirmation')
    track = obj['track_id']
    if track is None:
        require(hits == 0, 'Unassociated mask has hits')
    else:
        require(isinstance(track, str), 'Track ID must be string')
        parts = track.split(':')
        require(len(parts) == 4 and parts[0] == str(generation) and parts[1] == source and parts[2] == str(cid), 'Wrong track namespace')
        integer(parts[3], 'track ID', 1)
        require(hits >= 1, 'Associated track lacks observed hit')
        if track in track_history:
            prior_frame, prior_hits = track_history[track]
            require(prior_frame < base['frame_index'] and prior_hits < hits, 'Repeated track or non-increasing hits')
        track_history[track] = base['frame_index'], hits
    fields = ('observation_id', 'track_id', 'track_confirmed', 'observed_hits', 'model_source', 'class_id', 'class_name',
              'confidence_score', 'touches_crop_border', 'full_mask_sha256', 'intersection_mask_sha256')
    for key in fields:
        same_value(csv_row[key], obj[key], 'OBJETS ' + key, True)
    same_value(csv_row['full_contour_count'], len(obj['full_predicted_mask_contours']), 'full contour count', True)
    same_value(csv_row['intersection_contour_count'], len(obj['corridor_intersection_contours']), 'intersection contour count', True)
    return full, inside

def check_point_rows(points, obj, base):
    for kind, key in (("full_predicted_mask", "full_predicted_mask_contours"),
                      ("corridor_intersection", "corridor_intersection_contours")):
        for contour in obj[key]:
            for index, (x, y) in enumerate(contour["points"]):
                row = points.pop()
                common_check(row, base, "POINTS_OBJETS")
                expected = {key: obj[key] for key in ("observation_id", "track_id", "model_source", "class_name", "confidence_score")}
                expected.update({key: contour[key] for key in ("contour_id", "component_id", "parent_contour_id", "is_hole", "depth", "closed", "boundary_kind")})
                expected.update(geometry_kind=kind, point_index=index, x_px=x, y_px=y)
                for key, value in expected.items():
                    same_value(row[key], value, "POINTS " + key, True)


def verify_delivery(folder, report_path=None):
    folder = Path(folder).resolve(strict=True)
    report_path = Path(report_path) if report_path else folder / 'VERIFICATION.json'
    start = time.perf_counter()
    report = dict(status='running', delivery=str(folder), started_at=datetime.now(timezone.utc).isoformat(),
                  verifier=str(Path(__file__).resolve()), verifier_sha256=sha(__file__),
                  scope='Independent V2 export, timing, geometry, tile provenance and quarantine verification; no model accuracy claim.',
                  frames_checked=0, objects_checked=0, quarantined_objects_checked=0, point_rows_checked=0)
    try:
        mp = folder / 'MANIFESTE.json'; mh = sha(mp); m = read_json(mp)
        require(m['status'] == 'complete' and m['schema_version'] == 2, 'Complete V2 manifest required')
        require(m['rails_recomputed'] is False, 'Rails must not be recomputed')
        require(set(OUTPUTS) == set(m['output_sha256']), 'Manifest output set differs from V2 contract')
        for name in OUTPUTS:
            require(sha(folder / name) == m['output_sha256'][name], 'Changed output: ' + name)
        for path, digest in m['inputs_sha256'].items():
            require(Path(path).is_absolute() and sha(path) == digest, 'Changed input: ' + path)
        bp = Path(m['parameters']['baseline']).resolve(strict=True)
        bm = read_json(bp / 'MANIFESTE.json'); bv = read_json(bp / 'VERIFICATION.json')
        require(bm['status'] == 'complete' and bm['schema_version'] == 1 and not bm['partial_source'] and bv['status'] == 'passed', 'Verified full V1 baseline required')
        require(sha(bp / 'MANIFESTE.json') == bv['manifest_sha256'] == m['baseline_manifest_sha256'], 'Baseline manifest proof mismatch')
        require(bm['model_names'] == m['model_names'], 'Model class dictionaries changed')
        for name, digest in bm['output_sha256'].items():
            require(sha(bp / name) == digest, 'Baseline output changed: ' + name)
        baseline_txt_hash = baseline_txt_proof(bp, bm, bv)
        require(baseline_txt_hash == m['baseline_txt_path_and_content_aggregate_sha256'], 'V2 baseline TXT proof mismatch')
        model_path = Path(m['parameters']['litter_model']).resolve(strict=True)
        original_model_path = str(Path(bm['parameters']['litter_model']).resolve())
        require(sha(model_path) == bm['inputs_sha256'][original_model_path], 'Litter checkpoint changed from controlled baseline')
        source = Path(m['source']).resolve(strict=True)
        require(sha(source) == m['source_sha256'] == bm['source_sha256'], 'Native source mismatch')
        require(m['video_id'] == bm['video_id'], 'Video ID changed')
        first_baseline_frame = read_json(bp / 'TXT/frame_000000.txt')
        sm = dict(frames=bm['frames'], width=first_baseline_frame['image_width'], height=first_baseline_frame['image_height'],
                  axis_export_sha256=bm['output_sha256']['RAILS.csv'])
        source_rail = bp / 'RAILS.csv'; source_frames = bp / 'FRAMES.csv'
        required_inputs = [source, model_path] + [bp / n for n in ('MANIFESTE.json', 'VERIFICATION.json', 'RAILS.csv', 'FRAMES.csv', 'OBJETS.csv')]
        for path in required_inputs:
            require(str(path.resolve()) in m['inputs_sha256'], 'Required input not frozen: ' + str(path))
        require(finite(m['parameters']['conf']) == finite(bm['parameters']['litter_conf']), 'Tile confidence differs from frozen baseline threshold')
        count = integer(m['frames'], 'frames', 1); total = integer(bm['frames'], 'baseline frames', 1)
        require(total == integer(sm['frames']), 'Baseline does not cover entire original source')
        limit = integer(m['parameters']['max_frames'])
        require(count == (min(limit, total) if limit else total), 'Wrong requested frame prefix')
        full = count == total
        require(m['partial_source'] is (not full) and m['rail_csv_byte_identical'] is full, 'Wrong partial-source/rail identity flags')
        if full:
            require(sha(folder / 'RAILS.csv') == sha(bp / 'RAILS.csv') == sha(source_rail) == sm['axis_export_sha256'], 'Full rail CSV changed')
        files = list((folder / 'TXT').iterdir())
        require(len(files) == count and {x.name for x in files} == {f'frame_{i:06d}.txt' for i in range(count)} and all(x.is_file() for x in files), 'V2 TXT file set mismatch')
        txt_digest = hashlib.sha256(); track_history = {}; prior_t = None; generation = 0
        holes = components = zero = rail_count = 0
        with ExitStack() as stack:
            def table(path, label):
                return CSVStream(stack.enter_context(path.open(newline='', encoding='utf-8')), label)
            frames = table(folder / 'FRAMES.csv', 'FRAMES')
            objects = table(folder / 'OBJETS.csv', 'OBJETS')
            reviews = table(folder / 'A_VERIFIER.csv', 'A_VERIFIER')
            points = table(folder / 'POINTS_OBJETS.csv', 'POINTS_OBJETS')
            rails = table(folder / 'RAILS.csv', 'RAILS')
            oldframes = table(bp / 'FRAMES.csv', 'baseline FRAMES')
            oldrails = table(bp / 'RAILS.csv', 'baseline RAILS')
            sourceframe = table(source_frames, 'source FRAMES')
            sourcerails = table(source_rail, 'source RAILS')
            require(rails.fields == oldrails.fields == sourcerails.fields, 'Rail schema changed')
            require(reviews.fields == objects.fields + ['review_reason'], 'Quarantine table schema differs')
            audits = stack.enter_context((folder / 'AUDIT_FUSION.jsonl').open(encoding='utf-8'))
            sc = stack.enter_context(av.open(str(source)))
            vc = stack.enter_context(av.open(str(folder / 'VIDEO_GABARIT_2D.mp4')))
            require(len(vc.streams.video) == 1 and len(vc.streams.audio) == 0, 'Expected exactly one silent video stream')
            natives = iter(sc.decode(sc.streams.video[0])); rendered = iter(vc.decode(vc.streams.video[0]))
            for index in range(count):
                nf, vf = next(natives, None), next(rendered, None)
                require(nf is not None and vf is not None and nf.pts is not None and vf.pts is not None, 'Missing source/rendered frame or timestamp')
                nt = Fraction(nf.pts) * nf.time_base; t = float(nt)
                require(prior_t is None or nt > prior_t, 'Source PTS not increasing')
                if prior_t is not None and nt - prior_t > Fraction(1, 2): generation += 1
                prior_t = nt
                target = Fraction(round(t * 1_000_000), 1_000_000)
                require(abs(Fraction(vf.pts) * vf.time_base - target) <= max(Fraction(1, 1_000_000), vf.time_base / 2), 'Rendered video shifted or resampled')
                width, height = nf.width, nf.height
                require((width, height) == (vf.width, vf.height) == (sm['width'], sm['height']), 'Frame dimensions changed')
                base = dict(video_id=m['video_id'], frame_index=index, time_video_s=t, source_pts=nf.pts,
                            time_base_num=nf.time_base.numerator, time_base_den=nf.time_base.denominator,
                            image_width=width, image_height=height)
                fr, oldfr, sfr = frames.pop(), oldframes.pop(), sourceframe.pop()
                common_check(fr, base, 'FRAMES'); common_check(oldfr, base, 'baseline FRAMES'); common_check(sfr, base, 'source FRAMES', True)
                require(fr['processing_status'] == 'ok', 'Frame not marked ok')
                rr, brr, srr = rails.frame(index), oldrails.frame(index), sourcerails.frame(index)
                require(rr == brr == srr, 'Rail rows changed')
                rail_count += len(rr)
                for row in rr: common_check(row, base, 'RAILS', True)
                corridor, support, ri = independent_corridor(rr, width, height)
                require(ri['supported_rows'] == integer(sfr['rail_supported_rows']) and ri['corridor_pixel_count'] == integer(sfr['rail_corridor_pixel_count']), 'Baseline rail support counts changed')
                same_value(fr['rail_supported_rows'], ri['supported_rows'], 'supported rows', True)
                same_value(fr['rail_corridor_pixel_count'], ri['corridor_pixel_count'], 'corridor pixels', True)
                require(fr['rail_diagnostic_status'] == oldfr['rail_diagnostic_status'] == sfr['rail_diagnostic_status'], 'Rail diagnostic changed')
                rel = f'TXT/frame_{index:06d}.txt'
                require(fr['txt_file'] == rel, 'Wrong frame TXT reference')
                data = (folder / rel).read_bytes(); txt_digest.update(rel.encode() + b'\0' + hashlib.sha256(data).digest())
                payload = json_text(data.decode('utf-8')); old_payload = read_json(bp / rel)
                common_check(payload, base, 'TXT'); common_check(old_payload, base, 'baseline TXT')
                require(payload['schema_version'] == 2 and payload['processing_status'] == 'ok', 'Wrong TXT schema/status')
                require(payload['coordinate_convention'] == 'original pixel centres, x right, y down', 'Wrong coordinate convention')
                require(payload['rail_support'] == ri and payload['rail_diagnostic_status'] == sfr['rail_diagnostic_status'], 'TXT rail metadata changed')
                baseline = {o['observation_id']: o for o in old_payload['objects']}
                require(len(baseline) == len(old_payload['objects']) == integer(oldfr['overlap_candidates']) == integer(fr['baseline_candidates']), 'Wrong baseline candidate count')
                line = next(audits, None); require(line is not None, 'Audit ended early')
                audit = json_text(line); common_check(audit, base, 'AUDIT')
                expected_views = expected_tile_views(corridor, m['parameters']['tile_size'], m['parameters']['tile_overlap'])
                require(audit['tile_views'] == expected_views, 'Tile grid differs from declared recipe')
                require(integer(fr['tile_views']) == len(expected_views), 'Tile view count mismatch')
                views = {v['view_id']: v for v in expected_views}
                retained, quarantined = payload['objects'], payload['quarantined_predictions']
                require(isinstance(retained, list) and isinstance(quarantined, list), 'Object groups must be lists')
                orows, qrows = objects.frame(index), reviews.frame(index)
                require(len(retained) == len(orows) == integer(fr['overlap_candidates']), 'Retained count mismatch')
                require(len(quarantined) == len(qrows) == integer(fr['quarantined_candidates']), 'Quarantine count mismatch')
                if not retained: zero += 1
                seen = set()
                for is_review, group, rows in ((False, retained, orows), (True, quarantined, qrows)):
                    for ordinal, (obj, row) in enumerate(zip(group, rows)):
                        common_check(row, base, 'object table')
                        require(row['txt_file'] == rel, 'Wrong object TXT reference')
                        require(obj['candidate_id'] not in seen, 'Candidate repeated in retained/quarantined groups')
                        seen.add(obj['candidate_id'])
                        check_object(obj, row, base, corridor, support, m, ordinal, generation, track_history,
                                     bm, baseline, views, quarantined=is_review)
                        if not is_review: check_point_rows(points, obj, base)
                        for key in ('full_predicted_mask_contours', 'corridor_intersection_contours'):
                            holes += sum(c['is_hole'] for c in obj[key]); components += sum(not c['is_hole'] for c in obj[key])
                if points.pending is not None:
                    require(integer(points.pending['frame_index']) > index, 'Extra/misassigned point rows, possibly quarantine leaked into main points')
                require(integer(fr['confirmed_overlap_candidates']) == sum(o['track_confirmed'] for o in retained), 'Accepted tracked count mismatch')
                raw, suppressed = audit['merged_predictions'], audit['suppressed']
                require(isinstance(raw, list) and isinstance(suppressed, list), 'Malformed fusion audit')
                public_count = sum(o['model_source'] == 'public' for o in baseline.values())
                litter_count = len(baseline) - public_count
                tile_count = integer(fr['tile_instances'])
                require(integer(fr['raw_public_instances']) == public_count, 'Wrong reused public count')
                require(integer(fr['raw_litter_instances']) == litter_count + tile_count, 'Wrong global+tile litter count')
                require(len(raw) == integer(fr['merged_instances']) == len(baseline) + tile_count - len(suppressed), 'Fusion counts inconsistent')
                require(len(suppressed) == integer(fr['suppressed_duplicate_instances']), 'Suppression count mismatch')
                require(integer(audit['empty_masks']) == integer(fr['empty_mask_instances']), 'Empty-mask count mismatch')
                require(sum(r['model_source'] == 'public' for r in raw) == public_count, 'Public predictions unexpectedly removed before quarantine')
                require(sum(r['relation_2d'] == 'indeterminate' for r in raw) == integer(fr['indeterminate_instances']), 'Indeterminate count mismatch')
                raw_ids = [r['candidate_id'] for r in raw]
                require(len(raw_ids) == len(set(raw_ids)), 'Duplicate raw candidate IDs')
                all_ids = {f'b{i:04d}' for i in range(len(baseline))} | {f't{i:04d}' for i in range(tile_count)}
                suppressed_ids = [s['discarded_candidate_id'] if s['rule'] == 'public_mask_precedence_for_cross_model_duplicate' else s['suppressed_candidate_id'] for s in suppressed]
                require(len(suppressed_ids) == len(set(suppressed_ids)), 'Candidate suppressed multiple times')
                require(set(raw_ids).isdisjoint(suppressed_ids) and set(raw_ids) | set(suppressed_ids) == all_ids, 'Candidate lineage lost, invented or repeated')
                by_id = {r['candidate_id']: r for r in raw}
                overlaps = {r['candidate_id'] for r in raw if r['relation_2d'] == 'overlap_observed'}
                require(overlaps == seen, 'An observed overlap was dropped or invented')
                for r in raw:
                    bounds = r['mask_bbox']; require(isinstance(bounds, list) and len(bounds) == 4, 'Invalid audit bounds')
                    x0,y0,x1,y1 = [integer(v) for v in bounds]
                    require(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, 'Audit bounds outside image')
                    require(integer(r['bbox_overlap_pixel_count']) == int(corridor[y0:y1, x0:x1].sum()), 'Wrong audit rectangle overlap')
                    require(r['inference_origin'] in ('baseline', 'tile'), 'Unknown audit origin')
                    require(r['candidate_id'].startswith('b' if r['inference_origin'] == 'baseline' else 't'), 'Audit candidate ID/origin mismatch')
                for is_review, group in ((False, retained), (True, quarantined)):
                    for obj in group:
                        r = by_id[obj['candidate_id']]
                        for key in ('candidate_id','inference_origin','model_source','raw_detection_index','class_name','confidence_score','mask_bbox'):
                            require(obj[key] == r[key], 'Audit/object mismatch: ' + key)
                        for key in ('relation_2d','overlap_pixel_count','support_status'):
                            require(obj['analysis'][key] == r[key], 'Audit/analysis mismatch: ' + key)
                        require(r.get('review_reason') == ('ambiguous_train_tracklike' if is_review else None), 'Wrong audit quarantine decision')
                for s in suppressed:
                    rule = s['rule']
                    iou = finite(s['mask_iou'])
                    require(0 <= iou <= 1, 'Invalid duplicate IoU')
                    if rule == 'public_mask_precedence_for_cross_model_duplicate':
                        require(iou > .6, 'Wrong cross-model suppression threshold')
                        kept = s['kept_candidate_id']
                        require(kept in by_id and by_id[kept]['model_source'] == 'public', 'Cross-model suppression lacks retained public reference')
                        require(by_id[kept].get('review_reason') != 'ambiguous_train_tracklike', 'Quarantined train must not suppress litter')
                        discarded = s['discarded_candidate_id']
                    elif rule in ('cross_view_mask_iou','cross_view_truncated_mask_ios'):
                        kept, discarded = s['kept_candidate_id'], s['suppressed_candidate_id']
                        require(s['suppressed_view_id'] != s['kept_view_id'], 'Same-view objects suppressed by multiscale merge')
                        ios = finite(s['mask_ios']); require(0 <= ios <= 1, 'Invalid duplicate IoS')
                        if rule == 'cross_view_mask_iou': require(iou > .5, 'Wrong multi-view IoU rule')
                        else: require(iou <= .5 and ios > .8 and (s['suppressed_touches_crop_border'] or s['kept_touches_crop_border']), 'Wrong truncated-view IoS rule')
                    else: raise ValueError('Unknown suppression rule: ' + rule)
                    require(kept in all_ids and discarded in all_ids and kept != discarded, 'Invalid suppression candidate references')
                report.update(frames_checked=index+1, objects_checked=objects.count,
                              quarantined_objects_checked=reviews.count, point_rows_checked=points.count)
                if index == 0: same_value(m['first_time_s'], t, 'First time')
                if index == count-1: same_value(m['last_time_s'], t, 'Last time')
                if (index+1) % 200 == 0: print(json.dumps({'verified_frames':index+1,'objects':objects.count,'quarantined':reviews.count,'points':points.count}),flush=True)
            require(next(rendered, None) is None, 'Extra rendered frames')
            if full:
                require(next(natives, None) is None and oldframes.pending is None and oldrails.pending is None and sourceframe.pending is None and sourcerails.pending is None, 'Full source contains unmatched frames or rail rows')
            require(next(audits, None) is None, 'Extra audit frame')
            for stream in (frames, objects, reviews, points, rails): require(stream.pending is None, 'Unconsumed output rows: ' + stream.label)
            require(objects.count == integer(m['overlap_observations']) and reviews.count == integer(m['quarantined_observations']) and points.count == integer(m['point_rows']), 'Manifest final counts mismatch')
        require(sha(mp) == mh, 'Manifest changed during verification')
        for name in OUTPUTS: require(sha(folder / name) == m['output_sha256'][name], 'Output changed while verifying: ' + name)
        require(baseline_txt_proof(bp, bm, bv) == baseline_txt_hash, 'Baseline changed while verifying')
        report.update(status='passed', completed_at=datetime.now(timezone.utc).isoformat(), elapsed_s=time.perf_counter()-start,
                      source_sha256=m['source_sha256'], manifest_sha256=mh, txt_files_checked=count,
                      txt_path_and_content_aggregate_sha256=txt_digest.hexdigest(), rail_rows_checked=rail_count,
                      original_rail_byte_identical=full, source_and_rendered_pts_checked=count,
                      baseline_txt_path_and_content_aggregate_sha256=baseline_txt_hash,
                      zero_candidate_frames=zero, contour_holes_checked=holes, contour_components_checked=components,
                      all_candidate_masks_reconstructed=True, all_quarantined_masks_reconstructed=True,
                      all_intersections_recomputed=True, all_point_rows_checked=True, original_pixel_bounds_checked=True,
                      quarantine_guard_independently_recomputed=True, tile_grid_independently_recomputed=True,
                      unknown_distance_and_error_remain_null=True,
                      limitations=['Does not establish segmentation accuracy, danger or absence of obstacles.',
                                   'Suppressed and non-overlap masks are not exported; their logged summaries/counts are checked, not their full predicted masks.',
                                   'Baseline inference is reused from its frozen completed verification, not rerun.'])
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', stopped_at=datetime.now(timezone.utc).isoformat(),
                      elapsed_s=time.perf_counter()-start, error=dict(type=type(exc).__name__, message=str(exc)))
        temp = report_path.with_name(report_path.name+'.tmp'); temp.write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+'\n');temp.replace(report_path)
        raise
    temp = report_path.with_name(report_path.name+'.tmp');temp.write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+'\n');temp.replace(report_path)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--delivery',type=Path,required=True);p.add_argument('--report',type=Path)
    a=p.parse_args();cv2.setNumThreads(0);r=verify_delivery(a.delivery,a.report)
    print(json.dumps({k:r[k] for k in ('status','frames_checked','objects_checked','quarantined_objects_checked','point_rows_checked')}),flush=True)

if __name__=='__main__':main()
