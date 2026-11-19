#!/usr/bin/env python3
"""Independent, streaming verification of an infer_gabarit_2d.py delivery.

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
           "RAILS.csv", "AUDIT_FUSION.jsonl")


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


def check_object(obj, csv_row, base, corridor, support, manifest, ordinal, generation, track_history):
    width, height = base["image_width"], base["image_height"]
    require(obj["observation_id"] == f"{base['video_id']}:f{base['frame_index']:06d}:o{ordinal:03d}", "Invalid observation ID")
    require(integer(obj["display_number"]) == ordinal + 1, "Wrong display number")
    source, cid = obj["model_source"], integer(obj["class_id"])
    names = manifest["model_names"].get(source)
    require(isinstance(names, dict) and names.get(str(cid)) == obj["class_name"], "Class does not match frozen model dictionary")
    score = finite(obj["confidence_score"])
    minimum = finite(manifest["parameters"][f"{source}_conf"])
    require(minimum <= score <= 1, "Candidate confidence outside inference threshold")
    for key in ("mask_quality_score", "distance_m", "actual_error_px", "physical_danger_confirmed"):
        require(obj[key] is None and csv_row[key] == "", f"{key}: unknown must be null/empty, never zero")
    full = decode_contours(obj["full_predicted_mask_contours"], width, height)
    inside = decode_contours(obj["corridor_intersection_contours"], width, height)
    require(np.array_equal(inside, full & corridor), "Intersection mask differs from real mask AND observed corridor")
    require(mask_sha(full) == obj["full_mask_sha256"], "Full mask SHA-256 mismatch")
    require(mask_sha(inside) == obj["intersection_mask_sha256"], "Intersection mask SHA-256 mismatch")
    analysis = independent_analysis(full, corridor, support)
    require(set(obj["analysis"]) == set(analysis), "Unexpected mask analysis schema")
    for key, value in analysis.items():
        same_value(obj["analysis"][key], value, "analysis " + key)
        if key not in ("mask_valid", "physical_intrusion_assessed"):
            same_value(csv_row[key], value, "OBJETS " + key, csv_value=True)
    ys, xs = np.nonzero(full)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    require(obj["mask_bbox"] == bbox, "Mask bounding extent mismatch")
    for key, value in zip(("mask_x_min_px", "mask_y_min_px", "mask_x_max_px_exclusive", "mask_y_max_px_exclusive"), bbox):
        same_value(csv_row[key], value, key, True)
    crop = obj["inference_crop_xyxy"]
    if source == "public":
        require(crop is None and obj["touches_crop_border"] is False, "Public mask unexpectedly cropped")
    else:
        require(source == "litter", "Unknown model source")
        require(crop == expected_crop(corridor, finite(manifest["parameters"]["litter_crop_padding"])), "Litter crop differs from declared corridor crop")
        x0, y0, x1, y1 = crop
        require(not full[:y0].any() and not full[y1:].any() and not full[:, :x0].any() and not full[:, x1:].any(), "Litter mask extends outside inference crop")
        touch = bool((x0 > 0 and full[y0:y1, x0].any()) or (x1 < width and full[y0:y1, x1 - 1].any()) or
                     (y0 > 0 and full[y0, x0:x1].any()) or (y1 < height and full[y1 - 1, x0:x1].any()))
        require(obj["touches_crop_border"] is touch, "Wrong crop-border indicator")
    model_box = [finite(x) for x in obj["model_bbox_original_px"]]
    require(len(model_box) == 4 and 0 <= model_box[0] < model_box[2] <= width and 0 <= model_box[1] < model_box[3] <= height,
            "Model box is not in original-image bounds")
    hits = integer(obj["observed_hits"])
    require(obj["track_confirmed"] is (hits >= 2), "Wrong tracker confirmation")
    track = obj["track_id"]
    if track is None:
        require(hits == 0, "Unassociated mask has tracker hits")
    else:
        require(isinstance(track, str), "Track ID must be a string or null")
        parts = track.split(":")
        require(len(parts) == 4 and parts[0] == str(generation) and parts[1] == source and parts[2] == str(cid), "Track generation/source/class mismatch")
        integer(parts[3], "track ID", 1)
        require(hits >= 1, "Associated mask has no observation")
        # The export contains only overlap candidates. Earlier provisional
        # tracker observations and observations outside the corridor can already
        # contribute hits: the first exported ID need not start at one, and
        # later exported counts need not increase by exactly one.
        if track in track_history:
            prior_frame, prior_hits = track_history[track]
            require(prior_frame < base["frame_index"] and prior_hits < hits, "Repeated track in frame or non-increasing observed hits")
        track_history[track] = base["frame_index"], hits
    fields = ("observation_id", "track_id", "track_confirmed", "observed_hits", "model_source", "class_id", "class_name",
              "confidence_score", "touches_crop_border", "full_mask_sha256", "intersection_mask_sha256")
    for key in fields:
        same_value(csv_row[key], obj[key], "OBJETS " + key, True)
    same_value(csv_row["full_contour_count"], len(obj["full_predicted_mask_contours"]), "full contour count", True)
    same_value(csv_row["intersection_contour_count"], len(obj["corridor_intersection_contours"]), "intersection contour count", True)


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
    report_path = Path(report_path) if report_path else folder / "VERIFICATION.json"
    started = time.perf_counter()
    report = {"status": "running", "delivery": str(folder), "started_at": datetime.now(timezone.utc).isoformat(),
              "verifier": str(Path(__file__).resolve()), "verifier_sha256": sha(__file__),
              "scope": "Complete export/geometry/timing verification. No model inference, model accuracy or physical danger validation.",
              "frames_checked": 0, "objects_checked": 0, "point_rows_checked": 0}
    try:
        manifest_path = folder / "MANIFESTE.json"
        manifest_hash = sha(manifest_path)
        manifest = read_json(manifest_path)
        require(manifest["status"] == "complete" and manifest["schema_version"] == 1, "Complete supported inference manifest required")
        count = integer(manifest["frames"], "manifest frames", 1)
        require(manifest["rails_recomputed"] is False, "Rails must be reused")
        require(set(OUTPUTS) == set(manifest["output_sha256"]), "Manifest output hash set differs from required delivery")
        for name in OUTPUTS:
            require(sha(folder / name) == manifest["output_sha256"][name], f"Output SHA-256 mismatch: {name}")
        for name, digest in manifest["inputs_sha256"].items():
            require(Path(name).is_absolute() and sha(name) == digest, f"Input changed: {name}")
        source = Path(manifest["source"]).resolve(strict=True)
        require(sha(source) == manifest["source_sha256"], "Source video hash mismatch")
        rail_folder = Path(manifest["parameters"]["rail_delivery"]).resolve(strict=True)
        source_meta_path = rail_folder / "SOURCE_ET_PARAMETRES.json"
        source_meta = read_json(source_meta_path)
        source_rail = rail_folder / "AXE_ET_LIMITES.csv"
        source_frames = rail_folder / "FRAMES.csv"
        require(source_meta["status"] == "complete", "Incomplete original rail delivery")
        require(Path(manifest["parameters"].get("source") or source_meta["source"]).resolve() == source and source_meta["source_sha256"] == manifest["source_sha256"], "Wrong rail/video pairing")
        for path in (source, source_meta_path, source_rail, source_frames):
            require(str(path.resolve()) in manifest["inputs_sha256"], f"Required input not frozen: {path}")
        full = count == integer(source_meta["frames"], "source rail frames", 1)
        require(count <= int(source_meta["frames"]) and manifest["partial_source"] is (not full), "Partial-source declaration mismatch")
        limit = integer(manifest["parameters"]["max_frames"])
        require(count == (min(limit, int(source_meta["frames"])) if limit else int(source_meta["frames"])), "Frame count does not match requested prefix")
        require(manifest["rail_csv_byte_identical"] is full, "Rail byte-identity declaration mismatch")
        if full:
            require(sha(folder / "RAILS.csv") == sha(source_rail) == source_meta["axis_export_sha256"], "Full rail export is not byte-identical to source")
        txt_count = 0
        for path in (folder / "TXT").iterdir():
            require(path.is_file() and re.fullmatch(r"frame_\d{6,}\.txt", path.name) is not None, "Unexpected TXT directory entry")
            index = int(path.stem.split("_")[1])
            require(0 <= index < count and path.name == f"frame_{index:06d}.txt", "Unexpected TXT frame name")
            txt_count += 1
        require(txt_count == count, "Expected one TXT for each processed source frame")
        txt_digest = hashlib.sha256()
        track_history, generation, prior_t = {}, 0, None
        holes, components, empty_frames, rail_rows_total = 0, 0, 0, 0
        with ExitStack() as stack:
            def table(path, name):
                return CSVStream(stack.enter_context(path.open(newline="", encoding="utf-8")), name)
            frames = table(folder / "FRAMES.csv", "FRAMES")
            objects = table(folder / "OBJETS.csv", "OBJETS")
            points = table(folder / "POINTS_OBJETS.csv", "POINTS_OBJETS")
            rails = table(folder / "RAILS.csv", "RAILS")
            original_frames = table(source_frames, "source FRAMES")
            original_rails = table(source_rail, "source RAILS")
            require(rails.fields == original_rails.fields, "Rail CSV schema changed")
            audits = stack.enter_context((folder / "AUDIT_FUSION.jsonl").open(encoding="utf-8"))
            source_container = stack.enter_context(av.open(str(source)))
            output_container = stack.enter_context(av.open(str(folder / "VIDEO_GABARIT_2D.mp4")))
            decoded = iter(source_container.decode(source_container.streams.video[0]))
            rendered = iter(output_container.decode(output_container.streams.video[0]))
            for index in range(count):
                native, video = next(decoded, None), next(rendered, None)
                require(native is not None and video is not None, "Source or rendered video ended early")
                require(native.pts is not None and native.time_base is not None, "Source PTS missing")
                require(video.pts is not None and video.time_base is not None, "Rendered PTS missing")
                native_t = Fraction(native.pts) * native.time_base
                t = float(native_t)
                require(prior_t is None or native_t > prior_t, "Non-increasing native PTS")
                if prior_t is not None and native_t - prior_t > Fraction(1, 2):
                    generation += 1
                prior_t = native_t
                encoded_expected = Fraction(round(t * 1_000_000), 1_000_000)
                encoded_actual = Fraction(video.pts) * video.time_base
                tolerance = max(Fraction(1, 1_000_000), video.time_base / 2)
                require(abs(encoded_actual - encoded_expected) <= tolerance, "Rendered video shifted, duplicated or resampled source PTS")
                width, height = native.width, native.height
                require((width, height) == (source_meta["width"], source_meta["height"]) == (video.width, video.height), "Source/rendered dimensions mismatch")
                base = dict(video_id=manifest["video_id"], frame_index=index, time_video_s=t,
                            source_pts=native.pts, time_base_num=native.time_base.numerator,
                            time_base_den=native.time_base.denominator, image_width=width, image_height=height)
                frame, old_frame = frames.pop(), original_frames.pop()
                common_check(frame, base, "FRAMES")
                common_check(old_frame, base, "source FRAMES", True)
                require(frame["processing_status"] == "ok", "Processed frame not marked ok")
                rail_rows, original_rows = rails.frame(index), original_rails.frame(index)
                require(rail_rows == original_rows, "Exported rail rows differ from original source")
                rail_rows_total += len(rail_rows)
                for rr in rail_rows:
                    common_check(rr, base, "RAILS", True)
                corridor, support, rail_info = independent_corridor(rail_rows, width, height)
                require(rail_info["valid_sample_rows"] == integer(old_frame["rail_points_displayed"]), "Source rail display count mismatch")
                require(len(rail_rows) == integer(old_frame["rail_points_raw"]), "Source raw rail count mismatch")
                require(frame["rail_diagnostic_status"] == old_frame["rail_temporal_status"], "Rail diagnostic changed")
                same_value(frame["rail_supported_rows"], rail_info["supported_rows"], "supported rows", True)
                same_value(frame["rail_corridor_pixel_count"], rail_info["corridor_pixel_count"], "rail pixels", True)
                txt_rel = f"TXT/frame_{index:06d}.txt"
                require(frame["txt_file"] == txt_rel, "Wrong frame TXT link")
                txt_bytes = (folder / txt_rel).read_bytes()
                payload = json_text(txt_bytes.decode("utf-8"))
                txt_digest.update(txt_rel.encode() + b"\0" + hashlib.sha256(txt_bytes).digest())
                common_check(payload, base, "TXT")
                require(payload["schema_version"] == 1 and payload["processing_status"] == "ok", "Invalid TXT schema/status")
                require(payload["coordinate_convention"] == "original pixel centres, x right, y down", "Wrong coordinate convention")
                require(payload["rail_diagnostic_status"] == old_frame["rail_temporal_status"], "TXT rail status mismatch")
                require(payload["rail_support"] == rail_info, "TXT corridor metadata disagrees with independent reconstruction")
                candidates = payload["objects"]
                require(isinstance(candidates, list), "TXT objects must be a list, including empty frames")
                candidate_rows = objects.frame(index)
                require(len(candidates) == len(candidate_rows) == integer(frame["overlap_candidates"]), "TXT/CSV/frame object count mismatch")
                if not candidates:
                    empty_frames += 1
                for ordinal, (obj, obj_row) in enumerate(zip(candidates, candidate_rows)):
                    common_check(obj_row, base, "OBJETS")
                    require(obj_row["txt_file"] == txt_rel, "Wrong object TXT link")
                    check_object(obj, obj_row, base, corridor, support, manifest, ordinal, generation, track_history)
                    check_point_rows(points, obj, base)
                    for key in ("full_predicted_mask_contours", "corridor_intersection_contours"):
                        holes += sum(c["is_hole"] for c in obj[key])
                        components += sum(not c["is_hole"] for c in obj[key])
                if points.pending is not None:
                    require(integer(points.pending["frame_index"]) > index, "Unmatched extra point rows for current frame")
                require(integer(frame["confirmed_overlap_candidates"]) == sum(o["track_confirmed"] for o in candidates), "Confirmed candidate count mismatch")
                audit_line = next(audits, None)
                require(audit_line is not None, "Fusion audit ended early")
                audit = json_text(audit_line)
                common_check(audit, base, "AUDIT_FUSION")
                merged, suppressed = audit["merged_predictions"], audit["suppressed"]
                require(isinstance(merged, list) and isinstance(suppressed, list), "Invalid fusion audit lists")
                raw_public, raw_litter = integer(frame["raw_public_instances"]), integer(frame["raw_litter_instances"])
                require(len(merged) == integer(frame["merged_instances"]) == raw_public + raw_litter - len(suppressed), "Fusion counts inconsistent")
                require(len(suppressed) == integer(frame["suppressed_duplicate_instances"]), "Suppression count mismatch")
                require(integer(audit["empty_masks"]) == integer(frame["empty_mask_instances"]), "Empty-mask audit mismatch")
                require(sum(p["model_source"] == "public" for p in merged) == raw_public, "Public instance audit mismatch")
                require(sum(p["model_source"] == "litter" for p in merged) + len(suppressed) == raw_litter, "Litter instance audit mismatch")
                for raw in merged:
                    if "bbox_overlap_pixel_count" in raw:
                        bounds = raw["mask_bbox"]
                        require(isinstance(bounds, list) and len(bounds) == 4, "Invalid audit mask bounding extent")
                        x0, y0, x1, y1 = [integer(v, "audit bbox coordinate") for v in bounds]
                        require(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, "Audit mask bounding extent outside image")
                        require(integer(raw["bbox_overlap_pixel_count"]) == int(corridor[y0:y1, x0:x1].sum()),
                                "Audit rectangle overlap differs from observed corridor")
                overlap = [p for p in merged if p["relation_2d"] == "overlap_observed"]
                require(len(overlap) == len(candidates), "Audit overlap candidates missing from TXT")
                require(sum(p["relation_2d"] == "indeterminate" for p in merged) == integer(frame["indeterminate_instances"]), "Indeterminate count mismatch")
                for candidate, raw in zip(candidates, overlap):
                    for key in ("model_source", "raw_detection_index", "class_name", "confidence_score", "mask_bbox"):
                        require(candidate[key] == raw[key], f"Fusion/object mismatch: {key}")
                    for key in ("relation_2d", "overlap_pixel_count", "support_status"):
                        require(candidate["analysis"][key] == raw[key], f"Fusion/analysis mismatch: {key}")
                for item in suppressed:
                    require(item["rule"] == "public_mask_precedence_for_cross_model_duplicate" and 0.6 < finite(item["mask_iou"]) <= 1,
                            "Invalid suppression rule or recorded IoU")
                report.update(frames_checked=index + 1, objects_checked=objects.count, point_rows_checked=points.count)
                if index == 0:
                    same_value(manifest["first_time_s"], t, "First source time")
                if index == count - 1:
                    same_value(manifest["last_time_s"], t, "Last source time")
                if (index + 1) % 200 == 0:
                    print(json.dumps({"verified_frames": index + 1, "objects": objects.count, "points": points.count}), flush=True)
            require(next(rendered, None) is None, "Rendered video contains extra frames")
            if full:
                require(next(decoded, None) is None and original_frames.pending is None and original_rails.pending is None, "Full source contains unmatched frames/rows")
            require(next(audits, None) is None, "Fusion audit has extra frames")
            for table in (frames, objects, points, rails):
                require(table.pending is None, f"{table.label}: unconsumed rows")
            require(objects.count == integer(manifest["overlap_observations"]), "Manifest object total mismatch")
            require(points.count == integer(manifest["point_rows"]), "Manifest point total mismatch")
        require(sha(manifest_path) == manifest_hash, "Manifest changed during verification")
        for name in OUTPUTS:
            require(sha(folder / name) == manifest["output_sha256"][name], f"Output changed during verification: {name}")
        report.update(status="passed", completed_at=datetime.now(timezone.utc).isoformat(), elapsed_s=time.perf_counter() - started,
                      source_sha256=manifest["source_sha256"], manifest_sha256=manifest_hash,
                      txt_files_checked=txt_count, txt_path_and_content_aggregate_sha256=txt_digest.hexdigest(),
                      rail_rows_checked=rail_rows_total, original_rail_byte_identical=full,
                      source_and_rendered_pts_checked=count, zero_candidate_frames=empty_frames,
                      contour_holes_checked=holes, contour_components_checked=components,
                      all_candidate_masks_reconstructed=True, all_intersections_recomputed=True,
                      all_point_rows_checked=True, original_pixel_bounds_checked=True,
                      unknown_distance_and_error_remain_null=True,
                      limitations=["Does not establish model accuracy or real-world physical danger.",
                                   "Suppressed and non-overlap raw masks are not exported; their recorded summaries/counts are checked, not their model masks."])
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      stopped_at=datetime.now(timezone.utc).isoformat(), elapsed_s=time.perf_counter() - started,
                      error={"type": type(exc).__name__, "message": str(exc)})
        temporary = report_path.with_name(report_path.name + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(report_path)
        raise
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify_delivery(args.delivery, args.report)
    print(json.dumps({key: report[key] for key in ("status", "frames_checked", "objects_checked", "point_rows_checked")}), flush=True)


if __name__ == "__main__":
    main()
