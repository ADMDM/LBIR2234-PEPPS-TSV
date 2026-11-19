"""Explicit CSV synchronization and strict OCR coordinate parsing.

No coordinate is repaired by guessing a digit or assuming a country.
"""
from bisect import bisect_left
from datetime import datetime
import csv
import json
import math
from pathlib import Path
import re
import unicodedata
from pyproj import Geod

GEOD = Geod(ellps="WGS84")


def finite_number(value):
    x = float(str(value).strip().replace(",", "."))
    if not math.isfinite(x):
        raise ValueError("non_finite_number")
    return x


def parse_coordinate_part(text, hemisphere):
    text = unicodedata.normalize("NFKC", text).strip()
    text = text.replace(",", ".").replace("′", "'").replace("″", '"')
    # Some OCR outputs read the isolated final seconds mark as I or |.
    # Drop only this separate terminal mark after a complete three-number DMS;
    # never replace a letter inside a number or infer a missing digit.
    if re.search(r"\s+[I|]$", text) and len(re.findall(r"\d+(?:\.\d+)?", text)) == 3:
        text = re.sub(r"\s+[I|]$", "", text)
    # Permit punctuation used for degrees/minutes/seconds, but no guessed O/0, I/1.
    if re.search(r"[^0-9.\s°º'\"’`:+−-]", text):
        raise ValueError("unexpected_coordinate_character")
    values = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
    if len(values) not in (1, 2, 3):
        raise ValueError("coordinate_component_count")
    numbers = [finite_number(v) for v in values]
    if any(n < 0 for n in numbers):
        raise ValueError("signed_value_with_hemisphere")
    degrees = numbers[0]
    limit = 90 if hemisphere in "NS" else 180
    if len(numbers) > 1:
        if not degrees.is_integer() or not 0 <= numbers[1] < 60:
            raise ValueError("invalid_degrees_or_minutes")
        degrees += numbers[1] / 60
    if len(numbers) == 3:
        if not numbers[1].is_integer() or not 0 <= numbers[2] < 60:
            raise ValueError("invalid_minutes_or_seconds")
        degrees += numbers[2] / 3600
    if not 0 <= degrees <= limit:
        raise ValueError("coordinate_out_of_range")
    return (-1 if hemisphere in "SW" else 1) * degrees


def parse_gps_text(text, coordinate_format="auto"):
    """Explicit N/S then E/W, decimal degrees or degrees-minutes(-seconds)."""
    text = unicodedata.normalize("NFKC", text).upper().replace("\u0415", "E")
    match = re.fullmatch(r"\s*([NS])\s*(.*?)\s*([EW])\s*(.*?)\s*", text)
    if not match:
        raise ValueError("hemisphere_pattern_not_found")
    ns, latitude, ew, longitude = match.groups()
    if coordinate_format == "tsv_dms":
        # The known overlay prints two-digit minutes. Re-establish separators
        # only in this explicit DMS profile; no digit substitution is permitted.
        def format_part(part, degree_width):
            part = part.strip().replace(",", ".")
            pattern = r"^(\d{1,"+str(degree_width)+r"}(?:\.0)?)[\s.°º]+(\d{2})[\s'\"’′°º]*(\d{1,2}(?:\.\d+)?)[\s'\"’′″°ºI|]*$"
            found = re.fullmatch(pattern,part)
            return " ".join(found.groups()) if found else part
        latitude = format_part(latitude,2)
        longitude = format_part(longitude,3)
        if any(len(re.findall(r"\d+(?:\.\d+)?", p)) != 3 for p in [latitude,longitude]):
            raise ValueError("expected_three_DMS_components")
    return parse_coordinate_part(latitude, ns), parse_coordinate_part(longitude, ew)


def extract_gps_candidate(lines, min_confidence=.75, coordinate_format="auto"):
    candidates, errors = [], []
    for line in lines:
        text = line["text"].strip()
        if not re.match(r"^[NS]\s*\d", text.upper()):
            continue
        try:
            latitude, longitude = parse_gps_text(text,coordinate_format)
            confidence = float(line.get("confidence", 0))
            if confidence < min_confidence:
                errors.append("low_ocr_score")
                continue
            candidates.append((latitude, longitude, confidence, text))
        except ValueError as error:
            errors.append(str(error))
    distinct = {(round(c[0], 10), round(c[1], 10)) for c in candidates}
    if len(distinct) > 1:
        return {"status": "rejected_conflicting_coordinates"}
    if not candidates:
        return {"status": "rejected_" + (errors[0] if errors else "gps_text_not_found")}
    latitude, longitude, confidence, text = max(candidates, key=lambda x: x[2])
    clock = next((m.group() for line in lines if (m := re.search(r"\b\d{2}:\d{2}:\d{2}\b", line["text"]))), "")
    return {"status": "parsed_ocr_not_ground_validated", "latitude_deg": latitude,
            "longitude_deg": longitude, "ocr_confidence": confidence, "ocr_coordinate_text": text,
            "overlay_clock_raw": clock}


def reject_spatial_outliers(rows, max_speed_m_s=100, jump_tolerance_m=20, isolated_excess_m=20):
    """Reject impossible steps, keeping the raw audit rows. Thresholds are not accuracy."""
    previous = None
    for row in rows:
        if "latitude_deg" not in row or str(row["status"]).startswith("rejected"):
            continue
        if previous is not None:
            dt = float(row["video_time_s"]) - float(previous["video_time_s"])
            distance = GEOD.inv(previous["longitude_deg"], previous["latitude_deg"], row["longitude_deg"], row["latitude_deg"])[2]
            row["distance_from_previous_accepted_m"] = distance
            if dt <= 0 or distance > max_speed_m_s * dt + jump_tolerance_m:
                row["status"] = "rejected_implausible_gps_step"
                continue
        previous = row
    # Mark an isolated short excursion, not an assumed corrected position.
    # Reversals lasting longer than this short window are retained.
    valid = [r for r in rows if "latitude_deg" in r and not str(r["status"]).startswith("rejected")]
    to_reject = []
    def distance(a,b):
        return GEOD.inv(a["longitude_deg"],a["latitude_deg"],b["longitude_deg"],b["latitude_deg"])[2]
    for a,b,c in zip(valid,valid[1:],valid[2:]):
        if float(c["video_time_s"])-float(a["video_time_s"]) > 3: continue
        excess = distance(a,b)+distance(b,c)-distance(a,c)
        if isolated_excess_m is not None and excess > isolated_excess_m:
            to_reject.append((b,excess))
    for row,excess in to_reject:
        row["status"]="rejected_isolated_short_gps_excursion";row["local_excess_distance_m"]=excess
    return rows


def nearest_frame(frames, time_s, tolerance_s):
    times = [float(f["video_time_s"]) for f in frames]
    pos = bisect_left(times, time_s)
    possible = [i for i in (pos-1, pos) if 0 <= i < len(times)]
    if not possible:
        raise ValueError("empty_video_timeline")
    index = min(possible, key=lambda i: abs(times[i]-time_s))
    if abs(times[index]-time_s) > tolerance_s:
        raise ValueError("no_frame_within_time_tolerance")
    return frames[index], times[index]-time_s


def parse_utc(value):
    date = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError("UTC_timestamp_requires_explicit_timezone")
    return date.timestamp()


def normalize_csv(path, frames, config, video_id, source_sha256):
    """Columns and clock origins are explicit; unknown origins are never guessed."""
    latitude_column = config.get("latitude_column", "latitude_deg")
    longitude_column = config.get("longitude_column", "longitude_deg")
    time_column = config.get("time_column", "video_time_s")
    frame_column = config.get("frame_column")
    tolerance = finite_number(config.get("max_time_delta_s", .1))
    offset = finite_number(config.get("time_offset_s", 0))
    time_scale = finite_number(config.get("input_time_scale", 1))
    reference = config.get("time_reference", "native_pts")
    if reference not in ("native_pts", "video_elapsed", "utc") or time_scale <= 0 or tolerance < 0:
        raise ValueError("invalid_time_configuration")
    first_pts = float(frames[0]["video_time_s"])
    real_per_video = finite_number(config.get("real_seconds_per_video_second", 1))
    if real_per_video <= 0:
        raise ValueError("invalid_time_acceleration")
    epoch = parse_utc(config["video_start_utc"]) if reference == "utc" else None
    frame_by_id = {int(f["frame_index"]): f for f in frames}
    audit, accepted, used = [], [], set()
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file, delimiter=config.get("delimiter", ","))
        required = {latitude_column, longitude_column, time_column}
        if frame_column:
            required.add(frame_column)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV missing columns: " + ", ".join(sorted(required-set(reader.fieldnames or []))))
        previous_input_time = -math.inf
        for line, row in enumerate(reader, 2):
            record = {"source_csv_line": line, "source_row_json": json.dumps(row, ensure_ascii=False), "status": "rejected_unparsed"}
            try:
                latitude = finite_number(row[latitude_column]); longitude = finite_number(row[longitude_column])
                if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                    raise ValueError("coordinate_out_of_range")
                timestamp = parse_utc(row[time_column]) if reference == "utc" else finite_number(row[time_column])
                if timestamp <= previous_input_time:
                    raise ValueError("input_times_not_strictly_increasing")
                previous_input_time = timestamp
                video_time = ((timestamp-epoch)/real_per_video + first_pts if reference == "utc" else
                              timestamp*time_scale + (first_pts if reference == "video_elapsed" else 0)) + offset
                frame, residual = nearest_frame(frames, video_time, tolerance)
                index = int(frame["frame_index"])
                if frame_column and (finite_number(row[frame_column]) != index):
                    raise ValueError("provided_frame_disagrees_with_video_time")
                if row.get("video_id") and row["video_id"] != video_id:
                    raise ValueError("video_id_mismatch")
                if row.get("source_video_sha256") and row["source_video_sha256"] != source_sha256:
                    raise ValueError("video_hash_mismatch")
                if index in used:
                    raise ValueError("multiple_gps_rows_for_same_frame")
                speed = ""
                column = config.get("speed_column")
                if column and row.get(column, "") != "":
                    speed = finite_number(row[column])
                    unit = config.get("speed_unit", "m/s")
                    if unit not in ("m/s", "km/h") or speed < 0:
                        raise ValueError("invalid_speed_or_unit")
                    if unit == "km/h": speed /= 3.6
                result = {"video_id": video_id, "frame_index": index, "video_time_s": float(frame["video_time_s"]),
                          "input_gps_video_time_s": video_time, "time_alignment_residual_s": residual,
                          "latitude_deg": latitude, "longitude_deg": longitude,
                          "gps_source": "provided_csv", "gps_status": "provided_csv_not_independently_validated",
                          "source_video_sha256": source_sha256, "gps_speed_provided_m_s": speed,
                          "gps_speed_status": "provided_csv_not_independently_validated" if speed != "" else "not_provided",
                          "utc_time": row.get(config.get("utc_column", "utc_time"), ""), "source_csv_line": line,
                          "gps_accuracy_m": row.get(config.get("accuracy_column", "gps_accuracy_m"), ""),
                          "input_time_reference": reference}
                used.add(index); accepted.append(result); record.update(result, status="accepted_csv")
            except (ValueError, KeyError) as error:
                record["status"] = "rejected_"+str(error)
            audit.append(record)
    return accepted, audit
