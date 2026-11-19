"""CPU geometry for observed rail profiles and real predicted object masks.

All coordinates are ORIGINAL image pixel centres: x increases to the right,
y increases downwards, and 0 <= x < width, 0 <= y < height. Rail limits are
included. Areas are raster pixel counts, never polygon areas of centre points.

This module neither detects objects nor turns a rectangle into a mask. A 2D
overlap is not a physical clearance, collision probability or danger estimate.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

import cv2
import numpy as np


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite integer") from exc
    if not math.isfinite(number) or not number.is_integer() or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(number)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a coordinate, not a boolean")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _boolean(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value in {"True", "False", "true", "false"}:
        return value.casefold() == "true"
    raise ValueError(f"{name} must be a boolean or CSV True/False")


def _field(row: Mapping[str, Any], primary: str, alias: str) -> Any:
    if primary in row:
        return row[primary]
    if alias in row:
        return row[alias]
    raise ValueError(f"Missing required rail field {primary}")


def _binary(mask: Any, name: str) -> np.ndarray:
    """Accept binary bool, 0/1 and OpenCV 0/255 rasters without thresholding."""
    values = np.asarray(mask)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError(f"{name} must be a nonempty two-dimensional raster")
    if values.dtype.kind not in "buif":
        raise ValueError(f"{name} must be a numeric or boolean binary raster")
    if values.dtype.kind == "f" and not np.isfinite(values).all():
        raise ValueError(f"{name} contains nonfinite values")
    if values.dtype != np.bool_:
        if not (np.isin(values, (0, 1)).all() or np.isin(values, (0, 255)).all()):
            raise ValueError(f"{name} is not binary; threshold model output explicitly")
    return np.ascontiguousarray(values != 0)


def corridor_from_rows(
    rows: Iterable[Mapping[str, Any]], width: int, height: int,
    max_vertical_gap_px: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Rasterize only valid, observed display profiles; return mask/support/meta.

    Input fields are the existing CSV's ``left_display_px``,
    ``right_display_px``, ``y_px``, ``display_valid`` and ``segment_index``.
    The aliases left_display/right_display/y are also accepted. Rows must be
    ordered by increasing integer y, with each segment appearing once. Invalid
    rows require y and segment_index, but their display coordinates may be empty.

    support[y, :] is True only on an observed row or an interpolated row between
    two VALID adjacent samples of the SAME segment. Invalid rows split support;
    no interpolation or extrapolation crosses segment boundaries or large gaps.

    Default gap inference: take the median of the smaller half of consecutive
    within-segment sample spacings, allow twice that spacing, capped at 16 px.
    This conservative project default handles 6/8 px sampling and resists large
    gaps even for a two-row input. For intentionally coarser sampling, pass an
    explicit positive maximum. Every inferred choice and split is in metadata.
    """
    width = _integer(width, "width", 1)
    height = _integer(height, "height", 1)
    parsed: list[dict[str, Any]] = []
    last_y: int | None = None
    last_segment: int | None = None
    seen_segments: set[int] = set()
    spacings: list[int] = []
    for number, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Rail row {number} must be a mapping")
        y = _integer(_field(row, "y_px", "y"), f"row {number} y")
        if y >= height:
            raise ValueError(f"Rail row {number} y is outside the original image")
        sid = _integer(row.get("segment_index"), f"row {number} segment_index")
        valid = _boolean(row.get("display_valid"), f"row {number} display_valid")
        if last_y is not None and y <= last_y:
            raise ValueError("Rail rows must be strictly increasing in y; overlapping segments are ambiguous")
        if sid != last_segment:
            if sid in seen_segments:
                raise ValueError("A rail segment cannot reappear after another segment")
            seen_segments.add(sid)
        elif last_y is not None:
            spacings.append(y - last_y)
        item: dict[str, Any] = {"y": y, "segment_index": sid, "valid": valid}
        if valid:
            left = _finite(_field(row, "left_display_px", "left_display"), f"row {number} left")
            right = _finite(_field(row, "right_display_px", "right_display"), f"row {number} right")
            if not 0 <= left < right < width:
                raise ValueError(f"Rail row {number} limits are reversed or outside original pixels")
            if "axis_display_px" in row and row["axis_display_px"] not in ("", None):
                axis = _finite(row["axis_display_px"], f"row {number} axis")
                if abs(axis - (left + right) / 2) > 1e-6:
                    raise ValueError(f"Rail row {number} axis is not the boundary midpoint")
            item.update(left=left, right=right)
        parsed.append(item)
        last_y, last_segment = y, sid

    characteristic_spacing = None
    if max_vertical_gap_px is None:
        if spacings:
            ordered = sorted(spacings)
            lower_half = ordered[: (len(ordered) + 1) // 2]
            characteristic_spacing = float(np.median(lower_half))
            gap_limit = min(16.0, 2.0 * characteristic_spacing)
        else:
            gap_limit = 16.0
        gap_method = "2x_median_lower_half_spacing_capped_at_16px"
    else:
        gap_limit = _finite(max_vertical_gap_px, "max_vertical_gap_px")
        if gap_limit <= 0:
            raise ValueError("max_vertical_gap_px must be positive")
        gap_method = "explicit"

    corridor = np.zeros((height, width), dtype=bool)
    support = np.zeros_like(corridor)
    spans: list[dict[str, Any]] = []
    gap_breaks: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None

    def draw_row(y: int, left: float, right: float) -> None:
        support[y, :] = True
        # Include original integer pixel centres exactly within the two limits.
        corridor[y, math.ceil(left): math.floor(right) + 1] = True

    for item in parsed:
        if not item["valid"]:
            previous = None
            continue
        y, left, right = item["y"], item["left"], item["right"]
        connect = previous is not None and previous["segment_index"] == item["segment_index"]
        if connect:
            delta = y - previous["y"]
            if delta > gap_limit:
                gap_breaks.append({"segment_index": item["segment_index"],
                                   "y_before": previous["y"], "y_after": y,
                                   "gap_px": delta})
                connect = False
        if connect:
            for yi in range(previous["y"] + 1, y):
                fraction = (yi - previous["y"]) / (y - previous["y"])
                draw_row(yi, previous["left"] + fraction * (left - previous["left"]),
                         previous["right"] + fraction * (right - previous["right"]))
            spans[-1]["y_last"] = y
            spans[-1]["sample_count"] += 1
        else:
            spans.append({"source_segment_index": item["segment_index"],
                          "y_first": y, "y_last": y, "sample_count": 1})
        draw_row(y, left, right)
        previous = item

    metadata = {
        "coordinates": "original_image_pixel_centres",
        "units": "original_image_pixels",
        "profile_kind": "display",
        "boundary_inclusion": "left <= integer_x <= right",
        "interpolation": "linear_between_adjacent_valid_samples_in_same_segment",
        "extrapolation": False,
        "gap_limit_method": gap_method,
        "characteristic_spacing_px": characteristic_spacing,
        "max_vertical_gap_px": gap_limit,
        "gap_breaks": gap_breaks,
        "input_rows": len(parsed),
        "valid_sample_rows": sum(item["valid"] for item in parsed),
        "invalid_sample_rows": sum(not item["valid"] for item in parsed),
        "input_segments": len(seen_segments),
        "observed_spans": spans,
        "supported_rows": int(np.count_nonzero(support[:, 0])),
        "corridor_pixel_count": int(np.count_nonzero(corridor)),
        "physical_intrusion_assessed": False,
    }
    return corridor, support, metadata


def analyze_object_mask(mask: Any, corridor: Any, support: Any) -> dict[str, Any]:
    """Measure real-mask overlap and distinguish incomplete rail observations.

    ``support`` means the corridor can be assessed at that pixel. For profiles
    created above it spans the full width of each observed image row. A zero
    overlap on partial support must not be reported as an object outside rails.
    With no supported object pixels, overlap fields are None, not measured zero.
    """
    obj, rail, known = (_binary(mask, "mask"), _binary(corridor, "corridor"),
                        _binary(support, "support"))
    if obj.shape != rail.shape or obj.shape != known.shape:
        raise ValueError("Object, corridor and support masks must share original image dimensions")
    if np.any(rail & ~known):
        raise ValueError("Corridor pixels cannot lie outside observed support")
    count = int(np.count_nonzero(obj))
    supported = int(np.count_nonzero(obj & known))
    overlap = int(np.count_nonzero(obj & rail))
    if count == 0:
        status, relation, reason = "empty_mask", "indeterminate", "empty_object_mask"
    elif supported == 0:
        status, relation, reason = "none", "indeterminate", "no_observed_corridor_support_at_object"
    elif supported < count:
        status = "partial"
        relation = "overlap_observed" if overlap else "indeterminate"
        reason = "overlap_on_partial_support" if overlap else "no_overlap_on_partial_support"
    else:
        status = "full"
        relation = "overlap_observed" if overlap else "no_overlap_observed"
        reason = "overlap_on_full_support" if overlap else "no_overlap_on_full_support"
    return {
        "mask_valid": bool(count),
        "mask_pixel_count": count,
        "support_pixel_count": supported,
        "unsupported_pixel_count": count - supported,
        "support_fraction": supported / count if count else None,
        "support_status": status,
        "overlap_pixel_count": overlap if supported else None,
        "overlap_fraction": overlap / count if supported else None,
        "overlap_fraction_supported": overlap / supported if supported else None,
        "relation_2d": relation,
        "relation_reason": reason,
        "complete_relation_assessment": bool(count and supported == count),
        "touches_image_border": bool(obj[0].any() or obj[-1].any() or obj[:, 0].any() or obj[:, -1].any()),
        "physical_intrusion_assessed": False,
    }


def reconstruct_mask(contours: list[dict[str, Any]], width: int, height: int) -> np.ndarray:
    """Rebuild a binary raster from this module's contour hierarchy, including holes.

    This is raster drawing, not filling independent exterior polygons: a hole's
    contour is on neighbouring foreground pixel centres. Drawing all contours
    together with their hierarchy is required for exact OpenCV reconstruction.
    """
    width, height = _integer(width, "width", 1), _integer(height, "height", 1)
    raster = np.zeros((height, width), dtype=np.uint8)
    if not contours:
        return raster.astype(bool)
    arrays: list[np.ndarray] = []
    children: dict[int | None, list[int]] = {}
    hierarchy = np.full((1, len(contours), 4), -1, dtype=np.int32)
    for index, contour in enumerate(contours):
        if _integer(contour.get("contour_id"), "contour_id") != index:
            raise ValueError("Contour IDs must be contiguous and in export order")
        parent = contour.get("parent_contour_id")
        if parent is not None:
            parent = _integer(parent, "parent_contour_id")
            if parent >= index:
                raise ValueError("Contour parents must precede their children")
        points = np.asarray(contour.get("points"))
        if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
            raise ValueError("Contour points must be a nonempty list of xy pairs")
        if points.dtype.kind not in "iuf" or not np.isfinite(points).all():
            raise ValueError("Contour points must be finite integer pixel centres")
        if not np.equal(points, np.floor(points)).all():
            raise ValueError("Contour pixel centres must be integers")
        if not ((points[:, 0] >= 0).all() and (points[:, 0] < width).all()
                and (points[:, 1] >= 0).all() and (points[:, 1] < height).all()):
            raise ValueError("Contour coordinates are outside original image pixels")
        arrays.append(np.ascontiguousarray(points, dtype=np.int32).reshape(-1, 1, 2))
        hierarchy[0, index, 3] = -1 if parent is None else parent
        children.setdefault(parent, []).append(index)
    for parent, siblings in children.items():
        if parent is not None:
            hierarchy[0, parent, 2] = siblings[0]
        for pos, index in enumerate(siblings):
            if pos:
                hierarchy[0, index, 1] = siblings[pos - 1]
            if pos + 1 < len(siblings):
                hierarchy[0, index, 0] = siblings[pos + 1]
    cv2.drawContours(raster, arrays, -1, 1, thickness=cv2.FILLED,
                     lineType=cv2.LINE_8, hierarchy=hierarchy)
    return raster.astype(bool)


def extract_contours(mask: Any) -> list[dict[str, Any]]:
    """Export exact OpenCV pixel-centre boundaries with every hole/component.

    RETR_TREE preserves nested holes/islands, CHAIN_APPROX_NONE keeps every
    raster boundary step. Foreground connectivity is 8. Exterior contours with
    even hierarchy depth identify distinct foreground components; holes inherit
    their exterior parent's component ID. Empty masks return []. Single pixels
    and thin lines remain degenerate contours rather than fabricated polygons.

    The complete hierarchy is rasterized back and checked against the source
    mask before returning; any lossy result raises an exception.
    """
    binary = _binary(mask, "mask")
    raw, hierarchy = cv2.findContours(binary.astype(np.uint8), cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    result: list[dict[str, Any]] = []
    next_component = 0
    for index, points in enumerate(raw):
        parent_index = int(hierarchy[0, index, 3])
        parent = None if parent_index < 0 else parent_index
        depth = 0 if parent is None else result[parent]["depth"] + 1
        hole = bool(depth % 2)
        if hole:
            component = result[parent]["component_id"]
        else:
            component = next_component
            next_component += 1
        xy = points.reshape(-1, 2)
        # The contour's polygon area is used only to name degenerate geometry.
        # All reported object/intersection areas come from binary pixel counts.
        kind = "point" if len(xy) == 1 else ("polyline" if cv2.contourArea(points) == 0 else "polygon")
        result.append({
            "contour_id": index, "component_id": component,
            "parent_contour_id": parent, "is_hole": hole, "depth": depth,
            "closed": True, "boundary_kind": kind,
            "points": xy.tolist(),
        })
    restored = reconstruct_mask(result, binary.shape[1], binary.shape[0])
    if not np.array_equal(restored, binary):
        raise ValueError("Contour hierarchy did not reconstruct the original binary mask exactly")
    return result
