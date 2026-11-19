"""Conservative additions to an immutable baseline, without new inference.

The caller supplies tile instances retained by the frozen first-stage merge at
confidence 0.15. This module never replaces, merges, re-scores or deletes a
baseline mask. The caller concatenates its baseline with the returned additions.
Confidence/area settings must be selected on validation, not on the video.
"""
from __future__ import annotations

import math
import numpy as np

from small_object_refinement import _integer, _validated_instance

MAX_BASELINE_IOU = 0.1
MAX_BASELINE_IOS = 0.2
RULE_VERSION = "selective_additions_v1"


def _identity(obj, index, kind):
    return (obj.get("candidate_id") or obj.get("observation_id")
            or f"{kind}_index_{index}")


def select_additions(baseline_objects, first_stage_tile_objects,
                     min_confidence, max_area_fraction, *,
                     image_width=None, image_height=None):
    """Return (kept_additions, per_candidate_audit) on one native image grid.

    An addition must have score >= min_confidence and mask-pixel area divided by
    W*H <= max_area_fraction. Reject every crop-edge or image-edge mask. Reject
    overlap with ANY baseline class/source when mask IoU > 0.1 OR mask IoS > 0.2,
    where IoS is intersection/min(area_a, area_b). Comparisons are strict; exact
    threshold equality is allowed. Tile candidates must already have passed the
    frozen first stage. They are then processed by decreasing score (ties by
    candidate ID) and also rejected against already-selected additions using the
    same overlap limits. No masks are united or changed.

    Dimensions are inferred from masks, or can be supplied explicitly (both or
    neither). Empty lists are supported. Baseline lists must already exclude
    quarantined hypotheses if the caller does not want them to block additions.
    All baseline predictions, including low-score ones used for AP, remain the
    caller's responsibility; this function returns additions only.
    """
    min_confidence = float(min_confidence)
    max_area_fraction = float(max_area_fraction)
    if (not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1
            or not math.isfinite(max_area_fraction)
            or not 0 < max_area_fraction <= 1):
        raise ValueError("Invalid selective-addition thresholds")
    if (image_width is None) != (image_height is None):
        raise ValueError("Provide both image_width and image_height")
    shape = None
    if image_width is not None:
        shape = (_integer(image_height, "image_height", 1),
                 _integer(image_width, "image_width", 1))

    baseline, tiles = [], []
    for source, destination in ((baseline_objects, baseline),
                                (first_stage_tile_objects, tiles)):
        for obj in source:
            candidate = _validated_instance(obj, shape)
            shape = candidate["mask"].shape
            destination.append(candidate)
    if not tiles:
        return [], []
    height, width = shape
    image_area = height * width
    kept, audit, selected_refs = [], [], []
    ordered = sorted(enumerate(tiles),
                     key=lambda item: (-float(item[1]["confidence_score"]),
                                       str(_identity(item[1], item[0], "tile"))))
    for index, candidate in ordered:
        markers = [candidate[k] for k in ("view_kind", "inference_origin")
                   if k in candidate]
        if not markers or any(marker != "tile" for marker in markers):
            raise ValueError("Expected already-merged tile candidates")
        if candidate.get("model_source") != "litter":
            raise ValueError("Selective additions require litter tile candidates")
        x0, y0, x1, y1 = candidate["mask_bbox"]
        touches_image = x0 == 0 or y0 == 0 or x1 == width or y1 == height
        touches_crop = bool(candidate.get("touches_crop_border", False))
        score = float(candidate["confidence_score"])
        area = candidate["_refinement_area"]
        fraction = area / image_area
        reasons = []
        if score < min_confidence:
            reasons.append("below_min_confidence")
        if fraction > max_area_fraction:
            reasons.append("exceeds_max_area_fraction")
        if touches_crop:
            reasons.append("touches_crop_border")
        if touches_image:
            reasons.append("touches_image_border")
        conflicts = []
        # Evaluate overlaps even for other rejections so every audit is complete.
        for base_index, ref in enumerate(baseline):
            a, b = candidate["mask_bbox"], ref["mask_bbox"]
            xx0, yy0 = max(a[0], b[0]), max(a[1], b[1])
            xx1, yy1 = min(a[2], b[2]), min(a[3], b[3])
            if xx1 <= xx0 or yy1 <= yy0:
                continue
            overlap = int(np.count_nonzero(candidate["mask"][yy0:yy1, xx0:xx1]
                                           & ref["mask"][yy0:yy1, xx0:xx1]))
            base_area = ref["_refinement_area"]
            iou = overlap / (area + base_area - overlap)
            ios = overlap / min(area, base_area)
            if iou > MAX_BASELINE_IOU or ios > MAX_BASELINE_IOS:
                conflicts.append({
                    "baseline_index": base_index,
                    "baseline_id": _identity(ref, base_index, "baseline"),
                    "baseline_candidate_id": ref.get("candidate_id"),
                    "baseline_observation_id": ref.get("observation_id"),
                    "baseline_model_source": ref.get("model_source"),
                    "baseline_class_id": ref.get("class_id"),
                    "intersection_pixels": overlap,
                    "mask_iou": iou, "mask_ios": ios,
                })
        if conflicts:
            reasons.append("overlaps_baseline")
        selected_conflicts = []
        for ref, ref_area, ref_key in selected_refs:
            a, b = candidate["mask_bbox"], ref["mask_bbox"]
            xx0, yy0 = max(a[0], b[0]), max(a[1], b[1])
            xx1, yy1 = min(a[2], b[2]), min(a[3], b[3])
            if xx1 <= xx0 or yy1 <= yy0:
                continue
            overlap = int(np.count_nonzero(candidate["mask"][yy0:yy1, xx0:xx1]
                                           & ref["mask"][yy0:yy1, xx0:xx1]))
            iou = overlap / (area + ref_area - overlap)
            ios = overlap / min(area, ref_area)
            if iou > MAX_BASELINE_IOU or ios > MAX_BASELINE_IOS:
                selected_conflicts.append({
                    "selected_addition_id": ref_key,
                    "selected_addition_candidate_id": ref.get("candidate_id"),
                    "selected_addition_observation_id": ref.get("observation_id"),
                    "intersection_pixels": overlap,
                    "mask_iou": iou, "mask_ios": ios,
                })
        if selected_conflicts:
            reasons.append("overlap_with_selected_addition")
        accepted = not reasons
        audit.append({
            "rule_version": RULE_VERSION,
            "candidate_index": index,
            "candidate_id": candidate.get("candidate_id"),
            "candidate_key": _identity(candidate, index, "tile"),
            "observation_id": candidate.get("observation_id"),
            "view_id": candidate.get("view_id"),
            "raw_detection_index": candidate.get("raw_detection_index"),
            "accepted": accepted,
            "reason": "accepted_selective_addition" if accepted else reasons[0],
            "rejection_reasons": reasons,
            "confidence_score": score,
            "mask_area_pixels": area,
            "image_width": width, "image_height": height,
            "mask_area_fraction": fraction,
            "touches_crop_border": touches_crop,
            "touches_image_border": touches_image,
            "min_confidence": min_confidence,
            "max_area_fraction": max_area_fraction,
            "max_baseline_iou": MAX_BASELINE_IOU,
            "max_baseline_ios": MAX_BASELINE_IOS,
            "baseline_ids": [c["baseline_id"] for c in conflicts],
            "baseline_conflicts": conflicts,
            "selected_addition_conflicts": selected_conflicts,
        })
        if accepted:
            candidate.pop("_refinement_area")
            kept.append(candidate)
            selected_refs.append((candidate, area, _identity(candidate, index, "tile")))
    return kept, audit
