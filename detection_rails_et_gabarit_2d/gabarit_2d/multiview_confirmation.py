"""Require a second tile view without modifying a retained native mask or score.

This replays the frozen first-stage suppression audit. It is a consistency rule
between crops of ONE detector, not independent evidence or calibrated certainty.
No image, model, baseline mask, selected threshold, or existing export is changed.
"""
from __future__ import annotations

import math
import re


RULE_VERSION = "distinct_tile_iou_confirmation_v1"
_TILE_VIEW = re.compile(r"tile_[0-9]+\Z")


def _raw_index(value, field):
    if isinstance(value, bool):
        raise ValueError("Invalid " + field)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid " + field) from exc
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError("Invalid " + field)
    return int(number)


def _optional_id(value, field):
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError("Invalid " + field)
    return value


def filter_confirmed_tiles(candidates, suppression_audit):
    """Return (confirmed_candidates, per_candidate_audit), preserving input order.

    A survivor is confirmed only by a cross_view_mask_iou suppression with finite
    IoU > .5, the exact kept candidate ID/view, and a distinct suppressed tile
    view. Global views and IoS-only suppressions never count. Multiple suppressed
    predictions from the same view count as one view. When raw indices or
    suppressed IDs are provided, contradictory identities raise ValueError.
    Unrelated suppression records are ignored. Returned candidates are the
    original dictionaries: neither mask geometry nor confidence is modified.
    """
    candidates = list(candidates)
    by_id, by_raw, reports = {}, {}, {}
    for obj in candidates:
        ident = _optional_id(obj.get("candidate_id"), "candidate_id")
        if ident is None or ident in by_id:
            raise ValueError("Missing or duplicate survivor candidate ID")
        view = obj.get("view_id")
        if not isinstance(view, str) or _TILE_VIEW.fullmatch(view) is None:
            raise ValueError("Survivor must identify a tile view")
        markers = [obj[k] for k in ("view_kind", "inference_origin") if k in obj]
        if not markers or any(marker != "tile" for marker in markers):
            raise ValueError("Expected a tile survivor")
        if "model_source" in obj and obj["model_source"] != "litter":
            raise ValueError("Expected a litter tile survivor")
        if obj.get("tile_index") is not None:
            tile_index = _raw_index(obj["tile_index"], "tile_index")
            if tile_index != int(view.split("_", 1)[1]):
                raise ValueError("Tile index and view ID differ")
        raw = None
        if obj.get("raw_detection_index") is not None:
            raw = _raw_index(obj["raw_detection_index"], "raw_detection_index")
            if raw in by_raw:
                raise ValueError("Duplicate global tile raw index")
            by_raw[raw] = ident
        by_id[ident] = obj
        reports[ident] = {
            "rule_version": RULE_VERSION,
            "candidate_id": ident, "view_id": view,
            "raw_detection_index": raw,
            "confirmed": False, "reason": "no_distinct_tile_iou_support",
            "distinct_tile_views": [view], "distinct_tile_view_count": 1,
            "supporting_evidence": [], "rejected_evidence": [],
        }

    suppressed_identities, suppressed_raw_identities = {}, {}
    for audit_index, entry in enumerate(suppression_audit):
        if not isinstance(entry, dict):
            raise ValueError("Suppression audit record must be a dictionary")
        ident = entry.get("kept_candidate_id")
        if ident not in by_id:
            # Detect a wrong kept ID attached to the identity of a real survivor.
            raw = entry.get("kept_raw_detection_index")
            if entry.get("rule") == "cross_view_mask_iou" and raw is not None:
                raw = _raw_index(raw, "kept_raw_detection_index")
                actual = by_raw.get(raw)
                if actual is not None and entry.get("kept_view_id") == by_id[actual]["view_id"]:
                    raise ValueError("Kept candidate ID contradicts survivor view/raw index")
            continue
        obj, report = by_id[ident], reports[ident]
        reject = lambda reason: report["rejected_evidence"].append(
            {"audit_index": audit_index, "reason": reason})
        if entry.get("rule") != "cross_view_mask_iou":
            reject("not_cross_view_mask_iou")
            continue
        if entry.get("kept_view_id") != obj["view_id"]:
            raise ValueError("Kept view ID contradicts survivor candidate ID")
        kept_raw = entry.get("kept_raw_detection_index")
        if kept_raw is not None:
            kept_raw = _raw_index(kept_raw, "kept_raw_detection_index")
            if report["raw_detection_index"] is not None and kept_raw != report["raw_detection_index"]:
                raise ValueError("Kept raw index contradicts survivor candidate ID")
        try:
            iou = float(entry["mask_iou"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("Invalid confirmation IoU") from exc
        if isinstance(entry["mask_iou"], bool) or not math.isfinite(iou) or not 0 <= iou <= 1:
            raise ValueError("Invalid confirmation IoU")
        if iou <= .5:
            reject("mask_iou_not_strictly_above_half")
            continue
        view = entry.get("suppressed_view_id")
        if not isinstance(view, str) or _TILE_VIEW.fullmatch(view) is None:
            reject("support_is_not_a_tile_view")
            continue
        if view == obj["view_id"]:
            reject("support_is_same_tile_view")
            continue
        suppressed_id = _optional_id(entry.get("suppressed_candidate_id"), "suppressed_candidate_id")
        suppressed_raw = entry.get("suppressed_raw_detection_index")
        if suppressed_raw is not None:
            suppressed_raw = _raw_index(suppressed_raw, "suppressed_raw_detection_index")
            if suppressed_raw in by_raw:
                raise ValueError("Suppressed raw index also belongs to a survivor")
        if suppressed_id in by_id:
            raise ValueError("Suppressed candidate ID also belongs to a survivor")
        signature = (view, suppressed_raw, ident)
        if suppressed_id is not None:
            if suppressed_id in suppressed_identities and suppressed_identities[suppressed_id] != signature:
                raise ValueError("Suppressed candidate identity is inconsistent across evidence")
            suppressed_identities[suppressed_id] = signature
        if suppressed_raw is not None:
            raw_signature = (view, suppressed_id, ident)
            if (suppressed_raw in suppressed_raw_identities
                    and suppressed_raw_identities[suppressed_raw] != raw_signature):
                raise ValueError("Suppressed raw identity is inconsistent across evidence")
            suppressed_raw_identities[suppressed_raw] = raw_signature
        report["supporting_evidence"].append({
            "audit_index": audit_index, "suppressed_view_id": view,
            "suppressed_candidate_id": suppressed_id,
            "suppressed_raw_detection_index": suppressed_raw,
            "mask_iou": iou,
        })
        if view not in report["distinct_tile_views"]:
            report["distinct_tile_views"].append(view)

    confirmed = []
    for obj in candidates:
        report = reports[obj["candidate_id"]]
        report["distinct_tile_views"].sort()
        report["distinct_tile_view_count"] = len(report["distinct_tile_views"])
        report["confirmed"] = report["distinct_tile_view_count"] >= 2
        if report["confirmed"]:
            report["reason"] = "confirmed_by_distinct_tile_iou"
            confirmed.append(obj)
    return confirmed, [reports[obj["candidate_id"]] for obj in candidates]
