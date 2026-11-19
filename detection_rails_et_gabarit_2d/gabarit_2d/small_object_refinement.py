"""Native-mask refinement by overlapping crops and an explicit ambiguity guard.

Tiling changes the pixels presented to the network, not the coordinate system of
the exports. Cross-view duplicate handling keeps one observed mask; it never
unions masks or merges detections from the same view. The train guard is a
heuristic quarantine signal, not evidence that every flagged object is false.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Invalid {name}")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number < minimum:
        raise ValueError(f"Invalid {name}")
    return int(number)


def _crop(crop, width, height):
    width, height = _integer(width, "width", 1), _integer(height, "height", 1)
    if len(crop) != 4:
        raise ValueError("Crop must contain four coordinates")
    x0, y0, x1, y1 = [_integer(v, "crop coordinate") for v in crop]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Crop must lie inside the image and have positive area")
    return x0, y0, x1, y1


def _starts(lo, hi, size, stride):
    if hi - lo <= size:
        return [lo]
    starts = list(range(lo, hi - size + 1, stride))
    if starts[-1] != hi - size:
        starts.append(hi - size)
    return starts


def tile_windows(crop, width, height, tile_size=640, overlap=.25):
    """Cover exactly the crop with deterministic in-bounds overlapping windows.

    Last tiles are shifted back to the crop edge, rather than padded with invented
    pixels. Crops smaller than one tile return one window. Order is top-to-bottom,
    then left-to-right. A caller may omit the one-window case if it duplicates an
    already performed global crop inference.
    """
    x0, y0, x1, y1 = _crop(crop, width, height)
    size = _integer(tile_size, "tile_size", 1)
    overlap = float(overlap)
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("overlap must lie in [0, 1)")
    stride = max(1, int(math.floor(size * (1 - overlap))))
    return [[x, y, min(x + size, x1), min(y + size, y1)]
            for y in _starts(y0, y1, size, stride)
            for x in _starts(x0, x1, size, stride)]


def _native_masks(result, shape):
    """Validate binary masks on their current device, transfer only uint8 data."""
    import torch
    boxes = result.boxes.data.detach().cpu().numpy()
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError("Expected untracked xyxy/confidence/class predictions")
    if not len(boxes):
        if result.masks is not None and len(result.masks.data):
            raise ValueError("Masks without boxes")
        return boxes, np.empty((0, *shape), dtype=np.uint8)
    if result.masks is None:
        raise ValueError("Boxes without native instance masks")
    masks = result.masks.data.detach()
    if tuple(masks.shape) != (len(boxes), *shape):
        raise ValueError("Native tile mask dimensions mismatch")
    # A min/max check alone cannot detect a fractional value between 0 and 1.
    # Check the binary contract before casting, on the current tensor device.
    if bool(((masks != 0) & (masks != 1)).any().item()):
        raise ValueError("Native instance masks must be binary")
    compact = masks.to(dtype=torch.uint8).cpu().numpy()
    return boxes, compact


def predict_tiled_litter(model, bgr, crop, imgsz=960, conf=.15, device="mps", *,
                         tile_size=640, overlap=.25):
    """Return (instances, empty-mask count), compatible with V1 native masks.

    Each model call is sequential (no image/tile batch on the Apple GPU). All
    coordinates and boolean masks in the result use the original image grid.
    Category names are read from the model result, never from fixed COCO IDs.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected an HxWx3 image")
    height, width = bgr.shape[:2]
    conf = float(conf)
    if not math.isfinite(conf) or not 0 < conf <= 1:
        raise ValueError("Invalid confidence threshold")
    windows = tile_windows(crop, width, height, tile_size, overlap)
    instances, empty = [], 0
    for tile_index, window in enumerate(windows):
        x0, y0, x1, y1 = window
        result = model.predict(bgr[y0:y1, x0:x1], imgsz=imgsz, conf=conf,
                               iou=.5, nms=True, retina_masks=True, max_det=100,
                               device=device, verbose=False)[0]
        if bool(model.predictor.model.end2end):
            raise ValueError("Conventional NMS backend required")
        boxes, masks = _native_masks(result, (y1-y0, x1-x0))
        for detection_index, (box, local_u8) in enumerate(zip(boxes, masks)):
            if not np.isfinite(box).all():
                raise ValueError("Nonfinite prediction")
            cls = _integer(box[5], "class_id")
            score = float(box[4])
            if not 0 <= score <= 1:
                raise ValueError("Invalid confidence score")
            if not (0 <= box[0] <= box[2] <= x1-x0+.001
                    and 0 <= box[1] <= box[3] <= y1-y0+.001):
                raise ValueError("Prediction box lies outside its native tile")
            local = local_u8.astype(bool, copy=False)
            yy, xx = np.nonzero(local)
            if not len(xx):
                empty += 1
                continue
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = local
            bounds = [int(xx.min())+x0, int(yy.min())+y0,
                      int(xx.max())+x0+1, int(yy.max())+y0+1]
            touches = bool((x0 > 0 and local[:, 0].any())
                           or (x1 < width and local[:, -1].any())
                           or (y0 > 0 and local[0].any())
                           or (y1 < height and local[-1].any()))
            instances.append({
                "model_source": "litter", "class_id": cls,
                "class_name": str(result.names[cls]), "confidence_score": score,
                "raw_detection_index": len(instances), "mask": mask,
                "mask_bbox": bounds,
                "model_bbox_original_px": [float(box[0])+x0, float(box[1])+y0,
                                           float(box[2])+x0, float(box[3])+y0],
                "inference_crop_xyxy": list(window),
                "touches_crop_border": touches,
                "view_id": f"tile_{tile_index:04d}", "view_kind": "tile",
                "tile_index": tile_index, "tile_detection_index": detection_index,
            })
    return instances, empty


def _validated_instance(obj, shape):
    result = dict(obj)
    mask = result["mask"]
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("Expected an original-grid boolean mask")
    if shape is not None and mask.shape != shape:
        raise ValueError("Different original-image mask grids")
    bbox = result["mask_bbox"]
    if len(bbox) != 4:
        raise ValueError("Invalid mask bounding extent")
    bbox = [_integer(v, "mask extent") for v in bbox]
    x0, y0, x1, y1 = _crop(bbox, mask.shape[1], mask.shape[0])
    area = int(np.count_nonzero(mask))
    if not area:
        raise ValueError("Cannot merge an empty mask")
    # Validate support once; intersections below only visit intersecting bounds.
    yy, xx = np.nonzero(mask)
    if bbox != [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1]:
        raise ValueError("Mask extent does not match its pixels")
    score = float(result["confidence_score"])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid confidence score")
    result["mask_bbox"] = [x0, y0, x1, y1]
    result["_refinement_area"] = area
    return result


def merge_multiscale(global_objs, tile_objs, iou_threshold=.5, ios_threshold=.8):
    """Cross-view mask NMS with no mask union and no same-view suppression.

    IoU > .5 permits a duplicate match. IoS > .8 additionally permits a
    truncated view to match its complete counterpart. IoS is NOT used between
    two complete masks: a newly resolved small object must not disappear merely
    because an overbroad global mask contains it.

    Ranking prefers non-truncated masks, then confidence, then the global view.
    Thresholds/ranking are heuristics and require evaluation; scores are not
    calibrated probabilities. Returns (kept instances, suppression audit).
    """
    if not 0 < iou_threshold < 1 or not 0 < ios_threshold < 1:
        raise ValueError("Invalid duplicate thresholds")
    values, shape = [], None
    for kind, objects in (("global", global_objs), ("tile", tile_objs)):
        for index, obj in enumerate(objects):
            copied = _validated_instance(obj, shape)
            shape = copied["mask"].shape
            copied.setdefault("view_kind", kind)
            if "view_id" not in copied:
                if kind == "global":
                    copied["view_id"] = "global"
                elif "tile_index" in copied:
                    copied["view_id"] = f"tile_{_integer(copied['tile_index'], 'tile_index'):04d}"
                else:
                    raise ValueError("Tile instances must identify their prediction view")
            if not isinstance(copied["view_id"], str) or not copied["view_id"]:
                raise ValueError("Prediction view ID must be a nonempty string")
            copied["_refinement_order"] = len(values)
            values.append(copied)
    values.sort(key=lambda obj: (bool(obj.get("touches_crop_border", False)),
                                -float(obj["confidence_score"]),
                                obj["view_kind"] != "global", obj["_refinement_order"]))
    kept, audit = [], []
    for candidate in values:
        duplicate = None
        for ref in kept:
            if (candidate.get("model_source"), candidate.get("class_id")) != (ref.get("model_source"), ref.get("class_id")):
                continue
            if candidate["view_id"] == ref["view_id"]:
                continue
            a, b = candidate["mask_bbox"], ref["mask_bbox"]
            x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
            if x1 <= x0 or y1 <= y0:
                continue
            intersection = int(np.count_nonzero(candidate["mask"][y0:y1, x0:x1]
                                                & ref["mask"][y0:y1, x0:x1]))
            area_a, area_b = candidate["_refinement_area"], ref["_refinement_area"]
            iou = intersection / (area_a + area_b - intersection)
            ios = intersection / min(area_a, area_b)
            edge = bool(candidate.get("touches_crop_border") or ref.get("touches_crop_border"))
            rule = "cross_view_mask_iou" if iou > iou_threshold else (
                "cross_view_truncated_mask_ios" if edge and ios > ios_threshold else None)
            if rule:
                duplicate = {"suppressed_view_id": candidate["view_id"],
                             "kept_view_id": ref["view_id"],
                             "suppressed_candidate_id": candidate.get("candidate_id"),
                             "kept_candidate_id": ref.get("candidate_id"),
                             "suppressed_raw_detection_index": candidate.get("raw_detection_index"),
                             "kept_raw_detection_index": ref.get("raw_detection_index"),
                             "mask_iou": iou, "mask_ios": ios, "rule": rule,
                             "suppressed_touches_crop_border": bool(candidate.get("touches_crop_border")),
                             "kept_touches_crop_border": bool(ref.get("touches_crop_border"))}
                break
        if duplicate is None:
            kept.append(candidate)
        else:
            audit.append(duplicate)
    for obj in kept:
        obj.pop("_refinement_area")
        obj.pop("_refinement_order")
    return kept, audit


def train_guard(obj, corridor, *, confidence_ceiling=.5, corridor_coverage_min=.9,
                height_fraction_min=.75, bottom_fraction_min=.95):
    """Identify a weak, image-spanning train mask that blankets the track.

    The caller may quarantine this hypothesis from the main result, but must keep
    it with this reason in a review/audit output. A true low-confidence train can
    satisfy this heuristic: quarantine does not certify that the object is false.
    No class is excluded globally, and neither timestamps nor video IDs are used.
    """
    thresholds = {"confidence_ceiling": confidence_ceiling,
                  "corridor_coverage_min": corridor_coverage_min,
                  "height_fraction_min": height_fraction_min,
                  "bottom_fraction_min": bottom_fraction_min}
    if any(not math.isfinite(float(v)) or not 0 <= v <= 1 for v in thresholds.values()):
        raise ValueError("Invalid train-guard threshold")
    score = float(obj["confidence_score"])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid train confidence")
    result: dict[str, Any] = {"quarantine": False, "reason": "not_train_class",
                              "class_name": str(obj["class_name"]),
                              "confidence_score": score, "thresholds": thresholds,
                              "corridor_coverage": None, "mask_height_fraction": None,
                              "mask_bottom_fraction": None}
    if str(obj["class_name"]).strip().casefold() != "train":
        return result
    mask = obj["mask"]
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.bool_
            or not isinstance(corridor, np.ndarray) or corridor.dtype != np.bool_
            or mask.ndim != 2 or mask.shape != corridor.shape or not mask.size):
        raise ValueError("Train guard needs aligned original-grid boolean masks")
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError("Empty train mask")
    height = mask.shape[0]
    result["mask_height_fraction"] = float((yy.max()-yy.min()+1) / height)
    result["mask_bottom_fraction"] = float((yy.max()+1) / height)
    corridor_area = int(np.count_nonzero(corridor))
    result["corridor_pixel_count"] = corridor_area
    if not corridor_area:
        result["reason"] = "no_observed_corridor"
        return result
    intersection = int(np.count_nonzero(mask & corridor))
    result["corridor_intersection_pixel_count"] = intersection
    result["corridor_coverage"] = intersection / corridor_area
    result["quarantine"] = bool(score < confidence_ceiling
                                 and result["corridor_coverage"] >= corridor_coverage_min
                                 and result["mask_height_fraction"] >= height_fraction_min
                                 and result["mask_bottom_fraction"] >= bottom_fraction_min)
    result["reason"] = ("ambiguous_train_tracklike" if result["quarantine"]
                         else "train_not_matching_tracklike_guard")
    return result
