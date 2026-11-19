#!/usr/bin/env python3
"""Lire un TXT de contours, sans modèle ni GPU.

Exemple Python : frame = load_frame("TXT/frame_001242.txt")
                objets = frame["objects"]
                complet, interieur = object_masks(objets[0], frame["image_width"], frame["image_height"])

La reconstruction optionnelle exige numpy, opencv-python et corridor_geometry.py
à côté de ce fichier. La lecture/validation du TXT utilise seulement Python.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _int(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} : entier >= {minimum} attendu")
    return value


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} : nombre fini attendu")
    return value


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} : objet JSON attendu")
    return value


def _text(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} : texte non vide attendu")
    return value


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Clé JSON répétée : {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Valeur JSON non finie interdite : {value}")


def validate_contours(contours, width, height, label):
    """Contrôler points, ordre et hiérarchie sans transformer les contours."""
    if not isinstance(contours, list) or not contours:
        raise ValueError(f"{label} : liste non vide de contours attendue")
    next_component = 0
    for index, contour in enumerate(contours):
        c = _mapping(contour, f"{label}[{index}]")
        if _int(c.get("contour_id"), "contour_id") != index:
            raise ValueError("Les contour_id doivent suivre l'ordre 0, 1, 2...")
        parent = c.get("parent_contour_id")
        if parent is not None:
            _int(parent, "parent_contour_id")
            if parent >= index:
                raise ValueError("Le contour parent doit précéder son enfant")
        depth = 0 if parent is None else contours[parent]["depth"] + 1
        if _int(c.get("depth"), "depth") != depth:
            raise ValueError("Profondeur de contour incohérente")
        if type(c.get("is_hole")) is not bool or c["is_hole"] != bool(depth % 2):
            raise ValueError("is_hole incohérent avec la hiérarchie")
        expected_component = contours[parent]["component_id"] if depth % 2 else next_component
        if _int(c.get("component_id"), "component_id") != expected_component:
            raise ValueError("component_id incohérent avec la hiérarchie")
        if depth % 2 == 0:
            next_component += 1
        if c.get("closed") is not True or c.get("boundary_kind") not in {"point", "polyline", "polygon"}:
            raise ValueError("Description du contour invalide")
        points = c.get("points")
        if not isinstance(points, list) or not points:
            raise ValueError("Chaque contour doit contenir des points")
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("Un point doit être [x, y]")
            x, y = (_int(point[0], "x_px"), _int(point[1], "y_px"))
            if x >= width or y >= height:
                raise ValueError("Un point sort des dimensions originales de l'image")
        if c["boundary_kind"] == "point" and len(points) != 1:
            raise ValueError("Un contour de type point doit contenir exactement un point")


def load_frame(frame_file):
    """Retourner le document d'une frame après validation de sa structure.

    Ne lance aucune prédiction. Ne lit pas le CSV des rails ni la vidéo.
    Utiliser object_masks() pour reconstruire et vérifier les masques raster.
    """
    with Path(frame_file).open(encoding="utf-8") as handle:
        frame = json.load(handle, object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
    _mapping(frame, "frame")
    if _int(frame.get("schema_version"), "schema_version", 1) not in {1, 2, 3} or frame.get("processing_status") != "ok":
        raise ValueError("Schéma non pris en charge ou frame non traitée correctement")
    _text(frame.get("video_id"), "video_id")
    _int(frame.get("frame_index"), "frame_index")
    width = _int(frame.get("image_width"), "image_width", 1)
    height = _int(frame.get("image_height"), "image_height", 1)
    pts = _int(frame.get("source_pts"), "source_pts")
    numerator = _int(frame.get("time_base_num"), "time_base_num", 1)
    denominator = _int(frame.get("time_base_den"), "time_base_den", 1)
    t = _number(frame.get("time_video_s"), "time_video_s")
    if t < 0 or not math.isclose(t, pts * numerator / denominator, rel_tol=0, abs_tol=1e-7):
        raise ValueError("Le temps vidéo ne correspond pas à l'horodatage source")
    if frame.get("coordinate_convention") != "original pixel centres, x right, y down":
        raise ValueError("Convention de coordonnées non reconnue")
    _mapping(frame.get("rail_support"), "rail_support")
    objects = frame.get("objects")
    if not isinstance(objects, list):
        raise ValueError("objects doit être une liste ; [] est un résultat vide valide")
    review = frame.get("quarantined_predictions", [])
    if not isinstance(review, list) or (frame["schema_version"] == 1 and review):
        raise ValueError("Liste de prédictions à vérifier invalide")
    for obj in review:
        _text(obj.get("review_reason"), "review_reason")
    identifiers = set()
    for obj in objects + review:
        _mapping(obj, "objet")
        identifier = _text(obj.get("observation_id"), "observation_id")
        if identifier in identifiers:
            raise ValueError("Deux objets ont le même observation_id dans une frame")
        identifiers.add(identifier)
        _int(obj.get("class_id"), "class_id")
        _text(obj.get("class_name"), "class_name")
        if obj.get("model_source") not in {"public", "litter"}:
            raise ValueError("Source de modèle inconnue")
        score = _number(obj.get("confidence_score"), "confidence_score")
        if not 0 <= score <= 1:
            raise ValueError("confidence_score doit être compris entre 0 et 1")
        if obj.get("track_id") is not None:
            _text(obj["track_id"], "track_id")
        if type(obj.get("track_confirmed")) is not bool:
            raise ValueError("track_confirmed doit être un booléen")
        _int(obj.get("observed_hits"), "observed_hits")
        analysis = _mapping(obj.get("analysis"), "analysis")
        if analysis.get("relation_2d") != "overlap_observed":
            raise ValueError("Ce format exporte seulement les recouvrements observés")
        _int(analysis.get("mask_pixel_count"), "mask_pixel_count", 1)
        _int(analysis.get("overlap_pixel_count"), "overlap_pixel_count", 1)
        for key in ("full_predicted_mask_contours", "corridor_intersection_contours"):
            validate_contours(obj.get(key), width, height, key)
        for key in ("full_mask_sha256", "intersection_mask_sha256"):
            value = obj.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"{key} : empreinte SHA-256 manquante ou invalide")
    return frame


def get_objects(frame_file):
    """Liste des objets de la frame ; le temps commun se lit avec load_frame()."""
    return load_frame(frame_file)["objects"]


def object_masks(obj, width, height):
    """Retourner (masque_complet, masque_intersection), tableaux bool H×W.

    Les trous sont respectés. Les empreintes garantissent que ces tableaux
    reproduisent les masques exportés, pas que la prédiction est correcte.
    """
    import numpy as np
    from corridor_geometry import reconstruct_mask
    width = _int(width, "width", 1)
    height = _int(height, "height", 1)
    full = reconstruct_mask(obj["full_predicted_mask_contours"], width, height)
    inside = reconstruct_mask(obj["corridor_intersection_contours"], width, height)
    for mask, key in ((full, "full_mask_sha256"), (inside, "intersection_mask_sha256")):
        digest = hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()
        if digest != obj[key]:
            raise ValueError(f"Le masque reconstruit diffère de l'empreinte {key}")
    if np.any(inside & ~full):
        raise ValueError("L'intersection contient des pixels absents du masque complet")
    if int(full.sum()) != obj["analysis"]["mask_pixel_count"] or int(inside.sum()) != obj["analysis"]["overlap_pixel_count"]:
        raise ValueError("Les surfaces en pixels ne correspondent pas aux contours")
    return full, inside


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-file", type=Path, required=True, help="TXT/frame_XXXXXX.txt")
    parser.add_argument("--out", type=Path, help="Nouveau dossier facultatif pour les masques PNG ; aucun écrasement")
    args = parser.parse_args()
    frame = load_frame(args.frame_file)
    if args.out:
        import cv2
        args.out.mkdir(parents=True, exist_ok=False)
        for index, obj in enumerate(frame["objects"]):
            full, inside = object_masks(obj, frame["image_width"], frame["image_height"])
            for name, mask in (("complet", full), ("interieur", inside)):
                if not cv2.imwrite(str(args.out / f"objet_{index:03d}_{name}.png"), mask.astype("uint8") * 255):
                    raise OSError("Écriture du masque PNG impossible")
    summary = {key: frame[key] for key in ("video_id", "frame_index", "time_video_s", "image_width", "image_height")}
    summary["nombre_observations_dans_couloir"] = len(frame["objects"])
    summary["nombre_hypotheses_a_verifier"] = len(frame.get("quarantined_predictions", []))
    summary["objets"] = [{key: obj[key] for key in ("observation_id", "track_id", "class_name", "confidence_score")} for obj in frame["objects"]]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
