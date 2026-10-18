"""Explicitly separate nominal-gauge scenarios from planar calibration.

No function here silently turns an assumed gauge into a measured gauge.
Plane coordinates require measured correspondences and compatible undistorted
image coordinates. A fitted homography alone is not field validation.
"""
import math
import cv2
import numpy as np


def gauge_scenario(left, right, image_width, assumed_gauge_m=None,
                   min_width_fraction=.02):
    values = [left, right, image_width, min_width_fraction]
    if not all(math.isfinite(v) for v in values) or not 0 <= left < right < image_width:
        raise ValueError('Limites observées et dimensions finies ordonnées requises')
    if not 0 < min_width_fraction < 1 or (assumed_gauge_m is not None and (not math.isfinite(assumed_gauge_m) or assumed_gauge_m <= 0)):
        raise ValueError('Écartement supposé positif et seuil dans ]0,1[ requis')
    width, center = right-left, (left+right)/2
    result = dict(metric_status='nominal_gauge_scenario_not_calibrated',
                  assumed_gauge_m=assumed_gauge_m, measured_gauge_m=None,
                  scale_m_per_px_assumed=None, axis_offset_from_image_midline_m_assumed=None,
                  ground_x_m=None, ground_z_m=None, physical_uncertainty_m=None)
    if width < min_width_fraction*image_width:
        result['metric_status'] = 'rejected_narrow_corridor'
        return result
    if assumed_gauge_m is None:
        result['metric_status'] = 'not_calibrated_no_reference_gauge'
        return result
    result.update(scale_m_per_px_assumed=assumed_gauge_m/width,
        axis_offset_from_image_midline_m_assumed=(center-(image_width-1)/2)*assumed_gauge_m/width)
    return result


def project_plane(points, homography):
    points, h = np.asarray(points, dtype=float), np.asarray(homography, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or h.shape != (3,3):
        raise ValueError('Points Nx2 et homographie 3x3 requis')
    if not np.isfinite(points).all() or not np.isfinite(h).all() or np.linalg.matrix_rank(h) != 3:
        raise ValueError('Transformation finie non singulière requise')
    q = np.column_stack([points, np.ones(len(points))]) @ h.T
    if np.any(np.abs(q[:,2]) < 1e-10):
        raise ValueError('Projection trop proche de l’horizon du plan')
    return q[:,:2]/q[:,2,None]


def fit_plane_calibration(image_points, world_points, check_image_points, check_world_points):
    """Fit measured planar points; independent check points are mandatory.

    Both sets must already be undistorted with the same camera intrinsics.
    World coordinates are supplied by a person in metres, never generated from
    predicted rails. Returns held-out residuals without claiming acceptance.
    """
    sets = [np.asarray(x, float) for x in [image_points, world_points, check_image_points, check_world_points]]
    a,b,c,d = sets
    for arr in sets:
        if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all():
            raise ValueError('Correspondances finies Nx2 requises')
    if len(a) != len(b) or len(c) != len(d) or len(a) < 4 or len(c) < 3:
        raise ValueError('Au moins 4 points de calage et 3 de contrôle distincts requis')
    for arr in sets:
        if np.linalg.matrix_rank(arr-arr.mean(axis=0)) < 2:
            raise ValueError('Les correspondances ne peuvent pas être colinéaires')
    if any(np.linalg.norm(x-y) < 1e-6 for x in a for y in c) or any(np.linalg.norm(x-y) < 1e-6 for x in b for y in d):
        raise ValueError('Les points de contrôle doivent être indépendants du calage')
    h, _ = cv2.findHomography(a,b,method=0)
    if h is None:
        raise ValueError('Homographie non estimable')
    errors = np.linalg.norm(project_plane(c,h)-d,axis=1)
    return dict(homography_image_to_plane=h.tolist(), control_errors_m=errors.tolist(),
                control_rmse_m=float(np.sqrt(np.mean(errors**2))),
                control_max_error_m=float(errors.max()), fit_points=len(a), check_points=len(c),
                field_validated=False, coordinate_units='metres_on_assumed_local_plane',
                note='Requires compatible lens correction, fixed camera pose, local plane and field acceptance')


def project_with_pixel_uncertainty(point, homography, pixel_covariance):
    """Conditional first-order covariance; calibration/plane errors excluded."""
    p,h,cov = np.asarray(point,float),np.asarray(homography,float),np.asarray(pixel_covariance,float)
    if p.shape != (2,) or cov.shape != (2,2) or not np.isfinite(cov).all():
        raise ValueError('Point 2D et covariance 2x2 requis')
    if not np.allclose(cov,cov.T) or np.min(np.linalg.eigvalsh(cov)) < 0:
        raise ValueError('Covariance symétrique positive requise')
    xy = project_plane([p],h)[0]
    denom = h[2] @ np.r_[p,1.]
    jacobian = (h[:2,:2]-xy[:,None]*h[2,:2])/denom
    return xy, jacobian @ cov @ jacobian.T
