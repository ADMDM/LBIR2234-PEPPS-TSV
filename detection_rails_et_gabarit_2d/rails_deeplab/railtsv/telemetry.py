"""Indicative image-space measurements, never physical gauge or accuracy."""
import cv2
import numpy as np


CSV_FIELDS = ['frame_index', 'seconds', 'source_frame_index', 'source_seconds',
              'repeated_source_frame', 'row_y', 'left_inner_x', 'right_inner_x',
              'apparent_width_px', 'model_score_left', 'model_score_right',
              'status', 'reason', 'width_m', 'independent_accuracy_pct',
              'center_x_px','center_y_px','left_outer_x','right_outer_x',
              'rail_left_thickness_px','rail_right_thickness_px','temporal_status','axis_points_json',
              'representation','corridor_model_score']


def _interval(row, label):
    xs = np.flatnonzero(row == label)
    if len(xs) < 2:
        return None, 'rail_absent_or_too_thin'
    if np.any(np.diff(xs) != 1):
        return None, 'multiple_segments'
    if xs[0] == 0 or xs[-1] == len(row)-1:
        return None, 'rail_cut_by_image_edge'
    return (int(xs[0]), int(xs[-1])), ''


def measure_section(mask, probabilities, row_fraction=.8, min_score=.55):
    """Distance between predicted inner-edge pixel centres on one image row.

    Require a unique ordered pair on the centre row and most nearby rows.
    These heuristic checks cannot distinguish a plausible false positive.
    Softmax means apply only to the predicted rail pixels in this small band.
    No temporal smoothing, camera calibration or physical gauge is applied.
    """
    mask = np.asarray(mask)
    probs = np.asarray(probabilities)
    if mask.ndim != 2 or probs.shape != (3, *mask.shape):
        raise ValueError('Dimensions masque/probabilités incompatibles')
    if not 0 < row_fraction < 1 or not 0 <= min_score <= 1:
        raise ValueError('Ligne dans ]0,1[ et score minimal dans [0,1] requis')
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError('Probabilités non finies ou hors [0,1]')
    h, w = mask.shape
    y = round(row_fraction*(h-1))
    low, high = max(0,y-3), min(h,y+4)
    result = dict(row_y=y, left_inner_x=None, right_inner_x=None, apparent_width_px=None,
                  model_score_left=None, model_score_right=None, status='rejected',
                  reason='', width_m=None, independent_accuracy_pct=None,
                  center_x_px=None,center_y_px=None,left_outer_x=None,right_outer_x=None,
                  rail_left_thickness_px=None,rail_right_thickness_px=None,
                  representation='tsv',corridor_model_score=None)
    band = mask[low:high]
    for label,name in [(1,'left'),(2,'right')]:
        selected = band == label
        if selected.any():
            result['model_score_'+name] = float(probs[label,low:high][selected].mean())
    pairs=[]
    for yy in range(low,high):
        left, why_left = _interval(mask[yy],1)
        right, why_right = _interval(mask[yy],2)
        reason = why_left or why_right
        if not reason and left[1] >= right[0]:
            reason = 'left_right_order_invalid'
        if yy == y:
            if reason:
                result['reason'] = reason
                return result
            centre_pair = (left[1], right[0])
        if not reason:
            pairs.append((left[1],right[0]))
    if len(pairs) < max(1, int(np.ceil(.7*(high-low)))):
        result['reason'] = 'neighbouring_rows_inconsistent'
        return result
    if np.max(np.ptp(np.asarray(pairs),axis=0)) > max(4, .015*w):
        result['reason'] = 'edges_unstable_in_local_band'
        return result
    if min(result['model_score_left'],result['model_score_right']) < min_score:
        result['reason'] = 'low_uncalibrated_model_score'
        return result
    left,right = centre_pair
    outer_left,_=_interval(mask[y],1);outer_right,_=_interval(mask[y],2)
    result.update(left_inner_x=left,right_inner_x=right,apparent_width_px=right-left,
                  center_x_px=(left+right)/2,center_y_px=y,
                  left_outer_x=outer_left[0],right_outer_x=outer_right[1],
                  rail_left_thickness_px=outer_left[1]-outer_left[0]+1,
                  rail_right_thickness_px=outer_right[1]-outer_right[0]+1,
                  status='candidate',reason='indicative_pixel_measurement_not_validated')
    return result


def _corridor_interval(row):
    """One complete observed corridor interval, never an inferred hidden edge."""
    interval, reason = _interval(row, 1)
    return interval, {'rail_absent_or_too_thin': 'corridor_absent_or_too_thin',
                      'rail_cut_by_image_edge': 'corridor_cut_by_image_edge'}.get(reason, reason)


def measure_corridor(mask, probabilities, row_fraction=.8, min_score=.55):
    """Measure the predicted corridor boundaries and their image-row midpoint.

    The legacy inner-edge columns now contain corridor-boundary pixel centres.
    They do not establish that a metallic rail was detected. Outer metallic
    edges, rail thicknesses, per-rail scores, metres and accuracy stay unknown.
    """
    mask, probs = np.asarray(mask), np.asarray(probabilities)
    if mask.ndim != 2 or probs.shape != (2, *mask.shape) or not mask.size:
        raise ValueError('Dimensions masque/probabilités du couloir incompatibles')
    if not 0 < row_fraction < 1 or not 0 <= min_score <= 1:
        raise ValueError('Ligne dans ]0,1[ et score minimal dans [0,1] requis')
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError('Probabilités non finies ou hors [0,1]')
    h, w = mask.shape
    y = round(row_fraction*(h-1))
    low, high = max(0, y-3), min(h, y+4)
    result = dict(row_y=y, left_inner_x=None, right_inner_x=None,
                  apparent_width_px=None, model_score_left=None, model_score_right=None,
                  status='rejected', reason='', width_m=None, independent_accuracy_pct=None,
                  center_x_px=None, center_y_px=None, left_outer_x=None, right_outer_x=None,
                  rail_left_thickness_px=None, rail_right_thickness_px=None,
                  representation='corridor', corridor_model_score=None)
    selected = mask[low:high] == 1
    if selected.any():
        result['corridor_model_score'] = float(probs[1, low:high][selected].mean())
    pairs = []
    centre_pair = None
    for yy in range(low, high):
        interval, reason = _corridor_interval(mask[yy])
        if yy == y:
            if reason:
                result['reason'] = reason
                return result
            centre_pair = interval
        if not reason:
            pairs.append(interval)
    if len(pairs) < max(1, int(np.ceil(.7*(high-low)))):
        result['reason'] = 'neighbouring_rows_inconsistent'
        return result
    if np.max(np.ptp(np.asarray(pairs), axis=0)) > max(4, .015*w):
        result['reason'] = 'edges_unstable_in_local_band'
        return result
    if result['corridor_model_score'] < min_score:
        result['reason'] = 'low_uncalibrated_model_score'
        return result
    left, right = centre_pair
    result.update(left_inner_x=left, right_inner_x=right,
                  apparent_width_px=right-left, center_x_px=(left+right)/2,
                  center_y_px=y, status='candidate',
                  reason='indicative_corridor_pixel_measurement_not_validated')
    return result


def draw_telemetry(image, result, seconds, repeated=False,smoothed=False,frame_info=None):
    """Readable panel with explicit missing units/accuracy and no invented numbers."""
    h,w = image.shape[:2]
    # A six-line panel cannot be legible on tiny software-test videos.
    # Measurements are still exported to CSV at every resolution.
    if w < 480 or h < 240:
        return image
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(.30,min(.58,w/1800)) if frame_info else max(.30,min(.8,w/1600))
    spacing = max(14,round(29*scale/.8))
    top = 30
    panel_height = min(h-top, spacing*(8 if frame_info else 7)+12)
    if panel_height <= 0:
        return image
    shade = image[top:top+panel_height].astype(np.float32)*.20
    image[top:top+panel_height] = shade.astype(np.uint8)
    width_text = (f"{result['apparent_width_px']:.0f} px (indicatif)"
                  if result['apparent_width_px'] is not None else 'non disponible')
    corridor = result.get('representation') == 'corridor'
    if corridor:
        value = result.get('corridor_model_score')
        score_text = f'Couloir {100*value:.1f}%' if value is not None else 'Couloir n/d'
    else:
        scores=[]
        for side,short in [('left','G'),('right','D')]:
            value = result['model_score_'+side]
            scores.append(f'{short} {100*value:.1f}%' if value is not None else f'{short} n/d')
        score_text = ' / '.join(scores)
    lines=[f'Temps : {seconds:.2f} s | ligne y={result["row_y"]}',
           ('Largeur lisse du couloir : ' if corridor and smoothed else 'Largeur du couloir : ' if corridor else 'Ecartement apparent : ')+width_text,
           'Metres : n/d - calibration absente',
           'Score modele : '+score_text+' (non calibre)',
           'Precision sur cette image : n/d - reference manuelle absente',
           (f'Axe vert : milieu des bords | x={result["center_x_px"]:.1f} px'
            if result.get('center_x_px') is not None else 'Axe vert : portions detectees uniquement'),
           'Etat : image repetee, pas de nouvelle mesure' if repeated else
           ('Etat : mesure indicative, a verifier' if result['status']=='candidate' else
            'Etat : mesure rejetee - '+result['reason'])]
    if smoothed:
        lines[5]=lines[5].replace('milieu des bords','milieu lisse des bords')
    if frame_info:
        lines.insert(1, frame_info)
    for i,line in enumerate(lines):
        y=top+spacing*(i+1)
        if y < h:
            cv2.putText(image,line,(8,y),font,scale,(255,255,255),1,cv2.LINE_AA)
    if result['apparent_width_px'] is not None:
        y=result['row_y'];left=round(result['left_inner_x']);right=round(result['right_inner_x'])
        cv2.line(image,(left,y),(right,y),(255,255,0),2,cv2.LINE_AA)
        for x in [left,right]:
            cv2.circle(image,(x,y),4,(255,255,0),-1)
    return image
