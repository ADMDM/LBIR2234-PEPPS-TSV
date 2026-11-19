"""Observed image-space rail pair and exact midpoints, with explicit gaps."""
import cv2
import numpy as np
from .telemetry import _interval, _corridor_interval


def axis_profile(mask, probabilities, min_score=.55, step=8):
    mask=np.asarray(mask);p=np.asarray(probabilities)
    if mask.ndim!=2 or p.shape!=(3,*mask.shape) or step<1:
        raise ValueError('Dimensions ou pas de l’axe invalides')
    if not 0<=min_score<=1 or not np.isfinite(p).all():
        raise ValueError('Score ou probabilités invalides')
    h,w=mask.shape;segments=[];current=[]
    for y in range(0,h,step):
        left,why_l=_interval(mask[y],1);right,why_r=_interval(mask[y],2)
        valid=not (why_l or why_r) and left[1]<right[0]
        if valid:
            scores=[float(p[label,y,mask[y]==label].mean()) for label in [1,2]]
            valid=min(scores)>=min_score
        if valid:
            point=dict(y=y,left_inner_x=left[1],right_inner_x=right[0],
                       left_outer_x=left[0],right_outer_x=right[1],
                       center_x=(left[1]+right[0])/2,apparent_width_px=right[0]-left[1])
            if current and max(abs(point[k]-current[-1][k]) for k in ['left_inner_x','right_inner_x'])>max(8,.025*w):
                if len(current)>=3:segments.append(current)
                current=[]
            current.append(point)
        elif current:
            if len(current)>=3:segments.append(current)
            current=[]
    if len(current)>=3:segments.append(current)
    return dict(segments=segments,points=sum(len(s) for s in segments),
                definition='Exact image-row midpoint of predicted inner-edge pixel centres',
                units='original image pixels',inferred_hidden_parts=False)


def corridor_profile(mask, probabilities, min_score=.55, step=8):
    """Exact midpoint of one observed corridor interval per image row.

    Check every intervening row even when storing only one point every `step`
    rows. Missing, ambiguous, low-score or cut portions therefore stay gaps.
    No rail thickness, extrapolation, physical gauge or centreline smoothing.
    """
    mask, probs = np.asarray(mask), np.asarray(probabilities)
    if mask.ndim != 2 or probs.shape != (2, *mask.shape) or not mask.size or step < 1:
        raise ValueError('Dimensions ou pas de l’axe du couloir invalides')
    if not 0 <= min_score <= 1 or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError('Score ou probabilités invalides')
    h, w = mask.shape
    segments, current, previous = [], [], None
    for y in range(h):
        interval, reason = _corridor_interval(mask[y])
        valid = not reason
        if valid:
            valid = float(probs[1, y, mask[y] == 1].mean()) >= min_score
        if valid and previous is not None:
            if max(abs(a-b) for a, b in zip(interval, previous)) > max(8, .025*w):
                if len(current) >= 3:
                    segments.append(current)
                current = []
        if valid:
            left, right = interval
            if y % step == 0:
                current.append(dict(y=y, left_inner_x=left, right_inner_x=right,
                                    center_x=(left+right)/2,
                                    apparent_width_px=right-left))
            previous = interval
        else:
            if len(current) >= 3:
                segments.append(current)
            current, previous = [], None
    if len(current) >= 3:
        segments.append(current)
    return dict(segments=segments, points=sum(len(s) for s in segments),
                representation='corridor',
                definition='Exact image-row midpoint of predicted corridor-boundary pixel centres',
                units='original image pixels', inferred_hidden_parts=False)


def draw_axis(image,profile):
    for segment in profile['segments']:
        points=np.asarray([[round(p['center_x']),p['y']] for p in segment],np.int32)
        cv2.polylines(image,[points],False,(60,255,90),2,cv2.LINE_AA)
    return image


def draw_corridor_limits(image, profile):
    """Thin observed corridor boundaries; not metallic outer rail edges."""
    for segment in profile['segments']:
        for key in ['left_inner_x', 'right_inner_x']:
            points = np.asarray([[p[key], p['y']] for p in segment], np.int32)
            cv2.polylines(image, [points], False, (255, 210, 80), 1, cv2.LINE_AA)
    return image


class TemporalGuard:
    """Reject an abrupt image-space jump; do not replace observed coordinates.

    Thresholds are diagnostic heuristics, not calibrated motion or accuracy.
    Reacquire after three mutually consistent new candidates to avoid a stuck state.
    """
    def __init__(self,width):self.width=width;self.previous=None;self.pending=[]
    def check(self,section,seconds):
        if section['status']!='candidate':return 'no_candidate'
        pair=np.array([section['left_inner_x'],section['right_inner_x']],float)
        if self.previous is None:
            self.previous=(pair,seconds);return 'initial'
        prev,t=self.previous;limit=max(12,.04*self.width)*max(1,min(3,(seconds-t)/.1))
        if np.max(np.abs(pair-prev))<=limit:
            self.previous=(pair,seconds);self.pending=[];return 'consistent'
        if self.pending and np.max(np.abs(pair-self.pending[-1]))>max(12,.04*self.width):self.pending=[]
        self.pending.append(pair)
        if len(self.pending)>=3:
            self.previous=(pair,seconds);self.pending=[];return 'reacquired'
        return 'jump_rejected'
