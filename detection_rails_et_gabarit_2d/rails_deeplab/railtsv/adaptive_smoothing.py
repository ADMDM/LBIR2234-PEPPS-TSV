"""Causal speed-adaptive smoothing of observed paired boundaries.

Inspired by Casiez et al. (CHI 2012), https://gery.casiez.net/1euro/ .
Coordinates and derivatives are normalized by image width. A shared alpha
keeps both edges ordered and the centre exactly at their midpoint.
This display filter neither recovers hidden track nor selects a switch route.
"""
from copy import deepcopy
import math


class AdaptiveProfileSmoother:
    def __init__(self, width, min_cutoff_hz=.8, beta=15., derivative_cutoff_hz=1.,
                 max_gap_seconds=.15, jump_fraction=.04):
        positive = [width, min_cutoff_hz, derivative_cutoff_hz, max_gap_seconds, jump_fraction]
        if width < 2 or not all(math.isfinite(x) and x > 0 for x in positive):
            raise ValueError('Dimensions et paramètres positifs finis requis')
        if not math.isfinite(beta) or beta < 0:
            raise ValueError('beta doit être fini et positif ou nul')
        self.width, self.min_cutoff, self.beta = width, min_cutoff_hz, beta
        self.derivative_cutoff = derivative_cutoff_hz
        self.max_gap, self.jump = max_gap_seconds, max(12., width*jump_fraction)
        self.previous, self.time, self.last = {}, None, None
        self.tau = None  # no fixed time constant

    @staticmethod
    def alpha(cutoff, dt):
        return 1 / (1 + 1 / (2*math.pi*cutoff*dt))

    def update(self, profile, seconds):
        if not math.isfinite(seconds) or (self.time is not None and seconds <= self.time):
            raise ValueError('Temps finis strictement croissants requis ; exclure les répétitions')
        result, current, counts = deepcopy(profile), {}, {}
        dt = None if self.time is None else seconds-self.time
        for segment in result['segments']:
            for point in segment:
                y = point['y']
                raw = (float(point['left_inner_x']), float(point['right_inner_x']))
                if y in current or not all(math.isfinite(x) for x in raw) or not 0 <= raw[0] < raw[1] < self.width:
                    raise ValueError('Profil de limites invalide')
                prev = self.previous.get(y)
                state = 'initial' if dt is None else 'new_row'
                filtered, derivative, alpha = raw, (0., 0.), 1.
                if dt is not None and dt > self.max_gap:
                    state = 'reset_gap'
                elif prev is not None:
                    old_raw, old_filtered, old_derivative = prev
                    if max(abs(a-b) for a,b in zip(raw, old_raw)) > self.jump:
                        state = 'reset_jump'
                    else:
                        ad = self.alpha(self.derivative_cutoff, dt)
                        derivative = tuple(ad*(a-b)/(self.width*dt)+(1-ad)*d
                                           for a,b,d in zip(raw, old_raw, old_derivative))
                        cutoff = self.min_cutoff + self.beta*max(abs(d) for d in derivative)
                        alpha = self.alpha(cutoff, dt)
                        filtered = tuple(alpha*a+(1-alpha)*b for a,b in zip(raw, old_filtered))
                        state = 'filtered'
                left, right = filtered
                point.update(raw_left_inner_x=raw[0], raw_right_inner_x=raw[1],
                    raw_center_x=sum(raw)/2, raw_apparent_width_px=raw[1]-raw[0],
                    left_inner_x=left, right_inner_x=right, center_x=(left+right)/2,
                    apparent_width_px=right-left, smoothing_status=state,
                    smoothing_alpha=alpha)
                current[y] = (raw, filtered, derivative)
                counts[state] = counts.get(state, 0)+1
        result.update(units='original_image_pixels', raw_predictions_modified=False,
            inferred_hidden_parts=False,
            definition='Shared speed-adaptive causal filter of observed paired boundaries',
            smoothing=dict(method='one_euro_paired', min_cutoff_hz=self.min_cutoff,
                           beta_normalized=self.beta, derivative_cutoff_hz=self.derivative_cutoff,
                           states=counts))
        self.previous, self.time, self.last = current, seconds, deepcopy(result)
        return result
