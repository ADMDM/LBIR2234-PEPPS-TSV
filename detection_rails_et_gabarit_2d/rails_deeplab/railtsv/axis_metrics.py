"""Image-space axis errors against reviewed corridor references."""
import numpy as np
from .geometry import corridor_profile


class AxisMetrics:
    def __init__(self):
        self.errors=[];self.reference_points=0;self.matched_points=0;self.images=[]
        self.signed_errors=[];self.normalized_errors=[]

    def update(self,prediction,probabilities,reference,source=None):
        prediction=np.asarray(prediction);reference=np.asarray(reference)
        if prediction.shape!=reference.shape or reference.ndim!=2:
            raise ValueError('Dimensions de référence et prédiction incompatibles')
        if not set(np.unique(reference)).issubset({0,1,255}):
            raise ValueError('Référence de couloir attendue : 0,1,255')
        # These one-hot values only extract geometry from the annotation;
        # they are never presented as model confidence or reference certainty.
        ref_prob=np.stack([(reference!=1).astype(np.float32),(reference==1).astype(np.float32)])
        ref_profile=corridor_profile(reference,ref_prob)
        pred_profile=corridor_profile(prediction,probabilities)
        targets={p['y']:p['center_x'] for s in ref_profile['segments'] for p in s
                 if not np.any(reference[p['y'],max(0,p['left_inner_x']-1):p['right_inner_x']+2]==255)}
        predicted={p['y']:p['center_x'] for s in pred_profile['segments'] for p in s}
        signed=[predicted[y]-x for y,x in targets.items() if y in predicted]
        errors=[abs(e) for e in signed]
        normalized=[100*e/reference.shape[1] for e in signed]
        self.signed_errors.extend(signed);self.normalized_errors.extend(normalized)
        self.errors.extend(errors);self.reference_points+=len(targets);self.matched_points+=len(errors)
        self.images.append(dict(source=source,dimensions=list(reference.shape[::-1]),
            reference_points=len(targets),matched_points=len(errors),
            coverage_pct=100*len(errors)/len(targets) if targets else None,
            mean_absolute_center_error_px=float(np.mean(errors)) if errors else None,
            signed_center_bias_px=float(np.mean(signed)) if signed else None,
            root_mean_square_center_error_px=float(np.sqrt(np.mean(np.square(signed)))) if signed else None,
            mean_absolute_center_error_pct_width=float(np.mean(np.abs(normalized))) if normalized else None))

    def result(self):
        return dict(reference_points=self.reference_points,matched_points=self.matched_points,
            coverage_pct=100*self.matched_points/self.reference_points if self.reference_points else None,
            mean_absolute_center_error_px=float(np.mean(self.errors)) if self.errors else None,
            median_absolute_center_error_px=float(np.median(self.errors)) if self.errors else None,
            p95_absolute_center_error_px=float(np.percentile(self.errors,95)) if self.errors else None,
            max_absolute_center_error_px=float(np.max(self.errors)) if self.errors else None,
            signed_center_bias_px=float(np.mean(self.signed_errors)) if self.signed_errors else None,
            root_mean_square_center_error_px=float(np.sqrt(np.mean(np.square(self.signed_errors)))) if self.signed_errors else None,
            mean_absolute_center_error_pct_width=float(np.mean(np.abs(self.normalized_errors))) if self.normalized_errors else None,
            error_aggregation='Matched sections pooled; each normalized by its own image width. Not independent samples.',
            units='original_image_pixels',sample_step_y=8,
            definition='Horizontal centre deviation on matching observed rows; gaps counted in coverage',
            note='Reference polygon quality limits accuracy; not a physical 3D centreline or real-time accuracy percentage',
            per_image=self.images)
