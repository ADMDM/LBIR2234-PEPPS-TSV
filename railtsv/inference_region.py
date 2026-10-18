"""Explicit source-image region, with predictions restored to original pixels."""
import numpy as np


def predict_region(model, image, device, height, width, roi, predictor):
    h,w=image.shape[:2]
    if len(roi)!=4 or any(type(v) is not int for v in roi):
        raise ValueError('ROI : quatre entiers x0 y0 x1 y1 requis')
    x0,y0,x1,y1=roi
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):
        raise ValueError('ROI hors de l’image ou vide ; borne finale exclusive')
    mask,probs=predictor(model,image[y0:y1,x0:x1],device,height,width)
    full=np.zeros((h,w),mask.dtype)
    full_probs=np.zeros((probs.shape[0],h,w),probs.dtype);full_probs[0]=1
    full[y0:y1,x0:x1]=mask;full_probs[:,y0:y1,x0:x1]=probs
    return full,full_probs
