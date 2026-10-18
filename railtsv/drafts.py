"""Editable polygon suggestions; never claim that a prediction is reviewed."""
import cv2
import numpy as np
from .geometry import axis_profile,corridor_profile


def draft_annotation(mask,probabilities,image_name,checkpoint_sha256,min_score=.55,task='tsv'):
    h,w=mask.shape
    if task not in ('tsv','corridor'):raise ValueError('Task de suggestion incompatible')
    profile=(corridor_profile if task=='corridor' else axis_profile)(mask,probabilities,min_score)
    if profile['points']<12:return None
    shapes=[]
    labels=[(1,'couloir')] if task=='corridor' else [(1,'rail_left'),(2,'rail_right')]
    for value,label in labels:
        binary=((mask==value)&(probabilities[value]>=min_score)).astype(np.uint8)
        contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour)<max(50,w*h*.00005):continue
            polygon=cv2.approxPolyDP(contour,max(.6,min(w,h)*.001),True).reshape(-1,2)
            if len(polygon)<3:continue
            # Preserve separate visible components; humans add ignore where uncertain.
            shapes.append(dict(label=label,points=polygon.astype(float).tolist(),
                               group_id=None,shape_type='polygon',flags={},description='Suggestion du modèle, à corriger'))
    if {s['label'] for s in shapes}!={name for _,name in labels}:return None
    return dict(version='7.7.0',flags={'reviewed':False},shapes=shapes,
        imagePath='../images/'+image_name,imageData=None,imageHeight=h,imageWidth=w,
        description=('Couloir automatique non validé. Corriger ses deux limites et ajouter ignore avant reviewed.'
                     if task=='corridor' else 'Contours automatiques non validés. Corriger les rails et ajouter ignore avant reviewed.'),
        railtsv_provenance=dict(checkpoint_sha256=checkpoint_sha256,min_uncalibrated_score=min_score,
            manual_validation_required=True,kind='model_suggestion'))
