"""Conservative masks between public annotated rail polylines.

This generic target does not certify TSV inner-rail faces. Only rows with
both individually continuous, ordered rail traces receive supervision.
"""
import numpy as np
from PIL import Image,ImageDraw


def egopath_mask(annotation, size):
    width,height=size
    traces=[]
    for key in ['left_rail','right_rail']:
        xy=np.asarray(annotation[key],float)
        if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<2 or not np.isfinite(xy).all():
            raise ValueError('Polyligne finie avec au moins deux points requise')
        if np.any(xy<0) or np.any(xy[:,0]>=width) or np.any(xy[:,1]>=height):
            raise ValueError('Polyligne hors image')
        im=Image.new('L',size,0);ImageDraw.Draw(im).line([tuple(x) for x in xy],fill=1,width=1)
        traces.append(np.asarray(im))
    mask=np.full((height,width),255,np.uint8)
    for y in range(height):
        left=np.flatnonzero(traces[0][y]);right=np.flatnonzero(traces[1][y])
        if not len(left) or not len(right) or np.any(np.diff(left)!=1) or np.any(np.diff(right)!=1):continue
        # Do not infer a complete cross-section through a nearly horizontal trace.
        if len(left)>8 or len(right)>8:continue
        lo,hi=round(float(np.median(left))),round(float(np.median(right)))
        if not 0<lo<hi<width-1:continue
        mask[y]=0;mask[y,lo:hi+1]=1
    if not np.any(mask==1):raise ValueError('Aucune paire de rails exploitable')
    return mask
