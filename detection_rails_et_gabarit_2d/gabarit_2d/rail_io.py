"""Standalone rail CSV reading and rendering; inherited unchanged V1 geometry."""
import csv,math,itertools,hashlib
import cv2
import numpy as np

def finite(s):
    f=float(s)
    if not math.isfinite(f):raise ValueError(f'Non-finite value: {s!r}')
    return f


def boolean(s):
    if s=='True':return True
    if s=='False':return False
    raise ValueError(f'Boolean True/False required, got {s!r}')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


class FrameGroups:
    def __init__(self,handle,name):
        self.name=name;self.groups=itertools.groupby(csv.DictReader(handle),key=lambda r:int(r['frame_index']));self.pending=None;self._advance()
    def _advance(self):
        try:
            k,g=next(self.groups);self.pending=(k,list(g))
        except StopIteration:self.pending=None
    def take(self,index):
        if self.pending is None:return []
        k,rows=self.pending
        if k<index:raise ValueError(f'{self.name}: out-of-order or unmatched frame {k} before {index}')
        if k>index:return []
        self._advance();return rows


def verify_common(row,index,t,w,h,video_id,name):
    if int(row['frame_index'])!=index:raise ValueError(f'{name}: wrong frame index')
    if row['video_id']!=video_id:raise ValueError(f'{name}: wrong video ID')
    if abs(finite(row['time_video_s'])-t)>1e-9:raise ValueError(f'{name}: wrong PTS at {index}')
    if (int(row['image_width']),int(row['image_height']))!=(w,h):raise ValueError(f'{name}: wrong dimensions')


def display_segments(rows):
    segments=[];segment=[];last_id=None;last_y=None
    for row in rows:
        sid=int(row['segment_index']);y=finite(row['y_px'])
        if sid!=last_id or not boolean(row['display_valid']):
            if segment:segments.append(segment)
            segment=[];last_y=None
        last_id=sid
        if not boolean(row['display_valid']):continue
        if last_y is not None and y<=last_y:raise ValueError('Unsorted axis rows')
        left,right,axis=map(finite,[row['left_display_px'],row['right_display_px'],row['axis_display_px']])
        if abs(axis-(left+right)/2)>1e-6:raise ValueError('CSV display axis is not the boundary midpoint')
        segment.append((left,right,axis,y));last_y=y
    if segment:segments.append(segment)
    return segments


def draw_corridor(image,segments):
    layer=image.copy();h,w=image.shape[:2]
    for segment in segments:
        for left,right,axis,y in segment:
            if not (0<=left<right<w and 0<=y<h):raise ValueError('Corridor out of bounds')
        if len(segment)>=2:
            l=np.array([[round(p[0]),round(p[3])] for p in segment],np.int32)
            r=np.array([[round(p[1]),round(p[3])] for p in segment],np.int32)
            cv2.fillPoly(layer,[np.concatenate([l,r[::-1]])],(210,160,30))
    cv2.addWeighted(layer,.13,image,.87,0,image)
    for segment in segments:
        for field,color,thickness in [(0,(255,215,70),2),(1,(255,215,70),2),(2,(0,245,255),3)]:
            points=np.array([[round(p[field]),round(p[3])] for p in segment],np.int32)
            if len(points)>=2:cv2.polylines(image,[points],False,color,thickness,cv2.LINE_AA)
            elif len(points)==1:cv2.circle(image,tuple(points[0]),1,color,thickness,cv2.LINE_AA)
