#!/usr/bin/env python3
"""Native-frame instance masks intersected with an unchanged observed rail corridor.

TXT files are JSON documents with exact contour pixel centres and hierarchy.
No box is ever substituted for an object mask. Missing rail support is unknown.
"""
from __future__ import annotations
import argparse
import csv
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import av
import cv2
import numpy as np
from corridor_geometry import corridor_from_rows, analyze_object_mask, extract_contours

COMMON = ['video_id','frame_index','time_video_s','source_pts','time_base_num','time_base_den','image_width','image_height']
OBJECT_FIELDS = COMMON + ['observation_id','track_id','track_confirmed','observed_hits','model_source','class_id','class_name','confidence_score','mask_quality_score','mask_pixel_count','support_pixel_count','unsupported_pixel_count','support_fraction','support_status','overlap_pixel_count','overlap_fraction','overlap_fraction_supported','relation_2d','relation_reason','complete_relation_assessment','touches_image_border','touches_crop_border','mask_x_min_px','mask_y_min_px','mask_x_max_px_exclusive','mask_y_max_px_exclusive','full_mask_sha256','intersection_mask_sha256','full_contour_count','intersection_contour_count','txt_file','distance_m','actual_error_px','physical_danger_confirmed']
POINT_FIELDS = COMMON + ['observation_id','track_id','model_source','class_name','confidence_score','geometry_kind','contour_id','component_id','parent_contour_id','is_hole','depth','closed','boundary_kind','point_index','x_px','y_px']
FRAME_FIELDS = COMMON + ['processing_status','rail_diagnostic_status','rail_supported_rows','rail_corridor_pixel_count','raw_public_instances','raw_litter_instances','merged_instances','suppressed_duplicate_instances','empty_mask_instances','overlap_candidates','confirmed_overlap_candidates','indeterminate_instances','txt_file']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def mask_sha(mask):
    return hashlib.sha256(np.ascontiguousarray(mask,dtype=np.uint8).tobytes()).hexdigest()


def write_json(path,value):
    temp=Path(path).with_suffix(Path(path).suffix+'.part')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    temp.replace(path)


def crop_for_corridor(corridor,padding=.15):
    yy,xx=np.nonzero(corridor)
    if not len(xx):return None
    h,w=corridor.shape
    x0,x1=int(xx.min()),int(xx.max())+1
    y0,y1=int(yy.min()),int(yy.max())+1
    px=max(32,math.ceil((x1-x0)*padding));py=max(16,math.ceil((y1-y0)*padding))
    return [max(0,x0-px),max(0,y0-py),min(w,x1+px),min(h,y1+py)]


def predict_instances(model,bgr,source,imgsz,confidence,device,crop=None):
    h,w=bgr.shape[:2]
    x0,y0,x1,y1=crop or [0,0,w,h]
    view=bgr[y0:y1,x0:x1]
    result=model.predict(view,imgsz=imgsz,conf=confidence,iou=.5,nms=True,
                         retina_masks=True,max_det=100,device=device,verbose=False)[0]
    if bool(model.predictor.model.end2end):raise ValueError('Conventional NMS backend required')
    boxes=result.boxes.data.cpu().numpy()
    if not len(boxes):
        if result.masks is not None and len(result.masks):raise ValueError('Masks without boxes')
        return [],0
    if result.masks is None:raise ValueError('Detector returned boxes without instance masks')
    masks=result.masks.data.cpu().numpy()
    if masks.shape!=(len(boxes),y1-y0,x1-x0):raise ValueError('Native mask dimensions mismatch')
    if not np.isin(masks,(0,1)).all():raise ValueError('Expected binary native masks, not unthresholded logits')
    instances=[];empty=0
    for index,(box,local) in enumerate(zip(boxes,masks)):
        cls=int(box[5]);score=float(box[4])
        if not np.isfinite(box).all() or not 0<=score<=1 or float(box[5])!=cls:raise ValueError('Invalid prediction')
        name=str(result.names[cls])
        local=local.astype(bool,copy=False)
        if not local.any():empty+=1;continue
        mask=np.zeros((h,w),dtype=bool);mask[y0:y1,x0:x1]=local
        yy,xx=np.nonzero(mask)
        bounds=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]
        touches_crop=bool(crop and ((x0>0 and local[:,0].any()) or (x1<w and local[:,-1].any()) or (y0>0 and local[0,:].any()) or (y1<h and local[-1,:].any())))
        instances.append({'model_source':source,'class_id':cls,'class_name':name,'confidence_score':score,
                          'raw_detection_index':index,'mask':mask,'mask_bbox':bounds,
                          'model_bbox_original_px':[float(box[0]+x0),float(box[1]+y0),float(box[2]+x0),float(box[3]+y0)],
                          'inference_crop_xyxy':crop,'touches_crop_border':touches_crop})
    return instances,empty


def merge_instances(public,litter,threshold=.6):
    """Keep established public masks over duplicate litter hypotheses, not score calibration."""
    kept=list(public);audit=[]
    for obj in litter:
        duplicate=None
        for j,ref in enumerate(public):
            a,b=obj['mask_bbox'],ref['mask_bbox']
            if min(a[2],b[2])<=max(a[0],b[0]) or min(a[3],b[3])<=max(a[1],b[1]):continue
            inter=int(np.count_nonzero(obj['mask'] & ref['mask']))
            union=int(np.count_nonzero(obj['mask'] | ref['mask']))
            iou=inter/union if union else 0.0
            if iou>threshold:
                duplicate={'litter_raw_detection_index':obj['raw_detection_index'],'public_raw_detection_index':ref['raw_detection_index'],'mask_iou':iou,'rule':'public_mask_precedence_for_cross_model_duplicate'}
                break
        if duplicate:audit.append(duplicate)
        else:kept.append(obj)
    return kept,audit


class MaskTracks:
    """Associate observed masks by their bounding extents; never extrapolate contours."""
    def __init__(self,minimum_confidence=.25):
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker
        self.Boxes=Boxes;self.Tracker=BYTETracker
        self.args=SimpleNamespace(track_high_thresh=minimum_confidence,track_low_thresh=.1,new_track_thresh=minimum_confidence,track_buffer=18,match_thresh=.8,fuse_score=False)
        self.trackers={};self.hits={};self.generation=0;self.previous_time=None
    def update(self,objects,bgr,index,t):
        if self.previous_time is not None and t-self.previous_time>.5:
            self.generation+=1;self.trackers={};self.hits={}
        self.previous_time=t
        keys={(o['model_source'],o['class_id']) for o in objects}
        for key in keys:
            if key not in self.trackers:
                tracker=self.Tracker(self.args);tracker.frame_id=index;self.trackers[key]=tracker
        for obj in objects:obj.update(track_id=None,track_confirmed=False,observed_hits=0)
        for key,tracker in self.trackers.items():
            selected=[i for i,o in enumerate(objects) if (o['model_source'],o['class_id'])==key]
            values=np.array([[*objects[i]['mask_bbox'],objects[i]['confidence_score'],key[1]] for i in selected],dtype=np.float32).reshape(-1,6)
            boxes=self.Boxes(values,bgr.shape[:2])
            associated=tracker.update(boxes,bgr)
            # ByteTrack withholds new provisional tracks after frame one.
            # Count their current observation too, without emitting a predicted mask.
            for current in tracker.tracked_stracks:
                if current.frame_id==tracker.frame_id:
                    uid=f'{self.generation}:{key[0]}:{key[1]}:{int(current.track_id)}'
                    self.hits[uid]=self.hits.get(uid,0)+1
            for tr in associated:
                local=int(tr[-1])
                if not 0<=local<len(selected):raise ValueError('Tracker index outside current mask observations')
                obj=objects[selected[local]];uid=f'{self.generation}:{key[0]}:{key[1]}:{int(tr[4])}'
                hits=self.hits[uid]
                obj.update(track_id=uid,track_confirmed=hits>=2,observed_hits=hits)


def write_points(writer,base,obj,geometry,contours):
    count=0
    for c in contours:
        fields={k:c[k] for k in ['contour_id','component_id','parent_contour_id','is_hole','depth','closed','boundary_kind']}
        for j,(x,y) in enumerate(c['points']):
            writer.writerow(dict(base,observation_id=obj['observation_id'],track_id=obj['track_id'],model_source=obj['model_source'],class_name=obj['class_name'],confidence_score=obj['confidence_score'],geometry_kind=geometry,**fields,point_index=j,x_px=x,y_px=y))
            count+=1
    return count


def draw_result(frame,segments,candidates,t,draw_corridor):
    canvas=frame.copy();draw_corridor(canvas,segments)
    red=np.zeros(frame.shape[:2],bool)
    for obj in candidates:red|=obj['_intersection_mask']
    canvas[red]=(.65*canvas[red]+.35*np.array([30,50,245])).astype(np.uint8)
    for ordinal,obj in enumerate(candidates,1):
        contours=[np.array(c['points'],np.int32).reshape(-1,1,2) for c in obj['full_predicted_mask_contours']]
        if contours:cv2.drawContours(canvas,contours,-1,(20,145,255),2,cv2.LINE_AA)
        x,y=obj['mask_bbox'][:2];y=max(105,y-7);x=min(x,frame.shape[1]-250)
        label=f"{ordinal} {obj['class_name']} {obj['confidence_score']:.2f}"
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.58,1)
        cv2.rectangle(canvas,(x,y-th-4),(x+tw+5,y+4),(25,30,35),-1)
        cv2.putText(canvas,label,(x+2,y),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,230,210),1,cv2.LINE_AA)
    cv2.rectangle(canvas,(0,0),(frame.shape[1],84),(22,29,36),-1)
    cv2.putText(canvas,f'COULOIR 2D | {int(t)//60:02d}:{t%60:05.2f} | {len(candidates)} observations dans le couloir',(18,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(245,245,245),2,cv2.LINE_AA)
    cv2.putText(canvas,'Limites cyan | Axe jaune | Contours orange | Rouge : partie du masque dans le couloir',(18,56),cv2.FONT_HERSHEY_SIMPLEX,.57,(225,235,240),1,cv2.LINE_AA)
    cv2.putText(canvas,'Recouvrement dans l image, pas un danger physique confirme | Score != precision mesuree',(18,76),cv2.FONT_HERSHEY_SIMPLEX,.45,(200,215,220),1,cv2.LINE_AA)
    return canvas


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rail-delivery',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--public-model',type=Path,required=True);p.add_argument('--litter-model',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='auto',choices=['auto','mps','cpu'])
    p.add_argument('--public-imgsz',type=int,default=960);p.add_argument('--litter-imgsz',type=int,default=640)
    p.add_argument('--public-conf',type=float,default=.25);p.add_argument('--litter-conf',type=float,default=.25)
    p.add_argument('--max-frames',type=int,default=0);p.add_argument('--litter-crop-padding',type=float,default=.15)
    a=p.parse_args()
    if a.max_frames<0 or not 0<a.public_conf<=1 or not 0<a.litter_conf<=1:p.error('Invalid thresholds or frame limit')
    if a.out.exists():raise FileExistsError(a.out)
    os.environ.setdefault('YOLO_AUTOINSTALL','false');os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(tempfile.gettempdir())/'gabarit2d_yolo'))
    from rail_io import FrameGroups,verify_common,display_segments,draw_corridor
    import torch
    from ultralytics import YOLO,settings
    settings.update({'sync':False});torch.set_num_threads(4);cv2.setNumThreads(0)
    device=('mps' if torch.backends.mps.is_available() else 'cpu') if a.device=='auto' else a.device
    if device=='mps':
        if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable in this execution context')
        torch.mps.set_per_process_memory_fraction(.5)
    mp=a.rail_delivery/'SOURCE_ET_PARAMETRES.json';meta=json.loads(mp.read_text())
    if meta.get('status')!='complete':raise ValueError('Complete rail source required')
    source=(a.source or Path(meta['source'])).resolve();axes_path=a.rail_delivery/'AXE_ET_LIMITES.csv';frames_path=a.rail_delivery/'FRAMES.csv'
    paths=[source,axes_path,frames_path,mp,a.public_model,Path(__file__),Path(__file__).with_name('corridor_geometry.py'),Path(__file__).with_name('rail_io.py')]
    if a.litter_model:paths.append(a.litter_model)
    inputs={str(q.resolve()):sha(q) for q in paths}
    if inputs[str(source)]!=meta['source_sha256'] or inputs[str(axes_path.resolve())]!=meta['axis_export_sha256']:raise ValueError('Source or rail export hash mismatch')
    public=YOLO(str(a.public_model),task='segment');litter=YOLO(str(a.litter_model),task='segment') if a.litter_model else None
    if litter is not None and dict(litter.names)!={0:'dechet'}:raise ValueError('Expected one-class trained litter model')
    if public.task!='segment' or (litter is not None and litter.task!='segment'):raise ValueError('Instance segmentation weights required')
    a.out.mkdir(parents=True);(a.out/'TXT').mkdir();(a.out/'APERÇUS').mkdir()
    manifest={'status':'running','schema_version':1,'source':str(source),'source_sha256':meta['source_sha256'],'video_id':meta['video_id'],'device':device,'inputs_sha256':inputs,'parameters':{k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items()},'model_names':{'public':public.names,'litter':litter.names if litter else None},'geometry':'real native instance masks; full contours and separately named corridor intersections; original pixel centres','scope':'2D observed overlap, not calibrated physical danger','clock':'native PTS, no resampling or duplication','rails_recomputed':False,'unknown_error_and_distance':'null/empty, never zero','txt_encoding':'UTF-8 JSON document per frame, including empty objects list','tracking':'ByteTrack by source/class; track_id optional until associated; confirmed after two observed hits; no extrapolated masks; reset after gap >0.5s','merge_rule':'Public mask precedence if cross-model mask IoU >0.6; suppressed hypotheses listed in AUDIT_FUSION.jsonl','partial_source':bool(a.max_frames and a.max_frames<meta['frames'])}
    manifest['instance_prediction_settings']={'max_det_per_model_per_frame':100,'nms':True,'nms_iou':.5,'native_retina_masks':True}
    manifest['effective_cpu_threads']={'torch':torch.get_num_threads(),'opencv':cv2.getNumThreads()}
    manifest['runtime_scope']='from prepared output files/models to end of video/CSV writing and final input hashing; excludes initial model loading and initial hashes'
    write_json(a.out/'MANIFESTE.json',manifest)
    started=time.perf_counter();n=0;objects_total=points_total=0;times=[];output=None
    trackers=MaskTracks(min(a.public_conf,a.litter_conf));examples={0,147,1242,2022,2948,4105,4422,5160,5307}
    try:
        with ExitStack() as stack:
            input_frames=csv.DictReader(stack.enter_context(frames_path.open(newline='')))
            axes=FrameGroups(stack.enter_context(axes_path.open(newline='')),'AXE_ET_LIMITES.csv')
            with axes_path.open(newline='') as f:axis_fields=next(csv.reader(f))
            writers={}
            for name,fields in [('OBJETS.csv',OBJECT_FIELDS),('POINTS_OBJETS.csv',POINT_FIELDS),('FRAMES.csv',FRAME_FIELDS),('RAILS.csv',axis_fields)]:
                handle=stack.enter_context((a.out/name).open('x',newline=''));writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writers[name]=writer
            audit=stack.enter_context((a.out/'AUDIT_FUSION.jsonl').open('x'))
            src=stack.enter_context(av.open(str(source)));vs=src.streams.video[0]
            output=av.open(str(a.out/'VIDEO_GABARIT_2D.mp4'),'w',options={'movflags':'+faststart'})
            stream=output.add_stream('libx264',rate=vs.average_rate or Fraction(30,1));stream.width=meta['width'];stream.height=meta['height'];stream.pix_fmt='yuv420p';stream.time_base=Fraction(1,1000000);stream.codec_context.time_base=stream.time_base;stream.options={'crf':'23','preset':'veryfast','maxrate':'10M','bufsize':'20M'}
            for index,frame in enumerate(src.decode(vs)):
                if a.max_frames and index>=a.max_frames:break
                if frame.pts is None or frame.time_base is None:raise ValueError('Source PTS absent')
                t=float(frame.pts*frame.time_base);h,w=frame.height,frame.width
                if times and t<=times[-1]:raise ValueError('Non-increasing PTS')
                row=next(input_frames,None)
                if row is None:raise ValueError('Frame CSV exhausted')
                verify_common(row,index,t,w,h,meta['video_id'],'FRAMES.csv')
                rail_rows=axes.take(index)
                for rr in rail_rows:verify_common(rr,index,t,w,h,meta['video_id'],'AXE_ET_LIMITES.csv');writers['RAILS.csv'].writerow(rr)
                corridor,support,rail_info=corridor_from_rows(rail_rows,w,h)
                segments=display_segments(rail_rows)
                if sum(map(len,segments))!=int(row['rail_points_displayed']):raise ValueError('Rail display support mismatch')
                bgr=frame.to_ndarray(format='bgr24')
                pub,empty_pub=predict_instances(public,bgr,'public',a.public_imgsz,a.public_conf,device)
                crop=crop_for_corridor(corridor,a.litter_crop_padding)
                trash,empty_trash=predict_instances(litter,bgr,'litter',a.litter_imgsz,a.litter_conf,device,crop) if litter is not None and crop else ([],0)
                merged,suppressed=merge_instances(pub,trash);trackers.update(merged,bgr,index,t)
                base={'video_id':meta['video_id'],'frame_index':index,'time_video_s':t,'source_pts':int(frame.pts),'time_base_num':frame.time_base.numerator,'time_base_den':frame.time_base.denominator,'image_width':w,'image_height':h}
                candidates=[];indeterminate=0;raw_summaries=[]
                for di,obj in enumerate(merged):
                    analysis=analyze_object_mask(obj['mask'],corridor,support)
                    if analysis['relation_2d']=='indeterminate':indeterminate+=1
                    bx0,by0,bx1,by1=obj['mask_bbox']
                    raw_summaries.append({'model_source':obj['model_source'],'raw_detection_index':obj['raw_detection_index'],'class_name':obj['class_name'],'confidence_score':obj['confidence_score'],'mask_bbox':obj['mask_bbox'],'bbox_overlap_pixel_count':int(corridor[by0:by1,bx0:bx1].sum()),'relation_2d':analysis['relation_2d'],'overlap_pixel_count':analysis['overlap_pixel_count'],'support_status':analysis['support_status']})
                    if analysis['relation_2d']!='overlap_observed':continue
                    inside=obj['mask']&corridor
                    obj.update(observation_id=f"{meta['video_id']}:f{index:06d}:o{len(candidates):03d}",display_number=len(candidates)+1,analysis=analysis,
                               full_predicted_mask_contours=extract_contours(obj['mask']),corridor_intersection_contours=extract_contours(inside),
                               full_mask_sha256=mask_sha(obj['mask']),intersection_mask_sha256=mask_sha(inside),_intersection_mask=inside)
                    candidates.append(obj)
                txt_rel=f'TXT/frame_{index:06d}.txt'
                payload=dict(base,schema_version=1,processing_status='ok',coordinate_convention='original pixel centres, x right, y down',rail_diagnostic_status=row['rail_temporal_status'],rail_support=rail_info,objects=[])
                for obj in candidates:
                    record={k:v for k,v in obj.items() if k not in {'mask','_intersection_mask'}}
                    record.update(mask_quality_score=None,distance_m=None,actual_error_px=None,physical_danger_confirmed=None)
                    payload['objects'].append(record)
                    bbox=obj['mask_bbox'];ar=obj['analysis']
                    summary=dict(base,observation_id=obj['observation_id'],track_id=obj['track_id'],track_confirmed=obj['track_confirmed'],observed_hits=obj['observed_hits'],model_source=obj['model_source'],class_id=obj['class_id'],class_name=obj['class_name'],confidence_score=obj['confidence_score'],mask_quality_score=None,
                                 **{k:ar[k] for k in ['mask_pixel_count','support_pixel_count','unsupported_pixel_count','support_fraction','support_status','overlap_pixel_count','overlap_fraction','overlap_fraction_supported','relation_2d','relation_reason','complete_relation_assessment','touches_image_border']},
                                 touches_crop_border=obj['touches_crop_border'],mask_x_min_px=bbox[0],mask_y_min_px=bbox[1],mask_x_max_px_exclusive=bbox[2],mask_y_max_px_exclusive=bbox[3],full_mask_sha256=obj['full_mask_sha256'],intersection_mask_sha256=obj['intersection_mask_sha256'],full_contour_count=len(obj['full_predicted_mask_contours']),intersection_contour_count=len(obj['corridor_intersection_contours']),txt_file=txt_rel,distance_m=None,actual_error_px=None,physical_danger_confirmed=None)
                    writers['OBJETS.csv'].writerow(summary)
                    points_total+=write_points(writers['POINTS_OBJETS.csv'],base,obj,'full_predicted_mask',obj['full_predicted_mask_contours'])
                    points_total+=write_points(writers['POINTS_OBJETS.csv'],base,obj,'corridor_intersection',obj['corridor_intersection_contours'])
                with (a.out/txt_rel).open('x') as f:json.dump(payload,f,ensure_ascii=False,allow_nan=False,separators=(',',':'));f.write('\n')
                audit.write(json.dumps(dict(base,suppressed=suppressed,merged_predictions=raw_summaries,empty_masks=empty_pub+empty_trash),ensure_ascii=False,allow_nan=False)+'\n')
                writers['FRAMES.csv'].writerow(dict(base,processing_status='ok',rail_diagnostic_status=row['rail_temporal_status'],rail_supported_rows=rail_info['supported_rows'],rail_corridor_pixel_count=rail_info['corridor_pixel_count'],raw_public_instances=len(pub),raw_litter_instances=len(trash),merged_instances=len(merged),suppressed_duplicate_instances=len(suppressed),empty_mask_instances=empty_pub+empty_trash,overlap_candidates=len(candidates),confirmed_overlap_candidates=sum(o['track_confirmed'] for o in candidates),indeterminate_instances=indeterminate,txt_file=txt_rel))
                overlay=draw_result(bgr,segments,candidates,t,draw_corridor)
                if index in examples:
                    cv2.imwrite(str(a.out/'APERÇUS'/f'frame_{index:06d}.jpg'),overlay)
                vf=av.VideoFrame.from_ndarray(overlay,format='bgr24');vf.pts=round(t*1000000);vf.time_base=Fraction(1,1000000)
                for packet in stream.encode(vf):output.mux(packet)
                times.append(t);n+=1;objects_total+=len(candidates)
                if n%100==0:print(json.dumps({'frames':n,'video_time_s':t,'overlap_observations':objects_total,'elapsed_s':round(time.perf_counter()-started,1)}),flush=True)
            expected=min(a.max_frames,meta['frames']) if a.max_frames else meta['frames']
            if n!=expected:raise ValueError(f'Incomplete source: {n}/{expected}')
            full=n==meta['frames']
            if full and (next(input_frames,None) is not None or axes.pending is not None):raise ValueError('Unconsumed source CSV rows')
            for packet in stream.encode():output.mux(packet)
            output.close();output=None
        if full and sha(a.out/'RAILS.csv')!=meta['axis_export_sha256']:raise ValueError('Rail CSV changed during export')
        for filename,digest in inputs.items():
            if sha(filename)!=digest:raise ValueError(f'Input changed during inference: {filename}')
        manifest.update(status='complete',frames=n,first_time_s=times[0],last_time_s=times[-1],overlap_observations=objects_total,point_rows=points_total,rail_csv_byte_identical=full,elapsed_s=time.perf_counter()-started,
                        output_sha256={f:sha(a.out/f) for f in ['VIDEO_GABARIT_2D.mp4','OBJETS.csv','POINTS_OBJETS.csv','FRAMES.csv','RAILS.csv','AUDIT_FUSION.jsonl']})
    except Exception as exc:
        manifest.update(status='failed',frames=n,error=repr(exc));raise
    finally:
        if output is not None:output.close()
        write_json(a.out/'MANIFESTE.json',manifest)
    print(json.dumps({'status':manifest['status'],'frames':n,'overlap_observations':objects_total,'point_rows':points_total,'output':str(a.out)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
