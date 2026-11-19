#!/usr/bin/env python3
"""Refine verified V1 masks with overlapping litter views, preserving native PTS.

V1 remains immutable. Quarantined track-like train hypotheses retain full contours
in TXT and A_VERIFIER.csv, but are not drawn as accepted objects.
"""
from __future__ import annotations
import argparse,csv,json,os,tempfile,time
from pathlib import Path
from contextlib import ExitStack
from fractions import Fraction
import av,cv2,numpy as np
from corridor_geometry import corridor_from_rows,extract_contours,reconstruct_mask,analyze_object_mask
from rail_io import FrameGroups,verify_common,display_segments,draw_corridor
from infer_gabarit_2d import (COMMON,OBJECT_FIELDS,POINT_FIELDS,FRAME_FIELDS,MaskTracks,
    sha,mask_sha,write_json,crop_for_corridor,merge_instances,write_points,draw_result)
from package_gabarit_2d import verified_txt_files
from small_object_refinement import tile_windows,predict_tiled_litter,merge_multiscale,train_guard

EXTRA_FRAME=['baseline_candidates','tile_instances','quarantined_candidates','tile_views']
REVIEW_FIELDS=OBJECT_FIELDS+['review_reason']


def baseline_objects(payload,w,h):
    result=[]
    for i,old in enumerate(payload['objects']):
        mask=reconstruct_mask(old['full_predicted_mask_contours'],w,h).astype(bool)
        if mask_sha(mask)!=old['full_mask_sha256']:raise ValueError('Baseline reconstructed mask differs')
        keys=['model_source','class_id','class_name','confidence_score','raw_detection_index',
              'mask_bbox','model_bbox_original_px','inference_crop_xyxy','touches_crop_border']
        obj={k:old[k] for k in keys};obj.update(mask=mask,candidate_id=f'b{i:04d}',
            inference_origin='baseline',baseline_observation_id=old['observation_id'],
            view_id='baseline',view_kind='global',tile_index=None)
        result.append(obj)
    return result


def summarize(base,obj,txt_rel):
    bbox=obj['mask_bbox'];ar=obj['analysis']
    return dict(base,observation_id=obj['observation_id'],track_id=obj['track_id'],
        track_confirmed=obj['track_confirmed'],observed_hits=obj['observed_hits'],model_source=obj['model_source'],
        class_id=obj['class_id'],class_name=obj['class_name'],confidence_score=obj['confidence_score'],mask_quality_score=None,
        **{k:ar[k] for k in ['mask_pixel_count','support_pixel_count','unsupported_pixel_count','support_fraction','support_status','overlap_pixel_count','overlap_fraction','overlap_fraction_supported','relation_2d','relation_reason','complete_relation_assessment','touches_image_border']},
        touches_crop_border=obj['touches_crop_border'],mask_x_min_px=bbox[0],mask_y_min_px=bbox[1],
        mask_x_max_px_exclusive=bbox[2],mask_y_max_px_exclusive=bbox[3],full_mask_sha256=obj['full_mask_sha256'],
        intersection_mask_sha256=obj['intersection_mask_sha256'],full_contour_count=len(obj['full_predicted_mask_contours']),
        intersection_contour_count=len(obj['corridor_intersection_contours']),txt_file=txt_rel,
        distance_m=None,actual_error_px=None,physical_danger_confirmed=None)


def decorate(obj,corridor,analysis,identifier):
    inside=obj['mask']&corridor
    obj.update(observation_id=identifier,display_number=int(identifier.rsplit(':',1)[1][1:])+1,analysis=analysis,full_predicted_mask_contours=extract_contours(obj['mask']),
        corridor_intersection_contours=extract_contours(inside),full_mask_sha256=mask_sha(obj['mask']),
        intersection_mask_sha256=mask_sha(inside),_intersection_mask=inside,mask_quality_score=None,
        distance_m=None,actual_error_px=None,physical_danger_confirmed=None)
    return obj


def serial_record(obj):
    return {k:v for k,v in obj.items() if k not in {'mask','_intersection_mask'}}


def merge_public_masks(public,litter,corridor=None):
    kept=list(public);audit=[]
    for obj in litter:
        matched=None
        for ref in public:
            if corridor is not None and train_guard(ref,corridor)['quarantine']:continue
            a,b=obj['mask_bbox'],ref['mask_bbox']
            if min(a[2],b[2])<=max(a[0],b[0]) or min(a[3],b[3])<=max(a[1],b[1]):continue
            inter=int(np.count_nonzero(obj['mask'] & ref['mask']));union=int(np.count_nonzero(obj['mask'] | ref['mask']))
            if union and inter/union>.6:
                matched=dict(rule='public_mask_precedence_for_cross_model_duplicate',
                    discarded_candidate_id=obj['candidate_id'],kept_candidate_id=ref['candidate_id'],
                    discarded_view_id=obj['view_id'],kept_view_id=ref['view_id'],mask_iou=inter/union)
                break
        if matched:audit.append(matched)
        else:kept.append(obj)
    return kept,audit


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['baseline','litter-model','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source',type=Path);p.add_argument('--device',choices=['auto','mps','cpu'],default='auto')
    p.add_argument('--tile-size',type=int,default=640);p.add_argument('--tile-overlap',type=float,default=.25)
    p.add_argument('--imgsz',type=int,default=960);p.add_argument('--conf',type=float,default=.15)
    p.add_argument('--max-frames',type=int,default=0)
    return p


def main():
    a=parser().parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    if a.max_frames<0 or a.tile_size<32 or not 0<=a.tile_overlap<1 or not 0<a.conf<=1:raise ValueError('Invalid parameters')
    os.environ.setdefault('YOLO_AUTOINSTALL','false');os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(tempfile.gettempdir())/'gabarit2d_yolo'))
    import torch
    from ultralytics import YOLO,settings
    settings.update({'sync':False});torch.set_num_threads(4);cv2.setNumThreads(0)
    device=('mps' if torch.backends.mps.is_available() else 'cpu') if a.device=='auto' else a.device
    if device=='mps':
        if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
        torch.mps.set_per_process_memory_fraction(.5)
    bp=a.baseline.resolve();bm=json.loads((bp/'MANIFESTE.json').read_text());bv=json.loads((bp/'VERIFICATION.json').read_text())
    if bm['status']!='complete' or bm['schema_version']!=1 or bm['partial_source'] or bv['status']!='passed':raise ValueError('Verified full V1 baseline required')
    if sha(bp/'MANIFESTE.json')!=bv['manifest_sha256']:raise ValueError('Baseline verifier/manifest mismatch')
    verified_txt_files(bp,bm,bv)
    for name,digest in bm['output_sha256'].items():
        if sha(bp/name)!=digest:raise ValueError('Baseline output changed: '+name)
    if abs(a.conf-bm['parameters']['litter_conf'])>1e-12:raise ValueError('Cached global masks require the original litter confidence threshold')
    source=(a.source or Path(bm['source'])).resolve()
    if sha(source)!=bm['source_sha256']:raise ValueError('Wrong native source video')
    model_path=a.litter_model.resolve()
    if sha(model_path)!=bm['inputs_sha256'][str(Path(bm['parameters']['litter_model']).resolve())]:raise ValueError('Litter checkpoint differs from controlled baseline')
    paths=[source,model_path]+[bp/n for n in ['MANIFESTE.json','VERIFICATION.json','FRAMES.csv','OBJETS.csv','RAILS.csv']]
    paths += [Path(__file__).with_name(n) for n in ['refine_video.py','small_object_refinement.py','corridor_geometry.py','rail_io.py','infer_gabarit_2d.py','package_gabarit_2d.py']]
    raildir=Path(bm['parameters']['rail_delivery'])
    frozen={str(p.resolve()):sha(p) for p in paths}
    litter=YOLO(str(model_path),task='segment')
    if dict(litter.names)!={0:'dechet'}:raise ValueError('Expected one-class litter checkpoint')
    a.out.mkdir(parents=True);(a.out/'TXT').mkdir();(a.out/'APERÇUS').mkdir()
    params={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items()}
    params.update(rail_delivery=str(raildir),public_model=bm['parameters']['public_model'],litter_imgsz=a.imgsz,litter_conf=a.conf,public_conf=bm['parameters']['public_conf'])
    manifest=dict(status='running',schema_version=2,source=str(source),source_sha256=bm['source_sha256'],video_id=bm['video_id'],device=device,
        inputs_sha256=frozen,parameters=params,model_names=bm['model_names'],rails_recomputed=False,
        baseline_manifest_sha256=sha(bp/'MANIFESTE.json'),baseline_txt_path_and_content_aggregate_sha256=bv['txt_path_and_content_aggregate_sha256'],
        baseline_reused='verified retained V1 native masks; public/global passes are not re-inferred',
        geometry='native observed instance masks, full contours and separately named corridor intersections',
        partial_source=bool(a.max_frames and a.max_frames<bm['frames']),clock='native PTS; no duplication/resampling',
        scope='2D candidate masks; quarantined hypotheses are not evidence of a clear track',
        guard_rule='class=train AND confidence<0.5 AND corridor_coverage>=0.9 AND height_fraction>=0.75 AND bottom_fraction>=0.95',
        quarantine_storage='TXT.quarantined_predictions full contours + A_VERIFIER.csv; excluded from accepted OBJETS/POINTS and red overlay',
        runtime_scope='output processing and final hashes, excluding baseline preflight and initial model loading',
        effective_cpu_threads={'torch':torch.get_num_threads(),'opencv':cv2.getNumThreads()})
    write_json(a.out/'MANIFESTE.json',manifest)
    n=objects_total=quarantine_total=points_total=0;times=[];output=None;start=time.perf_counter()
    trackers=MaskTracks(min(a.conf,params['public_conf']));examples={0,147,1242,1559,1765,1820,1890,2022,2948,4105,4422,5077,5097,5160,5307}
    try:
        with ExitStack() as stack:
            frames=csv.DictReader(stack.enter_context((bp/'FRAMES.csv').open(newline='')))
            axes=FrameGroups(stack.enter_context((bp/'RAILS.csv').open(newline='')),'RAILS.csv')
            with (bp/'RAILS.csv').open() as f:axis_fields=next(csv.reader(f))
            writers={}
            for name,fields in [('OBJETS.csv',OBJECT_FIELDS),('POINTS_OBJETS.csv',POINT_FIELDS),('A_VERIFIER.csv',REVIEW_FIELDS),('FRAMES.csv',FRAME_FIELDS+EXTRA_FRAME),('RAILS.csv',axis_fields)]:
                writer=csv.DictWriter(stack.enter_context((a.out/name).open('x',newline='')),fieldnames=fields);writer.writeheader();writers[name]=writer
            audit=stack.enter_context((a.out/'AUDIT_FUSION.jsonl').open('x'))
            src=stack.enter_context(av.open(str(source)));vs=src.streams.video[0]
            output=av.open(str(a.out/'VIDEO_GABARIT_2D.mp4'),'w',options={'movflags':'+faststart'})
            stream=output.add_stream('libx264',rate=vs.average_rate or Fraction(30));stream.width=vs.width;stream.height=vs.height;stream.pix_fmt='yuv420p';stream.time_base=Fraction(1,1000000);stream.codec_context.time_base=stream.time_base;stream.options={'crf':'23','preset':'veryfast','maxrate':'10M','bufsize':'20M'}
            for index,frame in enumerate(src.decode(vs)):
                if a.max_frames and index>=a.max_frames:break
                if frame.pts is None:raise ValueError('Missing PTS')
                t=float(frame.pts*frame.time_base);w,h=frame.width,frame.height
                if times and t<=times[-1]:raise ValueError('Non-increasing PTS')
                row=next(frames,None);verify_common(row,index,t,w,h,bm['video_id'],'baselineFRAMES')
                rails=axes.take(index)
                for rr in rails:verify_common(rr,index,t,w,h,bm['video_id'],'RAILS');writers['RAILS.csv'].writerow(rr)
                corridor,support,railinfo=corridor_from_rows(rails,w,h);segments=display_segments(rails)
                old=json.loads((bp/'TXT'/f'frame_{index:06d}.txt').read_text());verify_common(old,index,t,w,h,bm['video_id'],'baselineTXT')
                baseline=baseline_objects(old,w,h);pub=[o for o in baseline if o['model_source']=='public'];global_lit=[o for o in baseline if o['model_source']=='litter']
                bgr=frame.to_ndarray(format='bgr24');crop=crop_for_corridor(corridor,.15)
                tiles,empty=predict_tiled_litter(litter,bgr,crop,a.imgsz,a.conf,device,tile_size=a.tile_size,overlap=a.tile_overlap) if crop else ([],0)
                for i,obj in enumerate(tiles):obj.update(candidate_id=f't{i:04d}',inference_origin='tile',baseline_observation_id=None)
                trash,dedup=merge_multiscale(global_lit,tiles)
                merged,cross=merge_public_masks(pub,trash,corridor)
                base=dict(video_id=bm['video_id'],frame_index=index,time_video_s=t,source_pts=int(frame.pts),time_base_num=frame.time_base.numerator,time_base_den=frame.time_base.denominator,image_width=w,image_height=h)
                accepted=[];review=[];raw=[];indeterminate=0
                for obj in merged:
                    ar=analyze_object_mask(obj['mask'],corridor,support);guard=train_guard(obj,corridor)
                    if ar['relation_2d']=='indeterminate':indeterminate+=1
                    x0,y0,x1,y1=obj['mask_bbox']
                    raw.append(dict(candidate_id=obj['candidate_id'],inference_origin=obj['inference_origin'],model_source=obj['model_source'],raw_detection_index=obj['raw_detection_index'],class_name=obj['class_name'],confidence_score=obj['confidence_score'],mask_bbox=obj['mask_bbox'],bbox_overlap_pixel_count=int(corridor[y0:y1,x0:x1].sum()),relation_2d=ar['relation_2d'],overlap_pixel_count=ar['overlap_pixel_count'],support_status=ar['support_status'],review_reason=guard['reason'] if guard['quarantine'] else None))
                    if ar['relation_2d']!='overlap_observed':continue
                    if guard['quarantine']:
                        obj.update(track_id=None,track_confirmed=False,observed_hits=0,review_reason=guard['reason'],review_geometry=guard)
                        review.append(decorate(obj,corridor,ar,f"{bm['video_id']}:v2:f{index:06d}:q{len(review):03d}"))
                    else:accepted.append(decorate(obj,corridor,ar,f"{bm['video_id']}:v2:f{index:06d}:o{len(accepted):03d}"))
                trackers.update(accepted,bgr,index,t)
                txt_rel=f'TXT/frame_{index:06d}.txt'
                payload=dict(base,schema_version=2,processing_status='ok',coordinate_convention='original pixel centres, x right, y down',rail_diagnostic_status=row['rail_diagnostic_status'],rail_support=railinfo,objects=[serial_record(o) for o in accepted],quarantined_predictions=[serial_record(o) for o in review])
                for obj in accepted:
                    writers['OBJETS.csv'].writerow(summarize(base,obj,txt_rel))
                    points_total+=write_points(writers['POINTS_OBJETS.csv'],base,obj,'full_predicted_mask',obj['full_predicted_mask_contours'])
                    points_total+=write_points(writers['POINTS_OBJETS.csv'],base,obj,'corridor_intersection',obj['corridor_intersection_contours'])
                for obj in review:writers['A_VERIFIER.csv'].writerow(dict(summarize(base,obj,txt_rel),review_reason=obj['review_reason']))
                with (a.out/txt_rel).open('x') as f:json.dump(payload,f,ensure_ascii=False,allow_nan=False,separators=(',',':'));f.write('\n')
                windows=tile_windows(crop,w,h,tile_size=a.tile_size,overlap=a.tile_overlap) if crop else []
                tile_views=[dict(view_id=f'tile_{i:04d}',view_kind='tile',tile_index=i,crop_xyxy=win) for i,win in enumerate(windows)]
                audit.write(json.dumps(dict(base,suppressed=dedup+cross,merged_predictions=raw,empty_masks=empty,tile_views=tile_views),ensure_ascii=False,allow_nan=False)+'\n')
                writers['FRAMES.csv'].writerow(dict(base,processing_status='ok',rail_diagnostic_status=row['rail_diagnostic_status'],rail_supported_rows=railinfo['supported_rows'],rail_corridor_pixel_count=railinfo['corridor_pixel_count'],raw_public_instances=len(pub),raw_litter_instances=len(global_lit)+len(tiles),merged_instances=len(merged),suppressed_duplicate_instances=len(dedup)+len(cross),empty_mask_instances=empty,overlap_candidates=len(accepted),confirmed_overlap_candidates=sum(o['track_confirmed'] for o in accepted),indeterminate_instances=indeterminate,txt_file=txt_rel,baseline_candidates=len(baseline),tile_instances=len(tiles),quarantined_candidates=len(review),tile_views=len(windows)))
                overlay=draw_result(bgr,segments,accepted,t,draw_corridor)
                cv2.putText(overlay,f'V2 : vues rapprochees | {len(review)} hypothese(s) a verifier',(1050,30),cv2.FONT_HERSHEY_SIMPLEX,.55,(225,225,255),1,cv2.LINE_AA)
                if index in examples:cv2.imwrite(str(a.out/'APERÇUS'/f'frame_{index:06d}.jpg'),overlay)
                vf=av.VideoFrame.from_ndarray(overlay,format='bgr24');vf.pts=round(t*1000000);vf.time_base=stream.time_base
                for packet in stream.encode(vf):output.mux(packet)
                times.append(t);n+=1;objects_total+=len(accepted);quarantine_total+=len(review)
                if n%100==0:print(json.dumps(dict(frames=n,video_time_s=t,observations=objects_total,quarantined=quarantine_total,elapsed_s=round(time.perf_counter()-start,1))),flush=True)
            expected=min(a.max_frames,bm['frames']) if a.max_frames else bm['frames']
            if n!=expected:raise ValueError('Incomplete source decode')
            full=n==bm['frames']
            if full and (next(frames,None) is not None or axes.pending is not None):raise ValueError('Unconsumed baseline rows')
            for packet in stream.encode():output.mux(packet)
            output.close();output=None
        if full and sha(a.out/'RAILS.csv')!=sha(bp/'RAILS.csv'):raise ValueError('Rail export changed')
        for filename,digest in frozen.items():
            if sha(filename)!=digest:raise ValueError('Frozen input changed: '+filename)
        verified_txt_files(bp,bm,bv)
        manifest.update(status='complete',frames=n,first_time_s=times[0],last_time_s=times[-1],overlap_observations=objects_total,quarantined_observations=quarantine_total,point_rows=points_total,rail_csv_byte_identical=full,elapsed_s=time.perf_counter()-start,output_sha256={name:sha(a.out/name) for name in ['VIDEO_GABARIT_2D.mp4','OBJETS.csv','POINTS_OBJETS.csv','A_VERIFIER.csv','FRAMES.csv','RAILS.csv','AUDIT_FUSION.jsonl']})
    except BaseException as e:manifest.update(status='failed',frames=n,error=repr(e));raise
    finally:
        if output is not None:output.close()
        write_json(a.out/'MANIFESTE.json',manifest)
    print(json.dumps(dict(status=manifest['status'],frames=n,observations=objects_total,quarantined=quarantine_total,points=points_total)),flush=True)

if __name__=='__main__':main()
