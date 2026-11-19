#!/usr/bin/env python3
"""Explicit fallback: preserve verified V1, only move ambiguous trains to review.

No new network, mask, track or small-object gain. Accepted CSV/point records and
object JSON are preserved exactly. The old video image is reused except frames
requiring removal of a quarantined overlay; those are redrawn from native input.
"""
import argparse,copy,csv,json,time,os
from pathlib import Path
from contextlib import ExitStack
from fractions import Fraction
import av,cv2
from guard_only_common import RawCSV,read,sha,require,rejection_evidence
from package_gabarit_2d import verified_txt_files
from corridor_geometry import corridor_from_rows,reconstruct_mask
from rail_io import display_segments,draw_corridor,verify_common
from infer_gabarit_2d import draw_result,COMMON,OBJECT_FIELDS,write_json
from small_object_refinement import train_guard

OUTPUTS=['VIDEO_GABARIT_2D.mp4','OBJETS.csv','POINTS_OBJETS.csv','A_VERIFIER.csv','FRAMES.csv','RAILS.csv','AUDIT_FUSION.jsonl']

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['baseline','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--rejection-report',type=Path,action='append',required=True)
    p.add_argument('--source',type=Path);p.add_argument('--max-frames',type=int,default=0)
    return p

def main():
    a=parser().parse_args();require(a.max_frames>=0,'Negative prefix');require(not a.out.exists(),'Output already exists');cv2.setNumThreads(0)
    bp=a.baseline.resolve();bm=read(bp/'MANIFESTE.json');bv=read(bp/'VERIFICATION.json')
    require(bm['status']=='complete' and bm['schema_version']==1 and not bm['partial_source'] and bv['status']=='passed','Verified full V1 required')
    require(sha(bp/'MANIFESTE.json')==bv['manifest_sha256'],'Baseline manifest changed');verified_txt_files(bp,bm,bv)
    for name,digest in bm['output_sha256'].items():require(sha(bp/name)==digest,'Baseline output changed')
    learned=bm['inputs_sha256'][str(Path(bm['parameters']['litter_model']).resolve())]
    rejects,rejection_hashes=rejection_evidence(a.rejection_report,learned)
    source=(a.source or Path(bm['source'])).resolve();require(sha(source)==bm['source_sha256'],'Wrong native source')
    files=[source,bp/'MANIFESTE.json',bp/'VERIFICATION.json']+[bp/n for n in bm['output_sha256']]
    files.extend(Path(__file__).with_name(n) for n in ['finalize_guard_only.py','guard_only_common.py','small_object_refinement.py','corridor_geometry.py','rail_io.py','infer_gabarit_2d.py','package_gabarit_2d.py'])
    frozen={str(p.resolve()):sha(p) for p in files};frozen.update(rejection_hashes)
    params=dict(baseline=str(bp),source=str(source),max_frames=a.max_frames,rejection_reports=[str(p.resolve()) for p in a.rejection_report])
    m=dict(status='running',schema_version=3,method='baseline_with_train_review',parameters=params,source=str(source),source_sha256=bm['source_sha256'],video_id=bm['video_id'],model_names=bm['model_names'],checkpoint_sha256=learned,
           baseline_manifest_sha256=sha(bp/'MANIFESTE.json'),baseline_txt_path_and_content_aggregate_sha256=bv['txt_path_and_content_aggregate_sha256'],inputs_sha256=frozen,rejected_experiments=rejects,
           rails_recomputed=False,neural_inference_performed=False,new_tracking_performed=False,small_object_gain_validated=False,selective_additions=0,
           retained_contract='V1 accepted JSON objects, IDs, tracking fields and retained CSV/point records unchanged; train review preserves historical fields.',
           guard_rule='class=train AND confidence<0.5 AND corridor_coverage>=0.9 AND height_fraction>=0.75 AND bottom_fraction>=0.95',
           partial_source=bool(a.max_frames and a.max_frames<bm['frames']),runtime_scope='cached filtering and video re-encoding only; previous inference excluded',
           decision='No tested tile recipe qualified; baseline preserved without a claimed improvement for small objects.')
    a.out.mkdir(parents=True);(a.out/'TXT').mkdir();(a.out/'APERÇUS').mkdir();write_json(a.out/'MANIFESTE.json',m)
    count=accepted_total=review_total=point_total=redrawn=0;times=[];output=None;started=time.perf_counter();examples={0,147,2022,5077,5097,5307}
    try:
        with ExitStack() as st:
            def reader(name):return RawCSV(st.enter_context((bp/name).open(newline='',encoding='utf-8')))
            oldframes,oldobjects,oldpoints,oldrails=[reader(n) for n in ('FRAMES.csv','OBJETS.csv','POINTS_OBJETS.csv','RAILS.csv')]
            outobj=st.enter_context((a.out/'OBJETS.csv').open('x',newline=''));outobj.write(oldobjects.header)
            outpoints=st.enter_context((a.out/'POINTS_OBJETS.csv').open('x',newline=''));outpoints.write(oldpoints.header)
            outrails=st.enter_context((a.out/'RAILS.csv').open('x',newline=''));outrails.write(oldrails.header)
            fw=csv.DictWriter(st.enter_context((a.out/'FRAMES.csv').open('x',newline='')),fieldnames=oldframes.fields+['baseline_overlap_candidates','quarantined_candidates']);fw.writeheader()
            qw=csv.DictWriter(st.enter_context((a.out/'A_VERIFIER.csv').open('x',newline='')),fieldnames=oldobjects.fields+['review_reason']);qw.writeheader()
            audit=st.enter_context((a.out/'AUDIT_FUSION.jsonl').open('x'))
            src=st.enter_context(av.open(str(source)));oldvideo=st.enter_context(av.open(str(bp/'VIDEO_GABARIT_2D.mp4')));oldvideoiter=iter(oldvideo.decode(oldvideo.streams.video[0]));vs=src.streams.video[0]
            output=av.open(str(a.out/'VIDEO_GABARIT_2D.mp4'),'w',options={'movflags':'+faststart'});stream=output.add_stream('libx264',rate=vs.average_rate or Fraction(30));stream.width=vs.width;stream.height=vs.height;stream.pix_fmt='yuv420p';stream.time_base=Fraction(1,1000000);stream.codec_context.time_base=stream.time_base;stream.options={'crf':'23','preset':'veryfast','maxrate':'10M','bufsize':'20M'}
            for index,native in enumerate(src.decode(vs)):
                if a.max_frames and index>=a.max_frames:break
                oldvf=next(oldvideoiter,None);require(oldvf is not None and native.pts is not None and oldvf.pts is not None,'Missing source frame');t=float(native.pts*native.time_base);w,h=native.width,native.height
                require((oldvf.width,oldvf.height)==(w,h) and abs(float(oldvf.pts*oldvf.time_base)-t)<=1e-6,'Baseline rendered frame does not match native frame/time')
                fr,_=oldframes.pop();verify_common(fr,index,t,w,h,bm['video_id'],'FRAMES');rails=oldrails.frame(index)
                for _,raw in rails:outrails.write(raw)
                rr=[r for r,_ in rails];old=read(bp/f'TXT/frame_{index:06d}.txt');verify_common(old,index,t,w,h,bm['video_id'],'TXT')
                retained=[];review=[];guards=[];corridor=None
                for o in old['objects']:
                    guard=None
                    if o['class_name'].strip().casefold()=='train':
                        if corridor is None:corridor,_,_=corridor_from_rows(rr,w,h)
                        mask=reconstruct_mask(o['full_predicted_mask_contours'],w,h).astype(bool);guard=train_guard(dict(o,mask=mask),corridor);guards.append(dict(observation_id=o['observation_id'],geometry=guard))
                    if guard and guard['quarantine']:review.append(dict(o,review_reason=guard['reason'],review_geometry=guard))
                    else:retained.append(o)
                accepted_ids={o['observation_id'] for o in retained};review_ids={o['observation_id'] for o in review};require(not accepted_ids&review_ids,'Duplicate observation')
                for row,raw in oldobjects.frame(index):
                    if row['observation_id'] in accepted_ids:outobj.write(raw)
                    else:require(row['observation_id'] in review_ids,'Unmatched object CSV');qw.writerow(dict(row,review_reason='ambiguous_train_tracklike'))
                for row,raw in oldpoints.frame(index):
                    require(row['observation_id'] in accepted_ids|review_ids,'Unmatched point CSV')
                    if row['observation_id'] in accepted_ids:outpoints.write(raw);point_total+=1
                payload=dict(old,schema_version=3,method='baseline_with_train_review',objects=retained,quarantined_predictions=review)
                with (a.out/f'TXT/frame_{index:06d}.txt').open('x') as f:json.dump(payload,f,ensure_ascii=False,allow_nan=False,separators=(',',':'));f.write('\n')
                base={k:old[k] for k in COMMON};audit.write(json.dumps(dict(base,accepted_observation_ids=[o['observation_id'] for o in retained],quarantined_observation_ids=[o['observation_id'] for o in review],train_guard_reviews=guards),allow_nan=False)+'\n')
                updated=dict(fr,baseline_overlap_candidates=len(old['objects']),overlap_candidates=len(retained),confirmed_overlap_candidates=sum(o['track_confirmed'] for o in retained),quarantined_candidates=len(review));fw.writerow(updated)
                if review:
                    drawobjects=[]
                    for o in retained:
                        mask=reconstruct_mask(o['full_predicted_mask_contours'],w,h).astype(bool);inside=reconstruct_mask(o['corridor_intersection_contours'],w,h).astype(bool);drawobjects.append(dict(o,mask=mask,_intersection_mask=inside))
                    overlay=draw_result(native.to_ndarray(format='bgr24'),display_segments(rr),drawobjects,t,draw_corridor);redrawn+=1
                else:overlay=oldvf.to_ndarray(format='bgr24')
                cv2.putText(overlay,f'V2 : reference conservee | {len(review)} hypothese(s) a verifier',(1010,30),cv2.FONT_HERSHEY_SIMPLEX,.52,(225,225,255),1,cv2.LINE_AA)
                if index in examples or review:cv2.imwrite(str(a.out/'APERÇUS'/f'frame_{index:06d}.jpg'),overlay)
                vf=av.VideoFrame.from_ndarray(overlay,format='bgr24');vf.pts=round(t*1e6);vf.time_base=stream.time_base
                for packet in stream.encode(vf):output.mux(packet)
                count+=1;accepted_total+=len(retained);review_total+=len(review);times.append(t)
                if count%200==0:print(json.dumps(dict(frames=count,accepted=accepted_total,quarantined=review_total,elapsed_s=round(time.perf_counter()-started,1))),flush=True)
            full=count==bm['frames'];require(count==(min(a.max_frames,bm['frames']) if a.max_frames else bm['frames']),'Incomplete decode')
            if full:require(all(r.pending is None for r in (oldframes,oldobjects,oldpoints,oldrails)) and next(oldvideoiter,None) is None,'Unconsumed baseline records/video')
            for packet in stream.encode():output.mux(packet)
            output.close();output=None
        for path,digest in frozen.items():require(sha(path)==digest,'Frozen input changed')
        verified_txt_files(bp,bm,bv)
        if full:require(sha(a.out/'RAILS.csv')==bm['output_sha256']['RAILS.csv'],'Rail bytes changed')
        m.update(status='complete',frames=count,first_time_s=times[0],last_time_s=times[-1],overlap_observations=accepted_total,quarantined_observations=review_total,point_rows=point_total,rail_csv_byte_identical=full,redrawn_native_frames=redrawn,elapsed_s=time.perf_counter()-started,output_sha256={n:sha(a.out/n) for n in OUTPUTS})
    except BaseException as exc:m.update(status='failed',frames=count,error=repr(exc));raise
    finally:
        if output is not None:output.close()
        write_json(a.out/'MANIFESTE.json',m)
    print(json.dumps(dict(status=m['status'],frames=count,accepted=accepted_total,quarantined=review_total,point_rows=point_total)),flush=True)

if __name__=='__main__':main()
