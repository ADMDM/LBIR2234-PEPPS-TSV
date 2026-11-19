#!/usr/bin/env python3
"""Independently verify an explicit V1-preserving train-review fallback.

All accepted masks/tracks/IDs and CSV records must equal verified V1 exactly.
Only train masks are reconstructed for an independent gate calculation; every
retained point record is compared to its original bytes instead of re-derived.
"""
import argparse,hashlib,json,time
from pathlib import Path
from contextlib import ExitStack
from fractions import Fraction
from datetime import datetime,timezone
import av,cv2
import verify_refinement as v
from guard_only_common import RawCSV,rejection_evidence,read,sha,require

OUTPUTS=v.OUTPUTS

def verify_delivery(folder,report_path=None):
    folder=Path(folder).resolve(strict=True);report_path=Path(report_path) if report_path else folder/'VERIFICATION.json';started=time.perf_counter()
    report=dict(status='running',delivery=str(folder),verifier=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),scope='Exact verified-V1 preservation, native-time export checks and independent train-review filtering; no validated small-object gain.',frames_checked=0,objects_checked=0,quarantined_objects_checked=0,point_rows_checked=0)
    try:
        m=read(folder/'MANIFESTE.json');mh=sha(folder/'MANIFESTE.json')
        require(m['status']=='complete' and m['schema_version']==3 and m['method']=='baseline_with_train_review','Completed guard-only fallback required')
        require(all(m[k] is False for k in ['rails_recomputed','neural_inference_performed','new_tracking_performed','small_object_gain_validated']) and m['selective_additions']==0,'Fallback must not claim new inference/tracking/additions or small-object gain')
        require(set(m['output_sha256'])==set(OUTPUTS),'Unexpected output set')
        for name,digest in m['output_sha256'].items():require(sha(folder/name)==digest,'Changed output: '+name)
        frozen=dict(m['inputs_sha256'])
        for path,digest in frozen.items():require(sha(path)==digest,'Changed frozen input')
        bp=Path(m['parameters']['baseline']).resolve(strict=True);bm=read(bp/'MANIFESTE.json');bv=read(bp/'VERIFICATION.json')
        require(bm['status']=='complete' and bm['schema_version']==1 and bm['partial_source'] is False and bv['status']=='passed','Verified complete V1 required')
        require(sha(bp/'MANIFESTE.json')==bv['manifest_sha256']==m['baseline_manifest_sha256'],'Baseline manifest mismatch')
        for name,digest in bm['output_sha256'].items():require(sha(bp/name)==digest,'Baseline output changed')
        baseline_hash=v.baseline_txt_proof(bp,bm,bv);require(baseline_hash==m['baseline_txt_path_and_content_aggregate_sha256'],'Baseline TXT changed')
        learned=bm['inputs_sha256'][str(Path(bm['parameters']['litter_model']).resolve())]
        rejects,rejection_hashes=rejection_evidence(m['parameters']['rejection_reports'],learned);require(rejects==m['rejected_experiments'] and learned==m['checkpoint_sha256'],'Rejection evidence differs')
        frozen.update(rejection_hashes)
        source=Path(m['source']).resolve(strict=True);require(sha(source)==m['source_sha256']==bm['source_sha256'] and m['video_id']==bm['video_id'] and m['model_names']==bm['model_names'],'Wrong source/classes')
        required=[source,bp/'MANIFESTE.json',bp/'VERIFICATION.json']+[bp/n for n in bm['output_sha256']]
        for path in required:require(str(path.resolve()) in m['inputs_sha256'],'Required input not frozen')
        count=v.integer(m['frames'],minimum=1);total=v.integer(bm['frames']);limit=v.integer(m['parameters']['max_frames']);full=count==total
        require(count==(min(limit,total) if limit else total) and m['partial_source'] is (not full) and m['rail_csv_byte_identical'] is full,'Wrong frame prefix/full declaration')
        if full:require(sha(folder/'RAILS.csv')==bm['output_sha256']['RAILS.csv'],'Full rails changed')
        files=list((folder/'TXT').iterdir());require(len(files)==count and {p.name for p in files}=={f'frame_{i:06d}.txt' for i in range(count)},'TXT set mismatch')
        txt_hash=hashlib.sha256();accepted=reviewed=points_count=redrawn=rails_count=train_masks=0
        with ExitStack() as st:
            def table(path):return RawCSV(st.enter_context(path.open(newline='',encoding='utf-8')))
            oldfr,oldobj,oldpts,oldrail=[table(bp/n) for n in ('FRAMES.csv','OBJETS.csv','POINTS_OBJETS.csv','RAILS.csv')]
            fr,obj,pts,rail,q=[table(folder/n) for n in ('FRAMES.csv','OBJETS.csv','POINTS_OBJETS.csv','RAILS.csv','A_VERIFIER.csv')]
            require(obj.header==oldobj.header and pts.header==oldpts.header and rail.header==oldrail.header,'Original CSV headers changed')
            require(fr.fields==oldfr.fields+['baseline_overlap_candidates','quarantined_candidates'] and q.fields==oldobj.fields+['review_reason'],'Fallback CSV schema mismatch')
            audits=st.enter_context((folder/'AUDIT_FUSION.jsonl').open());video=st.enter_context(av.open(str(folder/'VIDEO_GABARIT_2D.mp4')))
            require(len(video.streams.video)==1 and not video.streams.audio,'One silent rendered video required');decoded=iter(video.decode(video.streams.video[0]))
            for index in range(count):
                oldrow,_=oldfr.pop();row,_=fr.pop();oldrows=oldrail.frame(index);rows=rail.frame(index);require(oldrows==rows,'Rail rows/bytes changed');rails_count+=len(rows)
                rel=f'TXT/frame_{index:06d}.txt';old=read(bp/rel);data=(folder/rel).read_bytes();new=v.json_text(data.decode());txt_hash.update(rel.encode()+b'\0'+hashlib.sha256(data).digest())
                base={k:old[k] for k in v.COMMON};v.common_check(oldrow,base,'baseline FRAMES');v.common_check(row,base,'FRAMES')
                vf=next(decoded,None);require(vf is not None and vf.pts is not None,'Missing rendered frame');w,h=base['image_width'],base['image_height'];require((vf.width,vf.height)==(w,h),'Rendered dimensions changed')
                nt=Fraction(base['source_pts']*base['time_base_num'],base['time_base_den']);target=Fraction(round(float(nt)*1e6),1000000)
                require(abs(Fraction(vf.pts)*vf.time_base-target)<=max(Fraction(1,1000000),vf.time_base/2),'Rendered PTS differs from verified native timeline')
                retain=[];review=[];guards=[];corridor=None
                for original in old['objects']:
                    flagged=False;geometry=None
                    if original['class_name'].strip().casefold()=='train':
                        if corridor is None:corridor,_,_=v.independent_corridor([r for r,_ in oldrows],w,h)
                        mask=v.decode_contours(original['full_predicted_mask_contours'],w,h);flagged,geometry=v.independent_train_review(original,mask,corridor);train_masks+=1
                        guards.append(dict(observation_id=original['observation_id'],geometry=geometry))
                    if flagged:review.append(dict(original,review_reason='ambiguous_train_tracklike',review_geometry=geometry))
                    else:retain.append(original)
                expected=dict(old,schema_version=3,method='baseline_with_train_review',objects=retain,quarantined_predictions=review)
                require(new==expected,'Original object/mask/ID/tracking changed or wrong train review')
                keepids={o['observation_id'] for o in retain};qids={o['observation_id'] for o in review}
                for record,raw in oldobj.frame(index):
                    if record['observation_id'] in keepids:require(obj.pop()==(record,raw),'Accepted object CSV changed');accepted+=1
                    else:
                        require(record['observation_id'] in qids,'Unmatched original object');qr,_=q.pop();require(qr==dict(record,review_reason='ambiguous_train_tracklike'),'Quarantine CSV altered historical fields');reviewed+=1
                for record,raw in oldpts.frame(index):
                    require(record['observation_id'] in keepids|qids,'Unmatched baseline point')
                    if record['observation_id'] in keepids:require(pts.pop()==(record,raw),'Accepted point CSV changed');points_count+=1
                expected_frame=dict(oldrow,baseline_overlap_candidates=str(len(old['objects'])),overlap_candidates=str(len(retain)),confirmed_overlap_candidates=str(sum(o['track_confirmed'] for o in retain)),quarantined_candidates=str(len(review)))
                require(row==expected_frame,'Frame metadata/count changed')
                line=next(audits,None);require(line is not None,'Audit ended early');audit=v.json_text(line)
                require(audit==dict(base,accepted_observation_ids=[o['observation_id'] for o in retain],quarantined_observation_ids=[o['observation_id'] for o in review],train_guard_reviews=guards),'Guard audit differs from independent computation')
                redrawn+=bool(review)
                report.update(frames_checked=index+1,objects_checked=accepted,quarantined_objects_checked=reviewed,point_rows_checked=points_count)
                if index==0:v.same_value(m['first_time_s'],base['time_video_s'],'First time')
                if index==count-1:v.same_value(m['last_time_s'],base['time_video_s'],'Last time')
                if (index+1)%500==0:print(json.dumps(dict(verified_frames=index+1,accepted=accepted,quarantined=reviewed,points=points_count)),flush=True)
            require(next(decoded,None) is None and next(audits,None) is None,'Extra rendered/audit frames')
            require(all(s.pending is None for s in (fr,obj,pts,rail,q)),'Extra output CSV rows')
            if full:require(all(s.pending is None for s in (oldfr,oldobj,oldpts,oldrail)),'Unconsumed original rows')
        require(accepted==m['overlap_observations'] and reviewed==m['quarantined_observations'] and points_count==m['point_rows'] and redrawn==m['redrawn_native_frames'],'Final count mismatch')
        require(sha(folder/'MANIFESTE.json')==mh,'Manifest changed while verifying')
        for name,digest in m['output_sha256'].items():require(sha(folder/name)==digest,'Output changed while verifying')
        for path,digest in frozen.items():require(sha(path)==digest,'Frozen input changed while verifying')
        require(v.baseline_txt_proof(bp,bm,bv)==baseline_hash,'Baseline TXT changed while verifying')
        report.update(status='passed',elapsed_s=time.perf_counter()-started,completed_at=datetime.now(timezone.utc).isoformat(),manifest_sha256=mh,source_sha256=m['source_sha256'],txt_files_checked=count,txt_path_and_content_aggregate_sha256=txt_hash.hexdigest(),rail_rows_checked=rails_count,original_rail_byte_identical=full,
                      rendered_pts_checked=count,source_timing_provenance='Exact FRAMES/native PTS from immutable independently verified V1; source video hash checked.',retained_objects_exact_v1_equality=True,retained_csv_and_points_byte_equality=True,historical_ids_and_tracking_preserved=True,train_masks_independently_recomputed=train_masks,quarantine_guard_independently_recomputed=True,small_object_gain_validated=False,
                      limitations=['No validated small-object improvement; rejected experiments remain rejected.','Exact retained-mask geometry inherited from the frozen verified V1 proof, not re-inferred.','Quarantine is an ambiguity signal, not a clear-track certification.','Encoded pixel appearance is visually reviewed separately; timestamps and dimensions are checked here.'])
    except BaseException as exc:
        report.update(status='failed',elapsed_s=time.perf_counter()-started,error=dict(type=type(exc).__name__,message=str(exc)));tmp=report_path.with_name(report_path.name+'.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(report_path);raise
    tmp=report_path.with_name(report_path.name+'.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(report_path);return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--delivery',type=Path,required=True);p.add_argument('--report',type=Path);a=p.parse_args();cv2.setNumThreads(0);r=verify_delivery(a.delivery,a.report);print(json.dumps({k:r[k] for k in ['status','frames_checked','objects_checked','quarantined_objects_checked','point_rows_checked']}),flush=True)

if __name__=='__main__':main()
