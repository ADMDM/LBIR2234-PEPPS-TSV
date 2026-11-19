#!/usr/bin/env python3
"""Independent schema-3 verification of preserved V1 masks and selective additions.

No inference/exporter/selection helper is imported. Geometry, contours and PTS
validators come from the independently implemented V2 verifier. The selective
rules and the validation winner are recomputed here from frozen evidence.
"""
import argparse,copy,csv,hashlib,json,math,time
from pathlib import Path
from contextlib import ExitStack
from fractions import Fraction
from datetime import datetime,timezone
import av,cv2,numpy as np
import verify_refinement as v

OUTPUTS=v.OUTPUTS
require=v.require
sha=v.sha
read=v.read_json

def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

from selection_evidence import verify_selection_evidence

def prepare_inputs(old_baseline,old_candidates,width,height,corridor):
    baseline=[];review=[];pool=[]
    for index,old in enumerate(old_baseline['objects']):
        obj=copy.deepcopy(old);obj['candidate_id']=f'b{index:04d}'
        obj['mask']=v.decode_contours(old['full_predicted_mask_contours'],width,height)
        # Exporter baseline cache retains the source observation under a separate
        # field; its in-memory selection records have no new observation ID yet.
        obj['baseline_observation_id']=obj.pop('observation_id')
        flagged,_=v.independent_train_review(obj,obj['mask'],corridor)
        (review if flagged else baseline).append(obj)
    for old in old_candidates['objects']:
        if old['inference_origin']!='tile':continue
        obj=copy.deepcopy(old);obj['candidate_id']=f't{len(pool):04d}'
        obj['candidate_observation_id']=obj.pop('observation_id');obj['mask']=v.decode_contours(old['full_predicted_mask_contours'],width,height)
        pool.append(obj)
    return baseline,review,pool

def overlap(candidate,reference):
    mask,other=candidate['mask'],reference['mask'];a,b=candidate['mask_bbox'],reference['mask_bbox']
    x0,y0,x1,y1=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
    if x1<=x0 or y1<=y0:return None
    count=int(np.count_nonzero(mask[y0:y1,x0:x1]&other[y0:y1,x0:x1]));aa=int(mask.sum());bb=int(other.sum())
    iou=count/(aa+bb-count);ios=count/min(aa,bb)
    return (count,iou,ios) if iou>.1 or ios>.2 else None

def independent_select(baseline,pool,cfg,width,height):
    kept=[];audit=[]
    ordered=sorted(enumerate(pool),key=lambda z:(-z[1]['confidence_score'],z[1]['candidate_id']))
    for index,o in ordered:
        area=int(o['mask'].sum());fraction=area/(width*height);x0,y0,x1,y1=o['mask_bbox']
        crop=v.crop_border(o['mask'],o['inference_crop_xyxy']);edge=x0==0 or y0==0 or x1==width or y1==height
        reasons=[]
        if o['confidence_score']<cfg['min_confidence']:reasons.append('below_min_confidence')
        if fraction>cfg['max_area_fraction']:reasons.append('exceeds_max_area_fraction')
        if crop:reasons.append('touches_crop_border')
        if edge:reasons.append('touches_image_border')
        conflicts=[]
        for j,ref in enumerate(baseline):
            inter=overlap(o,ref)
            if inter:
                count,iou,ios=inter
                conflicts.append(dict(baseline_index=j,baseline_id=ref['candidate_id'],baseline_candidate_id=ref['candidate_id'],baseline_observation_id=None,baseline_model_source=ref['model_source'],baseline_class_id=ref['class_id'],intersection_pixels=count,mask_iou=iou,mask_ios=ios))
        if conflicts:reasons.append('overlaps_baseline')
        selected=[]
        for ref in kept:
            inter=overlap(o,ref)
            if inter:
                count,iou,ios=inter
                selected.append(dict(selected_addition_id=ref['candidate_id'],selected_addition_candidate_id=ref['candidate_id'],selected_addition_observation_id=None,intersection_pixels=count,mask_iou=iou,mask_ios=ios))
        if selected:reasons.append('overlap_with_selected_addition')
        accepted=not reasons
        audit.append(dict(rule_version='selective_additions_v1',candidate_index=index,candidate_id=o['candidate_id'],candidate_key=o['candidate_id'],observation_id=None,view_id=o['view_id'],raw_detection_index=o['raw_detection_index'],accepted=accepted,reason='accepted_selective_addition' if accepted else reasons[0],rejection_reasons=reasons,confidence_score=o['confidence_score'],mask_area_pixels=area,image_width=width,image_height=height,mask_area_fraction=fraction,touches_crop_border=crop,touches_image_border=edge,min_confidence=cfg['min_confidence'],max_area_fraction=cfg['max_area_fraction'],max_baseline_iou=.1,max_baseline_ios=.2,baseline_ids=[c['baseline_id'] for c in conflicts],baseline_conflicts=conflicts,selected_addition_conflicts=selected))
        if accepted:kept.append(o)
    return kept,audit

def verified_input(folder,version):
    folder=Path(folder).resolve(strict=True);m=read(folder/'MANIFESTE.json');p=read(folder/'VERIFICATION.json')
    require(m['status']=='complete' and m['schema_version']==version and m['partial_source'] is False and p['status']=='passed','Complete verified source run required')
    require(sha(folder/'MANIFESTE.json')==p['manifest_sha256'],'Input manifest changed')
    for name,hash_ in m['output_sha256'].items():require(sha(folder/name)==hash_,'Input export changed: '+name)
    aggregate=v.baseline_txt_proof(folder,m,p)
    return m,p,aggregate

def check_record(obj,row,base,corridor,support,manifest,ordinal,generation,tracks,bm,baseline_map,pool_map,quarantined):
    kind='q' if quarantined else 'o'
    expected=f"{base['video_id']}:v2s:f{base['frame_index']:06d}:{kind}{ordinal:03d}"
    require(obj['observation_id']==row['observation_id']==expected,'Wrong selective observation ID')
    normalized=copy.deepcopy(obj);normalized_row=dict(row)
    normalized['observation_id']=normalized_row['observation_id']=f"{base['video_id']}:v2:f{base['frame_index']:06d}:{kind}{ordinal:03d}"
    if obj['inference_origin']=='tile':
        ident=obj['candidate_id'];require(ident in pool_map,'Unknown selected tile candidate')
        original=pool_map[ident]
        require(obj['candidate_observation_id']==original['candidate_observation_id'] and obj['candidate_mask_sha256']==original['full_mask_sha256'],'Tile reference differs from verified candidate input')
        for key in ('model_source','class_id','class_name','confidence_score','raw_detection_index','full_mask_sha256','full_predicted_mask_contours','mask_bbox','model_bbox_original_px','inference_crop_xyxy','touches_crop_border','view_id','view_kind','tile_index'):
            require(obj[key]==original[key],'Selected tile geometry/provenance changed: '+key)
        normalized['candidate_id']=f"t{v.integer(obj['raw_detection_index']):04d}"
        normalized['tile_detection_index']=original['tile_detection_index']
    tile_views={x['view_id']:x for x in v.expected_tile_views(corridor,640,.25)}
    v.check_object(normalized,normalized_row,base,corridor,support,manifest,ordinal,generation,tracks,bm,baseline_map,tile_views,quarantined)

def verify_delivery(folder,report_path=None):
    folder=Path(folder).resolve(strict=True);report_path=Path(report_path) if report_path else folder/'VERIFICATION.json';started=time.perf_counter()
    report=dict(status='running',delivery=str(folder),verifier=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),started_at=datetime.now(timezone.utc).isoformat(),
                scope='Independent schema-3 export and selective-rule verification; no detector accuracy or physical danger claim.',frames_checked=0,objects_checked=0,quarantined_objects_checked=0,point_rows_checked=0)
    try:
        mp=folder/'MANIFESTE.json';m=read(mp);mh=sha(mp)
        require(m['status']=='complete' and m['schema_version']==3 and m['method']=='selective_additions_preserving_global_masks','Completed selective run required')
        require(m['neural_inference_performed'] is False and m['rails_recomputed'] is False,'Unexpected new inference or changed rails')
        require(set(m['output_sha256'])==set(OUTPUTS),'Unexpected output set')
        for name,hash_ in m['output_sha256'].items():require(sha(folder/name)==hash_,'Changed output: '+name)
        frozen=dict(m['inputs_sha256'])
        for path,hash_ in frozen.items():require(sha(path)==hash_,'Changed frozen input: '+path)
        bp=Path(m['parameters']['baseline']).resolve(strict=True);cp=Path(m['parameters']['candidates']).resolve(strict=True)
        bm,bv,bhash=verified_input(bp,1);cm,cv,chash=verified_input(cp,2)
        require(sha(bp/'MANIFESTE.json')==m['baseline_manifest_sha256']==cm['baseline_manifest_sha256'],'Wrong baseline lineage')
        require(sha(cp/'MANIFESTE.json')==m['candidate_manifest_sha256'],'Wrong candidate lineage')
        require(bhash==m['baseline_txt_path_and_content_aggregate_sha256'] and chash==m['candidate_txt_path_and_content_aggregate_sha256'],'Input TXT aggregate differs')
        require(bm['frames']==cm['frames'] and bm['video_id']==cm['video_id']==m['video_id'] and bm['source_sha256']==cm['source_sha256']==m['source_sha256'],'Input source/frame mismatch')
        require(bm['model_names']==cm['model_names']==m['model_names'],'Model dictionaries changed')
        for key,expected in [('tile_size',640),('tile_overlap',.25),('conf',.15),('imgsz',960)]:
            require(cm['parameters'][key]==m['parameters'][key]==expected,'Unexpected candidate settings: '+key)
        selection,selection_frozen=verify_selection_evidence(m['parameters']['selection_report'],m);cfg=selection['selected_config'];frozen.update(selection_frozen)
        model=Path(m['parameters']['litter_model']).resolve(strict=True);source=Path(m['source']).resolve(strict=True)
        require(sha(model)==selection['checkpoint_sha256']==cm['inputs_sha256'][str(Path(cm['parameters']['litter_model']).resolve())],'Checkpoint mismatch')
        require(sha(source)==m['source_sha256'],'Wrong native source')
        required=[model,source,Path(m['parameters']['selection_report']).resolve()]
        for path,source_manifest in [(bp,bm),(cp,cm)]:required.extend(path/n for n in ['MANIFESTE.json','VERIFICATION.json']+list(source_manifest['output_sha256']))
        for path in required:require(str(path.resolve()) in m['inputs_sha256'],'Required input not frozen: '+str(path))
        total=v.integer(bm['frames']);count=v.integer(m['frames'],minimum=1);limit=v.integer(m['parameters']['max_frames']);full=count==total
        require(count==(min(limit,total) if limit else total) and m['partial_source'] is (not full) and m['rail_csv_byte_identical'] is full,'Wrong full/prefix contract')
        if full:require(sha(folder/'RAILS.csv')==bm['output_sha256']['RAILS.csv']==cm['output_sha256']['RAILS.csv'],'Full rails changed')
        present=list((folder/'TXT').iterdir());require(len(present)==count and {x.name for x in present}=={f'frame_{i:06d}.txt' for i in range(count)},'Wrong TXT set')
        txt_hash=hashlib.sha256();prior_t=None;generation=0;tracks={};rail_count=0;additions_total=0;preserved_total=0
        with ExitStack() as stack:
            def table(path,label):return v.CSVStream(stack.enter_context(path.open(newline='',encoding='utf-8')),label)
            frames=table(folder/'FRAMES.csv','FRAMES');oldframes=table(bp/'FRAMES.csv','baseline FRAMES')
            objects=table(folder/'OBJETS.csv','OBJETS');reviews=table(folder/'A_VERIFIER.csv','A_VERIFIER');points=table(folder/'POINTS_OBJETS.csv','POINTS')
            rails=table(folder/'RAILS.csv','RAILS');oldrails=table(bp/'RAILS.csv','baseline RAILS')
            require(rails.fields==oldrails.fields and reviews.fields==objects.fields+['review_reason'],'Changed CSV columns')
            audits=stack.enter_context((folder/'AUDIT_FUSION.jsonl').open())
            native=stack.enter_context(av.open(str(source)));render=stack.enter_context(av.open(str(folder/'VIDEO_GABARIT_2D.mp4')))
            require(len(render.streams.video)==1 and len(render.streams.audio)==0,'One silent video stream required')
            ns=iter(native.decode(native.streams.video[0]));rs=iter(render.decode(render.streams.video[0]))
            for index in range(count):
                nf,rf=next(ns,None),next(rs,None);require(nf is not None and rf is not None and nf.pts is not None and rf.pts is not None,'Missing decoded frame/PTS')
                nt=Fraction(nf.pts)*nf.time_base;t=float(nt);require(prior_t is None or nt>prior_t,'Nonincreasing native PTS')
                if prior_t is not None and nt-prior_t>Fraction(1,2):generation+=1
                prior_t=nt
                require(abs(Fraction(rf.pts)*rf.time_base-Fraction(round(t*1e6),1000000))<=max(Fraction(1,1000000),rf.time_base/2),'Rendered time changed')
                w,h=nf.width,nf.height;require((w,h)==(rf.width,rf.height),'Rendered dimensions changed')
                base=dict(video_id=m['video_id'],frame_index=index,time_video_s=t,source_pts=nf.pts,time_base_num=nf.time_base.numerator,time_base_den=nf.time_base.denominator,image_width=w,image_height=h)
                fr,oldfr=frames.pop(),oldframes.pop();v.common_check(fr,base,'FRAMES');v.common_check(oldfr,base,'baseline FRAMES')
                rr,orr=rails.frame(index),oldrails.frame(index);require(rr==orr,'Rail rows changed');rail_count+=len(rr)
                for row in rr:v.common_check(row,base,'RAILS',True)
                corridor,support,info=v.independent_corridor(rr,w,h)
                rel=f'TXT/frame_{index:06d}.txt';data=(folder/rel).read_bytes();txt_hash.update(rel.encode()+b'\0'+hashlib.sha256(data).digest());d=v.json_text(data.decode())
                old=read(bp/rel);candidate=read(cp/rel)
                for label,payload in [('TXT',d),('baseline TXT',old),('candidate TXT',candidate)]:v.common_check(payload,base,label)
                require(d['schema_version']==3 and d['processing_status']==fr['processing_status']=='ok','Wrong schema/status')
                require(d['coordinate_convention']=='original pixel centres, x right, y down' and d['rail_support']==old['rail_support']==info,'Wrong coordinate/support metadata')
                require(d['rail_diagnostic_status']==fr['rail_diagnostic_status']==oldfr['rail_diagnostic_status'],'Rail diagnostic changed')
                require(fr['txt_file']==rel and v.integer(fr['rail_supported_rows'])==info['supported_rows'] and v.integer(fr['rail_corridor_pixel_count'])==info['corridor_pixel_count'],'Wrong frame metadata')
                baseline,review,pool=prepare_inputs(old,candidate,w,h,corridor);added,decisions=independent_select(baseline,pool,cfg,w,h)
                expected=baseline+added;require([o['candidate_id'] for o in d['objects']]==[o['candidate_id'] for o in expected],'Baseline lost/reordered or wrong selective additions')
                require([o['candidate_id'] for o in d['quarantined_predictions']]==[o['candidate_id'] for o in review],'Wrong baseline quarantine set')
                raw_audit=next(audits,None);require(raw_audit is not None,'Audit ended early');audit=v.json_text(raw_audit);v.common_check(audit,base,'AUDIT')
                require(audit['selection_config']==cfg and audit['baseline_kept_ids']==[o['candidate_id'] for o in baseline] and audit['baseline_quarantined_ids']==[o['candidate_id'] for o in review] and audit['added_candidate_ids']==[o['candidate_id'] for o in added],'Audit decisions differ')
                expected_pool=[dict(candidate_id=o['candidate_id'],candidate_observation_id=o['candidate_observation_id'],full_mask_sha256=o['full_mask_sha256']) for o in pool]
                require(audit['tile_candidate_pool']==expected_pool and audit['selective_decisions']==decisions,'Selective rule audit differs from independent recomputation')
                baseline_map={o['observation_id']:o for o in old['objects']};pool_map={o['candidate_id']:o for o in pool}
                for quarantined,group,rows in [(False,d['objects'],objects.frame(index)),(True,d['quarantined_predictions'],reviews.frame(index))]:
                    require(len(group)==len(rows),'TXT/CSV object counts differ')
                    for ordinal,(obj,row) in enumerate(zip(group,rows)):
                        v.common_check(row,base,'OBJETS');require(row['txt_file']==rel,'Wrong TXT reference')
                        check_record(obj,row,base,corridor,support,m,ordinal,generation,tracks,bm,baseline_map,pool_map,quarantined)
                        if not quarantined:v.check_point_rows(points,obj,base)
                if points.pending is not None:require(v.integer(points.pending['frame_index'])>index,'Extra/misassigned point rows')
                expected_counts=dict(baseline_candidates=len(old['objects']),tile_candidate_pool=len(pool),selective_additions=len(added),selective_rejections=len(pool)-len(added),overlap_candidates=len(expected),quarantined_candidates=len(review),confirmed_overlap_candidates=sum(o['track_confirmed'] for o in d['objects']))
                for key,value in expected_counts.items():require(v.integer(fr[key])==value,'Frame count mismatch: '+key)
                additions_total+=len(added);preserved_total+=len(baseline)
                report.update(frames_checked=index+1,objects_checked=objects.count,quarantined_objects_checked=reviews.count,point_rows_checked=points.count)
                if index==0:v.same_value(m['first_time_s'],t,'First time')
                if index==count-1:v.same_value(m['last_time_s'],t,'Last time')
                if (index+1)%200==0:print(json.dumps(dict(verified_frames=index+1,objects=objects.count,additions=additions_total,quarantined=reviews.count)),flush=True)
            require(next(rs,None) is None and next(audits,None) is None,'Extra video/audit frames')
            if full:require(next(ns,None) is None and oldframes.pending is None and oldrails.pending is None,'Unconsumed full baseline')
            for stream in (frames,objects,reviews,points,rails):require(stream.pending is None,'Extra output rows: '+stream.label)
            require(objects.count==m['overlap_observations'] and reviews.count==m['quarantined_observations'] and points.count==m['point_rows'] and additions_total==m['selective_additions'],'Manifest count mismatch')
        require(sha(mp)==mh,'Manifest changed during verification')
        for name,hash_ in m['output_sha256'].items():require(sha(folder/name)==hash_,'Output changed during verification')
        for path,hash_ in frozen.items():require(sha(path)==hash_,'Input changed during verification: '+path)
        require(v.baseline_txt_proof(bp,bm,bv)==bhash and v.baseline_txt_proof(cp,cm,cv)==chash,'Input TXT changed during verification')
        report.update(status='passed',elapsed_s=time.perf_counter()-started,completed_at=datetime.now(timezone.utc).isoformat(),manifest_sha256=mh,source_sha256=m['source_sha256'],txt_files_checked=count,txt_path_and_content_aggregate_sha256=txt_hash.hexdigest(),rail_rows_checked=rail_count,original_rail_byte_identical=full,source_and_rendered_pts_checked=count,
                      preserved_baseline_observations=preserved_total,selective_additions_checked=additions_total,all_masks_and_intersections_reconstructed=True,selective_rules_independently_recomputed=True,baseline_preservation_verified=True,quarantine_guard_independently_recomputed=True,selection_winner_recomputed=True,
                      limitations=['No ground-truth accuracy or physical danger assessment.','Candidate pool is the frozen V2 retained tile export; suppressed first-stage masks are not rerun.','Validation winner selection is exploratory on the development VAL set.'])
    except BaseException as exc:
        report.update(status='failed',elapsed_s=time.perf_counter()-started,error=dict(type=type(exc).__name__,message=str(exc)))
        temp=report_path.with_name(report_path.name+'.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(report_path);raise
    temp=report_path.with_name(report_path.name+'.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(report_path)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--delivery',type=Path,required=True);p.add_argument('--report',type=Path);a=p.parse_args();cv2.setNumThreads(0)
    result=verify_delivery(a.delivery,a.report);print(json.dumps({k:result[k] for k in ['status','frames_checked','objects_checked','quarantined_objects_checked','point_rows_checked','selective_additions_checked']}),flush=True)

if __name__=='__main__':main()
