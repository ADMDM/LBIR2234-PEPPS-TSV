#!/usr/bin/env python3
"""Render a validated selective addition policy from two verified mask exports.

No network inference or training is performed. All non-quarantined V1 masks
remain intact; only admissible small tile masks are added from the first trial.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,tempfile,time
from pathlib import Path
from contextlib import ExitStack
from fractions import Fraction
import av,cv2,numpy as np
from corridor_geometry import corridor_from_rows,reconstruct_mask,analyze_object_mask
from rail_io import FrameGroups,verify_common,display_segments,draw_corridor
from infer_gabarit_2d import COMMON,OBJECT_FIELDS,POINT_FIELDS,MaskTracks,sha,mask_sha,write_json,write_points,draw_result
from refine_video import baseline_objects,summarize,decorate,serial_record,REVIEW_FIELDS
from small_object_refinement import train_guard
from selective_refinement import select_additions
from package_gabarit_2d import verified_txt_files

FRAME_FIELDS=COMMON+['processing_status','rail_diagnostic_status','rail_supported_rows',
 'rail_corridor_pixel_count','baseline_candidates','tile_candidate_pool','selective_additions',
 'selective_rejections','overlap_candidates','confirmed_overlap_candidates',
 'quarantined_candidates','txt_file']
OUTPUTS=['VIDEO_GABARIT_2D.mp4','OBJETS.csv','POINTS_OBJETS.csv','A_VERIFIER.csv','FRAMES.csv','RAILS.csv','AUDIT_FUSION.jsonl']
SELECTIVE_PROTOCOL='exploratory_selective_tiles_v1'
CONSENSUS_PROTOCOL='exploratory_multiview_confirmation_v1'
CONFIRMATION_RULE='at least two distinct tile views including the kept candidate; a direct suppression audit record with rule cross_view_mask_iou and mask_iou>.5 must link the other tile to that kept candidate ID/view; globals and IoS-only records do not count; no score or mask changes'
CONFIRMATION_FRAME_FIELDS=['confirmation_survivors','confirmation_rejections']

def confirmation_mode(selection):
    rule=selection['frozen_rule'];name=rule['name']
    if name==CONSENSUS_PROTOCOL:
        if rule.get('confirmation_rule')!=CONFIRMATION_RULE:
            raise ValueError('Unexpected multiview confirmation rule')
        return True
    if name!=SELECTIVE_PROTOCOL or 'confirmation_rule' in rule:
        raise ValueError('Unknown or contradictory selection protocol')
    return False

def confirmed_pool_for_frame(pool,source_audit,index,t,w,h,video_id):
    if not isinstance(source_audit,dict) or not isinstance(source_audit.get('suppressed'),list):
        raise ValueError('Missing candidate suppression audit')
    verify_common(source_audit,index,t,w,h,video_id,'candidate AUDIT_FUSION')
    from multiview_confirmation import filter_confirmed_tiles
    return filter_confirmed_tiles(pool,source_audit['suppressed'])

def verified_run(path,version):
    path=Path(path).resolve(strict=True)
    m=json.loads((path/'MANIFESTE.json').read_text());v=json.loads((path/'VERIFICATION.json').read_text())
    if m['status']!='complete' or m['schema_version']!=version or m['partial_source'] or v['status']!='passed':
        raise ValueError('Complete verified input required: '+str(path))
    if sha(path/'MANIFESTE.json')!=v['manifest_sha256']:raise ValueError('Changed input manifest')
    verified_txt_files(path,m,v)
    for name,digest in m['output_sha256'].items():
        if sha(path/name)!=digest:raise ValueError('Changed input output: '+name)
    return m,v

def tile_candidates(payload,w,h):
    result=[]
    for old in payload['objects']:
        if old['inference_origin']!='tile':continue
        mask=reconstruct_mask(old['full_predicted_mask_contours'],w,h).astype(bool)
        if mask_sha(mask)!=old['full_mask_sha256']:raise ValueError('Changed candidate mask')
        keys=['model_source','class_id','class_name','confidence_score','raw_detection_index',
              'mask_bbox','model_bbox_original_px','inference_crop_xyxy','touches_crop_border',
              'view_id','view_kind','tile_index']
        obj={k:old[k] for k in keys}
        obj.update(mask=mask,candidate_id=old['candidate_id'],inference_origin='tile',
            baseline_observation_id=None,candidate_observation_id=old['observation_id'],
            candidate_mask_sha256=old['full_mask_sha256'])
        result.append(obj)
    return result

def selected_config(path):
    path=Path(path).resolve(strict=True)
    report=json.loads(path.read_text())
    manifest=json.loads((path.parent/'EVALUATION_MANIFEST.json').read_text())
    if (report['status']!='complete' or report['split']!='val' or report['images']!=288
            or report['selected_eligible'] is not True):
        raise ValueError('Complete eligible selective validation required')
    if (manifest['status']!='complete' or manifest['split']!='val'
            or manifest['image_count']!=288 or manifest['inputs_unchanged'] is not True
            or manifest['test_images_or_annotations_opened'] is not False
            or manifest['model_inference_performed'] is not False):
        raise ValueError('Complete unchanged VAL-only selection evidence required')
    proof=manifest['selection']
    if (Path(proof['path']).name!=path.name or proof['sha256']!=sha(path)
            or proof['bytes']!=path.stat().st_size):
        raise ValueError('Selection report differs from its evaluation manifest')
    configuration=manifest['configuration']
    digest=hashlib.sha256(json.dumps(configuration,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if (digest!=manifest['configuration_sha256'] or digest!=report['configuration_sha256']
            or configuration['checkpoint_sha256']!=report['checkpoint_sha256']
            or configuration['dataset_sha256']!=report['dataset_sha256']):
        raise ValueError('Selection configuration evidence differs')
    protocol=configuration['protocol']
    use_confirmation=confirmation_mode(report)
    expected={'split':'val','images':288,
        'reference_objects':871,'tile_size':640,'nominal_overlap':.25,'network_imgsz':960,
        'operating_baseline_confidence':.15,'ap_baseline_confidence':.001,
        'confidence_grid':[.35,.5,.65,.8],'area_fraction_grid':[.001,.002,.004],
        'test_used':False,
        'first_stage':'merge_multiscale(raw globals>=.15, raw tiles>=.15), keep tile survivors, no first-stage max100 truncation',
        'baseline_policy':'preserve every baseline mask; no replacement, union or suppression, including all low-score AP globals>=.001'}
    if (protocol!=report['frozen_rule'] or any(protocol.get(k)!=v for k,v in expected.items())
            or protocol.get('test_used') is not False or report['reference_objects']!=871):
        raise ValueError('Selection used an incompatible first-stage protocol')
    helpers=['selective_refinement.py','small_object_refinement.py']
    if use_confirmation:helpers.append('multiview_confirmation.py')
    for name in helpers:
        proofs=[p for p in configuration['programs'] if Path(p['path']).name==name]
        local=Path(__file__).with_name(name)
        if (len(proofs)!=1 or proofs[0]['sha256']!=sha(local)
                or proofs[0]['bytes']!=local.stat().st_size):
            raise ValueError('Selection helper differs from renderer: '+name)
    cfg=report['selected_config']
    if (set(cfg)!={'min_confidence','max_area_fraction'}
            or cfg['min_confidence'] not in (.35,.5,.65,.8)
            or cfg['max_area_fraction'] not in (.001,.002,.004)):
        raise ValueError('Configuration outside the declared search grid')
    index=report['selected_grid_index']
    expected_grid=[(c,f) for c in expected['confidence_grid'] for f in expected['area_fraction_grid']]
    if (type(index) is not int or not 0<=index<len(expected_grid)
            or expected_grid[index]!=(cfg['min_confidence'],cfg['max_area_fraction'])):
        raise ValueError('Selected configuration/grid index differ')
    winners=[c for c in report['candidates'] if c['grid_index']==index]
    if (len(winners)!=1 or winners[0]['eligible'] is not True
            or any(winners[0].get(k)!=v for k,v in cfg.items())
            or not winners[0]['eligibility_checks']
            or any(v is not True for v in winners[0]['eligibility_checks'].values())
            or report['selected']!=winners[0]['metrics']
            or report['selected_operating']!=winners[0]['operating']):
        raise ValueError('Selected configuration is not the documented eligible result')
    return cfg,report

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','candidates','selection-report','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source',type=Path);p.add_argument('--max-frames',type=int,default=0)
    return p

def main():
    a=parser().parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    if a.max_frames<0:raise ValueError('Negative frame limit')
    os.environ.setdefault('YOLO_AUTOINSTALL','false')
    os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(tempfile.gettempdir())/'gabarit2d_selective'))
    cv2.setNumThreads(0)
    bp=a.baseline.resolve();cp=a.candidates.resolve()
    bm,bv=verified_run(bp,1);cm,cv=verified_run(cp,2)
    cfg,selection=selected_config(a.selection_report)
    use_confirmation=confirmation_mode(selection)
    if cm['baseline_manifest_sha256']!=sha(bp/'MANIFESTE.json') or cm['source_sha256']!=bm['source_sha256']:
        raise ValueError('Candidate run does not refine this baseline')
    if cm['frames']!=bm['frames'] or cm['video_id']!=bm['video_id']:raise ValueError('Input video/frame mismatch')
    if cm['parameters']['tile_size']!=640 or cm['parameters']['tile_overlap']!=.25 or cm['parameters']['conf']!=.15:
        raise ValueError('Unexpected first-stage inference protocol')
    model=Path(cm['parameters']['litter_model']).resolve()
    if sha(model)!=selection['checkpoint_sha256']:raise ValueError('Selection checkpoint differs from video')
    source=(a.source or Path(bm['source'])).resolve()
    if sha(source)!=bm['source_sha256']:raise ValueError('Wrong source')
    paths=[source,model,a.selection_report.resolve(),
           a.selection_report.resolve().parent/'EVALUATION_MANIFEST.json']
    for folder,manifest in ((bp,bm),(cp,cm)):
        paths.extend(folder/n for n in ['MANIFESTE.json','VERIFICATION.json']+list(manifest['output_sha256']))
    programs=['finalize_selective_video.py','selective_refinement.py','small_object_refinement.py',
              'refine_video.py','infer_gabarit_2d.py','corridor_geometry.py','rail_io.py','package_gabarit_2d.py']
    if use_confirmation:programs.append('multiview_confirmation.py')
    paths.extend(Path(__file__).with_name(n) for n in programs)
    frozen={str(path.resolve()):sha(path) for path in paths}
    a.out.mkdir(parents=True);(a.out/'TXT').mkdir();(a.out/'APERÇUS').mkdir()
    params={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items()}
    params.update(cfg,litter_model=str(model),conf=.15,tile_size=640,tile_overlap=.25,imgsz=960)
    params.update(selection_protocol=selection['frozen_rule']['name'],multiview_confirmation_required=use_confirmation)
    method='multiview_confirmed_selective_additions_preserving_global_masks' if use_confirmation else 'selective_additions_preserving_global_masks'
    m=dict(status='running',schema_version=3,method=method,
        source=str(source),source_sha256=bm['source_sha256'],video_id=bm['video_id'],model_names=bm['model_names'],
        device='cpu_cached_masks',inputs_sha256=frozen,parameters=params,rails_recomputed=False,
        neural_inference_performed=False,baseline_manifest_sha256=sha(bp/'MANIFESTE.json'),
        candidate_manifest_sha256=sha(cp/'MANIFESTE.json'),
        baseline_txt_path_and_content_aggregate_sha256=bv['txt_path_and_content_aggregate_sha256'],
        candidate_txt_path_and_content_aggregate_sha256=cv['txt_path_and_content_aggregate_sha256'],
        selection_report_sha256=sha(a.selection_report),selection_config=cfg,
        selection_protocol=selection['frozen_rule']['name'],multiview_confirmation_required=use_confirmation,
        confirmation_rule=CONFIRMATION_RULE if use_confirmation else None,
        partial_source=bool(a.max_frames and a.max_frames<bm['frames']),clock='native PTS; no resampling',
        scope='2D candidate masks; not a physical obstacle or clear-track certification',
        guard_rule=cm['guard_rule'],quarantine_storage=cm['quarantine_storage'],
        baseline_retention='all V1 observed masks unchanged, except ambiguous train hypotheses moved to review',
        runtime_scope='cached-mask reprocessing and encoding only; earlier model inference excluded')
    write_json(a.out/'MANIFESTE.json',m)
    n=accepted_total=review_total=points_total=addition_total=0;times=[];output=None;start=time.perf_counter()
    pool_total=confirmation_total=confirmation_rejected_total=0
    trackers=MaskTracks(.15)
    examples={0,147,1242,1559,1765,1820,1890,2022,2948,4105,4422,5077,5097,5160,5307}
    try:
        with ExitStack() as stack:
            frames=csv.DictReader(stack.enter_context((bp/'FRAMES.csv').open(newline='')))
            axes=FrameGroups(stack.enter_context((bp/'RAILS.csv').open(newline='')),'RAILS.csv')
            candidate_audits=stack.enter_context((cp/'AUDIT_FUSION.jsonl').open()) if use_confirmation else None
            with (bp/'RAILS.csv').open() as f:axis_fields=next(csv.reader(f))
            writers={}
            frame_fields=FRAME_FIELDS+CONFIRMATION_FRAME_FIELDS if use_confirmation else FRAME_FIELDS
            for name,fields in [('OBJETS.csv',OBJECT_FIELDS),('POINTS_OBJETS.csv',POINT_FIELDS),('A_VERIFIER.csv',REVIEW_FIELDS),('FRAMES.csv',frame_fields),('RAILS.csv',axis_fields)]:
                writer=csv.DictWriter(stack.enter_context((a.out/name).open('x',newline='')),fieldnames=fields);writer.writeheader();writers[name]=writer
            audit=stack.enter_context((a.out/'AUDIT_FUSION.jsonl').open('x'))
            src=stack.enter_context(av.open(str(source)));vs=src.streams.video[0]
            output=av.open(str(a.out/'VIDEO_GABARIT_2D.mp4'),'w',options={'movflags':'+faststart'})
            stream=output.add_stream('libx264',rate=vs.average_rate or Fraction(30));stream.width=vs.width;stream.height=vs.height
            stream.pix_fmt='yuv420p';stream.time_base=Fraction(1,1000000);stream.codec_context.time_base=stream.time_base
            stream.options={'crf':'23','preset':'veryfast','maxrate':'10M','bufsize':'20M'}
            for index,frame in enumerate(src.decode(vs)):
                if a.max_frames and index>=a.max_frames:break
                if frame.pts is None:raise ValueError('Missing PTS')
                t=float(frame.pts*frame.time_base);w,h=frame.width,frame.height
                if times and t<=times[-1]:raise ValueError('Non-increasing PTS')
                row=next(frames,None);verify_common(row,index,t,w,h,bm['video_id'],'baseline FRAMES')
                rails=axes.take(index)
                for rr in rails:verify_common(rr,index,t,w,h,bm['video_id'],'RAILS');writers['RAILS.csv'].writerow(rr)
                corridor,support,railinfo=corridor_from_rows(rails,w,h)
                old=json.loads((bp/'TXT'/f'frame_{index:06d}.txt').read_text())
                candidates=json.loads((cp/'TXT'/f'frame_{index:06d}.txt').read_text())
                for label,record in [('baseline',old),('candidates',candidates)]:verify_common(record,index,t,w,h,bm['video_id'],label)
                original=baseline_objects(old,w,h);pool=tile_candidates(candidates,w,h)
                eligible_pool=pool;confirmation_decisions=[]
                if use_confirmation:
                    audit_line=next(candidate_audits,None)
                    if audit_line is None:raise ValueError('Incomplete candidate suppression audit')
                    eligible_pool,confirmation_decisions=confirmed_pool_for_frame(pool,json.loads(audit_line),index,t,w,h,bm['video_id'])
                baseline=[];review=[]
                for obj in original:
                    guard=train_guard(obj,corridor)
                    if guard['quarantine']:
                        obj.update(track_id=None,track_confirmed=False,observed_hits=0,review_reason=guard['reason'],review_geometry=guard)
                        review.append(obj)
                    else:baseline.append(obj)
                additions,decisions=select_additions(baseline,eligible_pool,cfg['min_confidence'],cfg['max_area_fraction'],image_width=w,image_height=h)
                accepted=baseline+additions
                base=dict(video_id=bm['video_id'],frame_index=index,time_video_s=t,source_pts=int(frame.pts),
                    time_base_num=frame.time_base.numerator,time_base_den=frame.time_base.denominator,image_width=w,image_height=h)
                for kind,objects in [('o',accepted),('q',review)]:
                    for i,obj in enumerate(objects):
                        ar=analyze_object_mask(obj['mask'],corridor,support)
                        if ar['relation_2d']!='overlap_observed':raise ValueError('Cached retained mask no longer overlaps observed corridor')
                        decorate(obj,corridor,ar,f"{bm['video_id']}:v2s:f{index:06d}:{kind}{i:03d}")
                bgr=frame.to_ndarray(format='bgr24');trackers.update(accepted,bgr,index,t)
                txt_rel=f'TXT/frame_{index:06d}.txt'
                payload=dict(base,schema_version=3,processing_status='ok',coordinate_convention='original pixel centres, x right, y down',
                    rail_diagnostic_status=row['rail_diagnostic_status'],rail_support=railinfo,
                    objects=[serial_record(o) for o in accepted],quarantined_predictions=[serial_record(o) for o in review])
                for obj in accepted:
                    writers['OBJETS.csv'].writerow(summarize(base,obj,txt_rel))
                    for name,key in [('full_predicted_mask','full_predicted_mask_contours'),('corridor_intersection','corridor_intersection_contours')]:
                        points_total+=write_points(writers['POINTS_OBJETS.csv'],base,obj,name,obj[key])
                for obj in review:writers['A_VERIFIER.csv'].writerow(dict(summarize(base,obj,txt_rel),review_reason=obj['review_reason']))
                with (a.out/txt_rel).open('x') as f:json.dump(payload,f,ensure_ascii=False,allow_nan=False,separators=(',',':'));f.write('\n')
                audit_row=dict(base,selection_config=cfg,baseline_kept_ids=[o['candidate_id'] for o in baseline],
                    baseline_quarantined_ids=[o['candidate_id'] for o in review],
                    tile_candidate_pool=[dict(candidate_id=o['candidate_id'],candidate_observation_id=o['candidate_observation_id'],full_mask_sha256=o['candidate_mask_sha256']) for o in pool],
                    added_candidate_ids=[o['candidate_id'] for o in additions],selective_decisions=decisions)
                if use_confirmation:audit_row.update(confirmation_required=True,confirmation_rule=CONFIRMATION_RULE,
                    confirmed_tile_candidate_ids=[o['candidate_id'] for o in eligible_pool],confirmation_decisions=confirmation_decisions)
                audit.write(json.dumps(audit_row,ensure_ascii=False,allow_nan=False)+'\n')
                frame_row=dict(base,processing_status='ok',rail_diagnostic_status=row['rail_diagnostic_status'],
                    rail_supported_rows=railinfo['supported_rows'],rail_corridor_pixel_count=railinfo['corridor_pixel_count'],
                    baseline_candidates=len(original),tile_candidate_pool=len(pool),selective_additions=len(additions),
                    selective_rejections=len(eligible_pool)-len(additions),overlap_candidates=len(accepted),
                    confirmed_overlap_candidates=sum(o['track_confirmed'] for o in accepted),quarantined_candidates=len(review),txt_file=txt_rel)
                if use_confirmation:frame_row.update(confirmation_survivors=len(eligible_pool),confirmation_rejections=len(pool)-len(eligible_pool))
                writers['FRAMES.csv'].writerow(frame_row)
                overlay=draw_result(bgr,display_segments(rails),accepted,t,draw_corridor)
                mode_label='V2 multi-vues' if use_confirmation else 'V2 selective'
                cv2.putText(overlay,f'{mode_label} | {len(additions)} ajout(s) | {len(review)} a verifier',(1050,30),cv2.FONT_HERSHEY_SIMPLEX,.55,(225,225,255),1,cv2.LINE_AA)
                if index in examples:cv2.imwrite(str(a.out/'APERÇUS'/f'frame_{index:06d}.jpg'),overlay)
                vf=av.VideoFrame.from_ndarray(overlay,format='bgr24');vf.pts=round(t*1000000);vf.time_base=stream.time_base
                for packet in stream.encode(vf):output.mux(packet)
                times.append(t);n+=1;accepted_total+=len(accepted);review_total+=len(review);addition_total+=len(additions)
                pool_total+=len(pool);confirmation_total+=len(eligible_pool);confirmation_rejected_total+=len(pool)-len(eligible_pool)
                if n%200==0:print(json.dumps(dict(frames=n,time_video_s=t,observations=accepted_total,additions=addition_total,quarantined=review_total,elapsed_s=round(time.perf_counter()-start,1))),flush=True)
            expected=min(a.max_frames,bm['frames']) if a.max_frames else bm['frames']
            if n!=expected:raise ValueError('Incomplete decode')
            full=n==bm['frames']
            if full and (next(frames,None) is not None or axes.pending is not None):raise ValueError('Unconsumed input rows')
            if full and candidate_audits is not None and next(candidate_audits,None) is not None:
                raise ValueError('Unconsumed candidate suppression audit rows')
            for packet in stream.encode():output.mux(packet)
            output.close();output=None
        if full and sha(a.out/'RAILS.csv')!=sha(bp/'RAILS.csv'):raise ValueError('Rail export changed')
        for filename,digest in frozen.items():
            if sha(filename)!=digest:raise ValueError('Frozen input changed: '+filename)
        verified_txt_files(bp,bm,bv);verified_txt_files(cp,cm,cv)
        m.update(status='complete',frames=n,first_time_s=times[0],last_time_s=times[-1],overlap_observations=accepted_total,
            quarantined_observations=review_total,selective_additions=addition_total,point_rows=points_total,
            rail_csv_byte_identical=full,elapsed_s=time.perf_counter()-start,output_sha256={name:sha(a.out/name) for name in OUTPUTS})
        if use_confirmation:m.update(tile_candidate_pool_count=pool_total,confirmation_survivors=confirmation_total,
            confirmation_rejections=confirmation_rejected_total)
    except BaseException as exc:m.update(status='failed',frames=n,error=repr(exc));raise
    finally:
        if output is not None:output.close()
        write_json(a.out/'MANIFESTE.json',m)
    print(json.dumps(dict(status=m['status'],frames=n,observations=accepted_total,additions=addition_total,quarantined=review_total)),flush=True)

if __name__=='__main__':main()
