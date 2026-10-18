"""Build an auditable delivery from completed real inference and frozen evaluations."""
from pathlib import Path
import csv, gzip, json, hashlib, math, shutil, argparse
from contextlib import ExitStack

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def rows(p):
    op=gzip.open if str(p).endswith('.gz') else open
    with op(p,'rt',encoding='utf-8-sig',newline='') as f: yield from csv.DictReader(f)

def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2))

def indexed(segments):
    result={}
    for segid,seg in enumerate(segments):
        assert all(a['y']<b['y'] for a,b in zip(seg,seg[1:]))
        for i,p in enumerate(seg):
            a,b=seg[max(0,i-1)],seg[min(len(seg)-1,i+1)]
            dx,dy=a['center_x']-b['center_x'],a['y']-b['y']
            length=math.hypot(dx,dy)
            assert p['y'] not in result
            assert abs(p['center_x']-(p['left_inner_x']+p['right_inner_x'])/2)<1e-8
            result[p['y']]=dict(x_px=p['center_x'],left_px=p['left_inner_x'],right_px=p['right_inner_x'],
                width_px=p['right_inner_x']-p['left_inner_x'],segment=segid,
                tangent_dx=dx/length if length else '',tangent_dy=dy/length if length else '')
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();root=a.root;out=a.out
    old=root/'03_RESULTATS/FUSION_AXE_ET_VITESSE_20261018'
    run=root/'runs/tsv_record_complet_horloge_20261018'
    evaluation=root/'runs/test_public_reserve_419_20261018_reprise'
    meta=json.loads((run/'video.json').read_text())
    assert meta['complete_source_video'] and meta['frames']==7608 and meta['clock']['filter_clock']=='physical_utc'
    assert json.loads((evaluation/'TERMINE.json').read_text())['complete']
    out.mkdir(parents=True,exist_ok=False)
    full=out/'01_TSV_RECORD_COMPLET';full.mkdir()
    for n in ['rails_annotes.mp4','apercu.png','video.json']:
        shutil.copy2(run/n,full/n)
    (full/'sources_gps').mkdir()
    for n in ['video.gps.csv','video.gpx','video.srt','VERIFICATION_ET_HORLOGES.json']:
        shutil.copy2(old/'vitesse_GPS_TSV_RECORD'/n,full/'sources_gps'/n)
    raw=list(rows(run/'mesures.csv'));smooth=list(rows(run/'mesures_lissees.csv'))
    gps=list(rows(old/'vitesse_GPS_TSV_RECORD/VITESSE_TRAJET_COMPLET_7608_FRAMES.csv'))
    assert len(raw)==len(smooth)==len(gps)==7608
    framefields=[k for k in gps[0] if k not in ['axis_available_in_current_comparison']]+[
        'image_width_px','image_height_px','raw_axis_points','smoothed_axis_points',
        'raw_axis_available','geometry_section_status','geometry_section_reason','temporal_status',
        'error_status','axis_error_px','image_to_metres_status']
    clockfields=['sequence_id','frame_index','video_time_s','utc_time','real_elapsed_s',
        'train_speed_km_h','train_speed_m_s','quality_flags']
    basefields=clockfields+['geometry_section_status','geometry_section_reason','temporal_status']
    geometry=['x_px','left_px','right_px','width_px','segment','tangent_dx','tangent_dy']
    pointfields=basefields+['y_px']+[prefix+'_'+g for prefix in ['raw','smoothed'] for g in geometry]+[
        'error_status','axis_error_px','x_m','depth_m','metric_status']
    vectorfields=basefields+['raw_segments_json','smoothed_segments_json','error_status','axis_error_px']
    stats=dict(frames=0,frames_with_raw_axis=0,frames_with_smoothed_axis=0,
        raw_observed_points=0,smoothed_observed_points=0,missing_speed_frames=0,
        telemetry_flagged_frames=0,utc_gaps_above_5s=0,point_rows=0)
    with ExitStack() as stack:
        def writer(path,fields):
            f=stack.enter_context(gzip.open(path,'wt',encoding='utf-8',newline='') if str(path).endswith('.gz') else path.open('w',encoding='utf-8',newline=''))
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();return w
        fw=writer(full/'FRAMES_VITESSE_GPS.csv',framefields)
        pw=writer(full/'AXE_POINTS.csv.gz',pointfields)
        vw=writer(full/'VECTEURS_PAR_FRAME.csv.gz',vectorfields)
        sw=writer(full/'UNE_FRAME_PAR_SECONDE_DE_LECTURE.csv',framefields)
        previous_second=None
        for i,(r,s,g) in enumerate(zip(raw,smooth,gps)):
            assert int(r['frame_index'])==int(s['frame_index'])==int(g['frame_index'])==i
            assert int(r['source_frame_index'])==i and r['repeated_source_frame']=='False'
            assert abs(float(r['seconds'])-float(g['video_time_s']))<1e-8
            assert abs(float(r['physical_seconds'])-float(g['real_elapsed_s']))<1e-8
            assert r['utc_time']==g['utc_time']
            assert (not r['train_speed_km_h'] and not g['train_speed_km_h']) or abs(float(r['train_speed_km_h'])-float(g['train_speed_km_h']))<1e-8
            # The display has an additional measurement row. The shared export grid is every 8 original pixels.
            rr=indexed([[p for p in seg if p['y']%8==0] for seg in json.loads(r['axis_points_json'])])
            ss=indexed([[p for p in seg if p['y']%8==0] for seg in json.loads(s['axis_points_json'])])
            assert set(ss)==set(rr), 'Smoothing must not invent or erase observed rows'
            fr={k:g[k] for k in gps[0] if k!='axis_available_in_current_comparison'}
            fr.update(image_width_px=1280,image_height_px=720,raw_axis_points=len(rr),smoothed_axis_points=len(ss),
                raw_axis_available=bool(rr),geometry_section_status=r['status'],geometry_section_reason=r['reason'],
                temporal_status=r['temporal_status'],error_status='no_ground_truth',axis_error_px='',image_to_metres_status='not_calibrated')
            fw.writerow(fr)
            second=int(float(g['video_time_s']))
            if second!=previous_second: sw.writerow(fr);previous_second=second
            base={k:g[k] for k in clockfields}
            base.update(geometry_section_status=r['status'],geometry_section_reason=r['reason'],temporal_status=r['temporal_status'])
            def as_segments(points):
                segs={}
                for y,p in sorted(points.items()):segs.setdefault(p['segment'],[]).append(dict(y_px=y,**p))
                return list(segs.values())
            vw.writerow(dict(base,raw_segments_json=json.dumps(as_segments(rr),separators=(',',':')),
                smoothed_segments_json=json.dumps(as_segments(ss),separators=(',',':')),error_status='no_ground_truth',axis_error_px=''))
            for y in range(0,720,8):
                point=dict(base,y_px=y,error_status='no_ground_truth',axis_error_px='',x_m='',depth_m='',metric_status='not_calibrated')
                for prefix,pp in [('raw',rr),('smoothed',ss)]:
                    for k,v in pp.get(y,{}).items():point[prefix+'_'+k]=v
                pw.writerow(point);stats['point_rows']+=1
            stats['frames']+=1;stats['frames_with_raw_axis']+=bool(rr);stats['frames_with_smoothed_axis']+=bool(ss)
            stats['raw_observed_points']+=len(rr);stats['smoothed_observed_points']+=len(ss)
            stats['missing_speed_frames']+=g['train_speed_km_h']==''
            stats['telemetry_flagged_frames']+=bool(g['quality_flags'])
            stats['utc_gaps_above_5s']+=float(g['real_delta_s'] or 0)>5
    dump(full/'COUVERTURE_EXPORT.json',dict(stats,coverage_is_not_accuracy=True))
    # Keep the two historical model outputs available for rectification research.
    comp=out/'02_COMPARAISON_MODELES';comp.mkdir()
    for capture in ['capture_validation','autre_capture_60s']:
        dst=comp/capture;dst.mkdir()
        for n in ['AXES_SYNCHRONISES.csv.gz','VECTEURS_PAR_FRAME.csv.gz','INDEX_FRAMES.csv']:
            shutil.copy2(old/'equipe_gabarit'/capture/n,dst/n)
    shutil.copy2(root/'runs/experience_egopath_20261016_141426/video_capture_validation/rails_annotes.mp4',comp/'capture_validation/rails_annotes_03676.mp4')
    errors=out/'03_ERREURS_12_REFERENCES';shutil.copytree(old/'fusion_validation',errors)
    shutil.copytree(old/'references',errors/'images_et_masques');shutil.copy2(old/'references.json',errors/'references.json')
    # references.json paths are rewritten to their actual delivered location.
    refs=json.loads((errors/'references.json').read_text())
    for row in refs:
        row['image_file']=row['image_file'].replace('references/','images_et_masques/')
        row['mask_file']=row['mask_file'].replace('references/','images_et_masques/')
    dump(errors/'references.json',refs)
    shutil.copytree(evaluation,out/'04_TEST_PUBLIC_419_IMAGES',ignore=shutil.ignore_patterns('DIAGNOSTIC_INTERNE_*'))
    tools=out/'outils';tools.mkdir()
    for n in ['lire_frame.py','verifier_paquet.py']:
        shutil.copy2(Path(__file__).with_name(n),tools/n)
    for n in ['00_LIRE_MOI.txt','MESSAGE_A_ENVOYER.txt','SCHEMA_CSV.json','05_RESULTATS_ET_LIMITES.txt','TESTS_LOGICIELS.json']:
        shutil.copy2(Path(__file__).with_name(n),out/n)
    provenance=dict(date='2026-10-18',new_training=False,new_manual_annotations=False,
        source_videos=[dict(id='TSV_record',filename='video.mp4',project_copy='data/raw/drive_20261014/tsv_record_video.mp4',
            source_sha256=meta['source_sha256'],width=1280,height=720,fps=5,frames=7608,
            playback_duration_s=meta['duration_seconds'],last_frame_playback_s=float(gps[-1]['video_time_s']),
            physical_first_to_last_s=float(gps[-1]['real_elapsed_s']),gps_source='TSV_record/video.gps.csv',
            input_roi_xyxy=[370,0,910,720],axes_coordinates='Original full-frame pixels; ROI offset restored'),
            dict(id='capture_validation',filename='03676_Track insp 9_SOS_0123_20170109ok.AVI',
            source_sha256='e7be5fd15ecd1d9d54ac8fdca7383c7484ee3f5f8ad59dd0d2c270d333889477',
            width=1920,height=1080,fps=30,output_frames=5399,repeated_frames=94,gps_available=False,
            key_warning='Use delivered normalized video and INDEX_FRAMES; original AVI decoded indices differ.')],
        new_inference_video=meta,models=json.loads((old/'protocol.json').read_text())['models'],
        baseline_comparison_provenance='Predictions from 16 October, exported on 18 October; historical smoothing uses playback clock.',
        reference_errors='12 TSV reference images of capture 03676 already used in model selection; not an independent final test.',
        public_test='419 remaining public test images; no byte-identical train/validation collision; scene independence not established.',
        physical_geometry='Not calibrated; pixel geometry only. Supplied speed and integrated travel have separate provenance.',
        archive_cleanup_audit='runs/nettoyage_zip_20261018/SUPPRESSIONS_VERIFIEES.json')
    dump(out/'PROVENANCE.json',provenance)
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
