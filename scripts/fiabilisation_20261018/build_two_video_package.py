"""One selected team output on two named videos, with exact alignment evidence."""
import csv,json,sys,os,shutil,hashlib,math,itertools,collections
from pathlib import Path
import numpy as np
from PIL import Image
from extract_common_frames import extract,digest
R=Path(sys.argv[1]);NAME='GABARIT_2_VIDEOS_V2_20261018';OUT=R/'03_RESULTATS'/NAME
REVISE=len(sys.argv)>2 and sys.argv[2]=='--revise-unreleased'
if REVISE:
    assert OUT.exists() and (OUT/'PROVENANCE.json').exists() and not (OUT/'RELEASED.json').exists()
    audit=R/'runs/fiabilisation_20261018/selection_brut_pour_gabarit';audit.mkdir(exist_ok=True)
    for name in ['ERREUR_REFERENCE_RESUME.json','ERREURS_SUR_FRAMES_REFERENCE.csv','RESUME_ERREURS_PAR_FRAME.csv','PROVENANCE.json']:
        shutil.copy2(OUT/name,audit/name)
    oldzip=R/'02_A_PARTAGER'/(NAME+'.zip')
    if oldzip.exists():
        (audit/'ZIP_BROUILLON_REMPLACE.json').write_text(json.dumps(dict(path=str(oldzip),sha256=digest(oldzip),reason='Unreleased derived draft: measured smoothing worsens the error on 12 matched frames; original smoothed outputs remain preserved in runs and V1.'),indent=2));oldzip.unlink()
else:OUT.mkdir(exist_ok=False)
TOOLS=OUT/'outils';TOOLS.mkdir(exist_ok=True);shutil.copy2(Path(__file__).with_name('extract_common_frames.py'),TOOLS/'extraire_frames_communes.py')
(TOOLS/'requirements.txt').write_text('numpy\nPillow\nimageio-ffmpeg\n')
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def table(path,rows,fields=None):
    with path.open('w' if REVISE else 'x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def read(path):
    with path.open(newline='') as f:yield from csv.DictReader(f)
def preserve_copy(src,dst):
    try:os.link(src,dst)
    except OSError:shutil.copy2(src,dst)
def true(v):return str(v).lower() in ['true','1']
SOURCE_R=R/'runs/tsv_record_complet_horloge_20261018'
SOURCE_V=R/'runs/experience_egopath_20261016_141426/video_capture_validation'
align=json.loads((R/'runs/fiabilisation_20261018/reference_video_alignment/RESULTAT.json').read_text())
assert align['exact_matches']==12
matches={r['frame_index']:r for r in align['references']}
refs=R/'03_RESULTATS/FUSION_AXE_ET_VITESSE_20261018/references'
gt={i:np.asarray(Image.open(refs/(r['image_id']+'_mask.png'))) for i,r in matches.items()}
gps={int(r['frame_index']):r for r in read(R/'03_RESULTATS/GABARIT_LIVRAISON_20261018/01_TSV_RECORD_COMPLET/FRAMES_VITESSE_GPS.csv')}
CHECKPOINT='ae33ea22f1e8eead7e28349304d81a481b3bac3409110ef9dafc60a20b651abd'
POINT_FIELDS=['video_id','frame_index','source_frame_index','video_time_s','source_pts_s','is_new_source_frame','utc_time','real_elapsed_s','image_width_px','image_height_px','segment_id','point_index','x_px','y_px','left_x_px','right_x_px','width_px','tangent_dx','tangent_dy','axis_status','observation_status','held_from_frame','train_speed_km_h','train_speed_m_s','speed_status','speed_uncertainty_m_s','latitude_deg','longitude_deg','telemetry_flags','reference_image_id','reference_x_px','error_signed_px','error_abs_px','error_status']
FRAME_FIELDS=['video_id','frame_index','source_frame_index','video_time_s','source_pts_s','is_new_source_frame','utc_time','real_elapsed_s','time_basis','image_width_px','image_height_px','train_speed_km_h','train_speed_m_s','speed_status','speed_uncertainty_m_s','latitude_deg','longitude_deg','telemetry_flags','axis_status','observation_status','held_from_frame','axis_points','axis_segments_json','reference_image_id','reference_sections','measured_error_sections','axis_mae_px','axis_rmse_px','axis_bias_px','error_status','source_video_sha256']
configs=[('01_TSV_RECORD','TSV_record',SOURCE_R,'',list(range(50,60))),('02_TSV_03676_180S','TSV_03676',SOURCE_V,'',sorted(set(range(1375,1405))|set(matches)))]
summary=[];point_errors=[];reference_summaries=[];global_checks=[]
for folder,vid,run,url,shared in configs:
    dest=OUT/folder;dest.mkdir(exist_ok=REVISE);meta=json.loads((run/'video.json').read_text());timing=json.loads((run/'timing.json').read_text());source=Path(meta['source'])
    assert digest(source)==meta['source_sha256'] and meta['checkpoint_sha256']==CHECKPOINT
    width,height=meta['dimensions'];fps=meta['fps'];count=meta['frames'];speed_present=vid=='TSV_record'
    spec=dict(video_id=vid,original_filename=source.name,source_url=url,source_sha256=meta['source_sha256'],dimensions=[width,height],fps=fps,normalized_frames=count,decoded_source_frames=timing['decoded_frames'],shared_frame_indices=shared,
      extraction_filter=f'fps=fps={fps:.12g}:start_time=0:round=near',indexing='0-based normalized frame_index, source_frame_index is 0-based decoded source index; do not join by old PNG number',
      input_images='Original full image, no crop or resize. Annotated MP4 is for viewing only.',inference_roi=meta.get('inference_roi_xyxy_exclusive'),network_size=[640,480],
      coordinate_system='Original full image pixels; origin top-left, x rightwards, y downwards; point grid y%8==0',
      source_time_is_physical_time=speed_present,physical_clock_note='UTC aligned with supplied telemetry' if speed_present else 'No speed/GPS file identified; physical clock not independently verified',
      selected_model_sha256=CHECKPOINT,model='DeepLabV3Plus ResNet50, public ego-path then TSV; one selected output, raw axis after geometric/temporal rejection, without display smoothing',selected_variant='raw',
      scope='Development outputs; not independent final test')
    dump(dest/'SOURCE_ET_DECOUPAGE.json',spec)
    # The selected raw coordinates are rendered separately after CSV creation.
    # Removing this new-draft hardlink does not touch the historical movie.
    if REVISE and (dest/'VIDEO_ANNOTEE.mp4').exists():(dest/'VIDEO_ANNOTEE.mp4').unlink()
    frames=[];previous=[];previous_status='absent';held_from=None;counts=collections.Counter();references_cache={}
    if not speed_present:
        for idx,m in gt.items():
            lines={}
            for y in range(0,m.shape[0],8):
                xs=np.flatnonzero(m[y]==1)
                if len(xs)>=2 and not np.any(np.diff(xs)>1) and xs[0]>0 and xs[-1]<m.shape[1]-1 and m[y,xs[0]-1]==0 and m[y,xs[-1]+1]==0:lines[y]=(xs[0]+xs[-1])/2
            references_cache[idx]=lines
    with (dest/'AXE_MEDIAN.csv').open('w' if REVISE else 'x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=POINT_FIELDS);writer.writeheader()
        for idx,(raw,smooth) in enumerate(itertools.zip_longest(read(run/'mesures.csv'),read(run/'mesures_lissees.csv'))):
            assert raw and smooth and int(raw['frame_index'])==int(smooth['frame_index'])==idx
            repeat=true(raw['repeated_source_frame']);segments=json.loads(raw['axis_points_json']);segments=[[p for p in s if p['y']%8==0] for s in segments];segments=[s for s in segments if s]
            filtered_points={p['y']:p for s in json.loads(smooth['axis_points_json']) for p in s}
            if repeat:segments=previous;status=previous_status;obs='held_duplicate';held=held_from
            else:
                status='absent' if not segments else 'ok_controles' if raw['status']=='candidate' else 'a_verifier';obs='new_observation';previous=segments;previous_status=status;held_from=idx;held=''
            g=gps[idx] if speed_present else {}
            speed=g.get('train_speed_km_h','');speed_ms=g.get('train_speed_m_s','')
            base=dict(video_id=vid,frame_index=idx,source_frame_index=int(raw['source_frame_index']),video_time_s=float(raw['seconds']),source_pts_s=float(raw['source_seconds']),is_new_source_frame=not repeat,utc_time=g.get('utc_time',''),real_elapsed_s=g.get('real_elapsed_s',''),image_width_px=width,image_height_px=height,
                axis_status=status,observation_status=obs,held_from_frame=held,train_speed_km_h=speed,train_speed_m_s=speed_ms,speed_status=('provided_with_quality_flags' if g.get('quality_flags') else 'provided') if speed else 'missing_in_telemetry' if speed_present else 'no_speed_source_for_this_video',speed_uncertainty_m_s='',latitude_deg=g.get('latitude_deg',''),longitude_deg=g.get('longitude_deg',''),telemetry_flags=g.get('quality_flags',''))
            counts['frames']+=1;counts[status]+=1;counts['repeated_frames']+=repeat;counts['speed_missing']+=not bool(speed)
            ref=references_cache.get(idx,{});reference_id=matches[idx]['image_id'] if not speed_present and idx in matches else ''
            errors=[];raw_errors=[];vector=[];written=[]
            for seg_id,seg in enumerate(segments):
                vector.append([[float(p['center_x']),int(p['y'])] for p in seg])
                for j,p in enumerate(seg):
                    q0,q1=seg[max(0,j-1)],seg[min(len(seg)-1,j+1)]
                    dx=q0['center_x']-q1['center_x'];dy=q0['y']-q1['y'];norm=math.hypot(dx,dy)
                    row=dict(base,segment_id=seg_id,point_index=j,x_px=p['center_x'],y_px=p['y'],left_x_px=p['left_inner_x'],right_x_px=p['right_inner_x'],width_px=p['right_inner_x']-p['left_inner_x'],tangent_dx=dx/norm if norm else '',tangent_dy=dy/norm if norm else '',reference_image_id=reference_id,reference_x_px='',error_signed_px='',error_abs_px='',error_status='not_measured_no_reference' if not reference_id else 'reference_not_evaluable_at_this_row')
                    if p['y'] in ref:
                        error=p['center_x']-ref[p['y']];errors.append(error);rawerr=p.get('raw_center_x',p['center_x'])-ref[p['y']];raw_errors.append(rawerr)
                        row.update(reference_x_px=ref[p['y']],error_signed_px=error,error_abs_px=abs(error),error_status='measured_vs_reviewed_polygon_on_same_frame')
                        fx=filtered_points[p['y']]['center_x'];fe=fx-ref[p['y']]
                        point_errors.append(dict(video_id=vid,frame_index=idx,video_time_s=idx/fps,reference_image_id=reference_id,y_px=p['y'],reference_x_px=ref[p['y']],x_selected_px=p['center_x'],x_raw_px=p['center_x'],x_smoothed_px=fx,error_selected_px=error,error_raw_px=rawerr,error_smoothed_px=fe,scope='Selected raw axis; smoothed historical result retained for comparison only. 12 development frames, same decoded pixels; imperfect human reference.'))
                    assert abs(row['x_px']-(row['left_x_px']+row['right_x_px'])/2)<1e-6
                    assert 0<=row['left_x_px']<row['right_x_px']<width and 0<=row['y_px']<height
                    written.append(row)
            if not written:written=[dict(base,reference_image_id=reference_id,error_status='no_axis_to_compare' if reference_id else 'not_measured_no_reference')]
            writer.writerows(written);counts['point_rows']+=len(written);counts['observed_or_held_axis_points']+=sum(map(len,segments))
            fr=dict(base,time_basis='utc_aligned_telemetry' if speed_present else 'video_playback_only',axis_points=sum(map(len,segments)),axis_segments_json=json.dumps(vector,separators=(',',':')),reference_image_id=reference_id,reference_sections=len(ref) if reference_id else '',measured_error_sections=len(errors),axis_mae_px=float(np.abs(errors).mean()) if errors else '',axis_rmse_px=float(np.sqrt(np.mean(np.square(errors)))) if errors else '',axis_bias_px=float(np.mean(errors)) if errors else '',error_status='partial_measured_vs_reviewed_polygon' if errors else 'no_axis_to_compare' if reference_id else 'not_measured_no_reference',source_video_sha256=spec['source_sha256'])
            frames.append(fr)
            if reference_id:
                fe=[filtered_points[y]['center_x']-ref[y] for y in ref if y in filtered_points]
                reference_summaries.append(dict(video_id=vid,frame_index=idx,reference_image_id=reference_id,reference_sections=len(ref),compared_sections=len(errors),coverage=len(errors)/len(ref) if ref else None,selected_variant='raw',selected_mae_px=fr['axis_mae_px'],selected_rmse_px=fr['axis_rmse_px'],raw_mae_px=float(np.abs(raw_errors).mean()) if errors else None,smoothed_mae_px=float(np.abs(fe).mean()) if fe else None))
                assert int(raw['source_frame_index'])==int(reference_id.rsplit('_',1)[1])
    assert len(frames)==count and all(abs(r['video_time_s']-i/fps)<1e-8 for i,r in enumerate(frames))
    table(dest/'FRAMES_VITESSE_VECTEUR.csv',frames,FRAME_FIELDS)
    for r in frames:
        if r['train_speed_km_h']:assert abs(float(r['train_speed_m_s'])-float(r['train_speed_km_h'])/3.6)<1e-8
    table(dest/'EXEMPLE_SEQUENCE_FRAMES.csv',[r for r in frames if r['frame_index'] in shared],FRAME_FIELDS)
    sample=[]
    for r in read(dest/'AXE_MEDIAN.csv'):
        if int(r['frame_index']) in shared:sample.append(r)
    table(dest/'EXEMPLE_SEQUENCE_POINTS.csv',sample,POINT_FIELDS)
    print(vid,dict(counts),'CSV ready; extracting shared PNG inputs',flush=True)
    images=list(read(dest/'IMAGES_COMMUNES/MANIFESTE_IMAGES.csv')) if REVISE else extract(source,spec,dest/'IMAGES_COMMUNES',shared)
    spec['shared_rgb_sha256']={str(r['frame_index']):r['rgb_sha256'] for r in images};dump(dest/'SOURCE_ET_DECOUPAGE.json',spec)
    if speed_present:
        shutil.copy2(R/'03_RESULTATS/GABARIT_NOTRE_EQUIPE_V1_20261018/technique/GPS_SOURCE.csv',dest/'GPS_SOURCE.csv')
    else:
        refdir=dest/'REFERENCES_12';refdir.mkdir(exist_ok=REVISE)
        for i,m in matches.items():
            target=f'{vid}_frame_{i:06d}_mask.png';shutil.copy2(refs/(m['image_id']+'_mask.png'),refdir/target)
        table(refdir/'CORRESPONDANCE_EXACTE.csv',list(matches.values()))
    summary.append(dict(video_id=vid,folder=folder,source=spec['original_filename'],source_url=url,source_sha256=spec['source_sha256'],dimensions=spec['dimensions'],fps=fps,duration_s=count/fps,decoded_source_frames=timing['decoded_frames'],normalized_frames=count,speed_available=speed_present,reference_frames=0 if speed_present else 12,counts=dict(counts),shared_pngs=len(images),inference_reused_from=str(run.relative_to(R)),checkpoint_sha256=CHECKPOINT))
table(OUT/'ERREURS_SUR_FRAMES_REFERENCE.csv',point_errors)
table(OUT/'RESUME_ERREURS_PAR_FRAME.csv',reference_summaries)
dump(OUT/'PROVENANCE.json',dict(videos=summary,model='One selected adapted model, not an ensemble with the other team',selected_variant='raw_without_display_smoothing',selection_reason='On 12 same-frame references, raw axis has lower mean error than historical adaptive filtering. This is a development decision, not independent final evidence.',new_training_used=False,new_model_inference=False,reference_alignment='All 12 reference PNGs matched to normalized decoded frames by full RGB byte equality',error_scope='Measured only where a reviewed polygon and predicted axis are available on the same frame; development data, not final test; errors are in native pixels',physical_accuracy_validated=False,blank_means='unknown/not measured, never zero',repeated_frame_policy='Held previous selected axis copied with observation_status=held_duplicate and is_new_source_frame=false; never a new observation'))
dump(OUT/'COLONNES.json',dict(frame_file='FRAMES_VITESSE_VECTEUR.csv',point_file='AXE_MEDIAN.csv',join_key=['video_id','frame_index'],cross_team_point_join=['video_id','frame_index','y_px'],frame_index='0-based normalized displayed frame; do not use original annotation filename number',source_frame_index='0-based decoded source image before regularization',video_time_s='Playback seconds = frame_index/fps',source_pts_s='Source presentation timestamp; not UTC',real_elapsed_s='Physical elapsed seconds only if telemetry verifies them',axis_segments_json='Selected RAW axis. JSON list of continuous segments, each a list of [x_px,y_px] points. Do not connect separate segments.',pixel_coordinates='Original full image, x right y down; no shared resize/crop required',tangent='Local unit 2D direction towards image top; not train velocity',speed='km/h and m/s from telemetry, blank if unknown; flags retained; instrumental uncertainty not provided',error_signed_px='Selected raw x minus annotated x at same y, only on reviewed matched frames',error_abs_px='Absolute value of that signed error',axis_mae_px='Mean absolute error on comparable rows of this reference frame; blank elsewhere',accuracy='No per-frame confidence/accuracy percentage is invented',held_duplicate='Axis repeated only for display alignment; should not be treated as independent data or used to integrate motion twice'))
print(json.dumps(summary,indent=2),flush=True)
