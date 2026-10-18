"""Exact frame-index GPS join for the supplied TSV_record movie (time-lapse)."""
from pathlib import Path
import csv,json,gzip,datetime,math,shutil,sys,xml.etree.ElementTree as ET
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'railtsv').is_dir())
sys.path.insert(0,str(ROOT))
from railtsv.common import sha256,dump
from scripts.diagnostic_plots import write_csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'runs/fusion_axes_20261018'
DEST=ROOT/'03_RESULTATS/FUSION_AXE_ET_VITESSE_20261018'
SOURCE=ROOT/'data/raw/TSV_record'

def read(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def distance(a,b):
    lat1,lon1,lat2,lon2=map(math.radians,[a[0],a[1],b[0],b[1]])
    z=math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371008.8*2*math.asin(min(1,math.sqrt(z)))

def main():
    dest=DEST/'vitesse_GPS_TSV_RECORD';dest.mkdir(exist_ok=False)
    source=read(SOURCE/'video.gps.csv');meta=json.loads((OUT/'equipe_gabarit/autre_capture_60s/CONTRAT.json').read_text())
    video_hash=sha256(SOURCE/'video.mp4');assert video_hash==meta['source_sha256']
    utc=[datetime.datetime.fromisoformat(r['utc_time'].replace('Z','+00:00')) for r in source]
    real=[(t-utc[0]).total_seconds() for t in utc]
    assert len(source)==7608 and all(b>a for a,b in zip(real,real[1:]))
    assert all(int(r['frame'])==i and abs(float(r['video_time_s'])-i/5)<1e-8 for i,r in enumerate(source))
    tree=ET.parse(SOURCE/'video.gpx');pts=tree.findall('.//{*}trkpt');assert len(pts)==len(source)
    for p,r in zip(pts,source):
        assert float(p.attrib['lat'])==float(r['latitude']) and float(p.attrib['lon'])==float(r['longitude'])
        assert p.find('{*}time').text==r['utc_time']
    rows=[];global_distance=0.;segment_distance=0.;segment_id=0;global_valid=True;gaps=0
    for i,r in enumerate(source):
        speed=float(r['speed_kmh']) if r['speed_kmh'] else None
        assert speed is None or math.isfinite(speed) and speed>=0
        dt=real[i]-real[i-1] if i else None
        previous=rows[-1]['train_speed_km_h'] if rows else None
        step=(previous+speed)/2/3.6*dt if i and previous is not None and speed is not None and dt<=5 else None
        if i and step is None:global_valid=False;segment_id+=1;segment_distance=0.;gaps+=1
        if step is not None:global_distance+=step;segment_distance+=step
        if i==0 and speed is None:global_valid=False
        pos=(float(r['latitude']),float(r['longitude']))
        d=distance((float(source[i-1]['latitude']),float(source[i-1]['longitude'])),pos) if i else None
        flags=[]
        if dt is not None and dt>5:flags.append('utc_gap_over_5s')
        if speed is None:flags.append('speed_missing')
        if i and d==0 and speed is not None and speed>5:flags.append('unchanged_gps_position_while_speed_nonzero')
        gps_speed=d/dt*3.6 if dt else None
        # Diagnostic flag, not a rejection or ground-truth uncertainty bound.
        if gps_speed is not None and speed is not None and abs(gps_speed-speed)>20:flags.append('position_speed_disagreement_over_20kmh')
        rows.append(dict(sequence_id='TSV_record',frame_index=i,video_time_s=float(r['video_time_s']),
            utc_time=r['utc_time'],real_elapsed_s=real[i],real_delta_s=dt,
            local_playback_acceleration=dt/.2 if dt else None,
            train_speed_km_h=speed,train_speed_m_s=speed/3.6 if speed is not None else None,
            speed_status='reported_by_supplied_gps_csv' if speed is not None else 'missing_in_source',
            speed_uncertainty_m_s=None,speed_uncertainty_status='not_provided_by_source',
            latitude_deg=pos[0],longitude_deg=pos[1],gps_step_distance_m=d,gps_position_derived_speed_km_h=gps_speed,
            interval_distance_from_speed_m=step,
            distance_from_record_start_m=global_distance if global_valid else None,
            continuous_segment_id=segment_id,segment_distance_from_speed_m=segment_distance if speed is not None else None,
            quality_flags=';'.join(flags),distance_status='trapezoid_on_real_utc_clock_max_gap_5s',
            source_sha256=video_hash,axis_available_in_current_comparison=i<300,
            camera_pose_status='not_calibrated',speed_source='video.gps.csv'))
    write_csv(dest/'VITESSE_TRAJET_COMPLET_7608_FRAMES.csv',rows)
    # All 300 existing axis frames have direct telemetry; no interpolation for speed.
    short=[]
    frames=read(DEST/'equipe_gabarit/autre_capture_60s/INDEX_FRAMES.csv')
    assert len(frames)==300
    for frame in frames:
        i=int(frame['frame_index']);gps=rows[i]
        assert abs(float(frame['time_s'])-gps['video_time_s'])<1e-8
        short.append(dict(frame,**{k:v for k,v in gps.items() if k not in ['sequence_id','frame_index']}))
    write_csv(dest/'VITESSE_POUR_AXES_300_FRAMES.csv',short)
    fields_to_add=['utc_time','real_elapsed_s','real_delta_s','train_speed_m_s','train_speed_km_h',
        'speed_status','speed_uncertainty_m_s','distance_from_record_start_m','interval_distance_from_speed_m',
        'continuous_segment_id','segment_distance_from_speed_m','quality_flags']
    src=DEST/'equipe_gabarit/autre_capture_60s/AXES_SYNCHRONISES.csv.gz'
    with gzip.open(src,'rt',newline='') as fi,gzip.open(dest/'AXES_ET_VITESSE_300_FRAMES.csv.gz','wt',newline='') as fo:
        writer=None;count=0
        for r in csv.DictReader(fi):
            gps=rows[int(r['frame_index'])];assert float(r['time_s'])==gps['video_time_s']
            r.update({k:gps[k] for k in fields_to_add})
            if writer is None:writer=csv.DictWriter(fo,fieldnames=list(r));writer.writeheader()
            writer.writerow(r);count+=1
    assert count==27000
    for name in ['video.gps.csv','video.gpx','video.srt']:shutil.copy2(SOURCE/name,dest/name)
    summary=dict(video_sha256=video_hash,gps_csv_sha256=sha256(SOURCE/'video.gps.csv'),
        matching_video='TSV_record/video.mp4 = data/raw/drive_20261014/tsv_record_video.mp4',
        frame_index_alignment=True,gpx_positions_and_utc_match=True,frames=len(rows),
        frames_with_speed=sum(r['train_speed_km_h'] is not None for r in rows),
        frames_without_speed=sum(r['train_speed_km_h'] is None for r in rows),
        full_video_last_frame_time_s=rows[-1]['video_time_s'],full_real_elapsed_s=real[-1],
        median_real_interval_s=float(np.median(np.diff(real))),max_real_interval_s=float(max(np.diff(real))),
        median_playback_acceleration=float(np.median(np.diff(real))/.2),
        first_300_real_elapsed_s=real[299],first_300_speed_min_km_h=min(r['train_speed_km_h'] for r in rows[:300]),
        first_300_speed_max_km_h=max(r['train_speed_km_h'] for r in rows[:300]),
        first_300_distance_from_speed_m=rows[299]['distance_from_record_start_m'],
        axis_frames_available=300,axis_points_grid_rows=27000,
        physical_distance_uses='Real UTC intervals, not video playback intervals',
        speed_error_calibration=None,quality_flags_count=sum(bool(r['quality_flags']) for r in rows),
        unintegrated_intervals=gaps,integration_max_gap_s=5,
        note='Speed values are supplied telemetry, not independent certified truth. Position-derived speeds can disagree; flagged, not replaced. No telemetry for video_03676.')
    dump(dest/'VERIFICATION_ET_HORLOGES.json',summary)
    # New graph: timebase and physical speed, never a model-accuracy graph.
    fig,axs=plt.subplots(1,3,figsize=(14,5))
    a=rows[:300];x=[r['video_time_s'] for r in a];rt=[r['real_elapsed_s'] for r in a]
    axs[0].plot(x,rt,color='#176b94');axs[0].plot(x,x,':',color='grey',label='Même durée (repère)');axs[0].set(xlabel='Temps de lecture vidéo (s)',ylabel='Temps réel UTC écoulé (s)',title='Deux horloges distinctes');axs[0].legend(fontsize=8)
    axs[1].plot(rt,[r['train_speed_km_h'] for r in a],color='#bc7222');axs[1].set(xlabel='Temps réel écoulé (s)',ylabel='Vitesse fournie (km/h)',title='Vitesse par frame')
    axs[2].plot(rt,[r['distance_from_record_start_m'] for r in a],color='#176b94');axs[2].set(xlabel='Temps réel écoulé (s)',ylabel='Avancement estimé (m)',title='Intégration sur les vrais intervalles')
    fig.suptitle('TSV_record : la vidéo est accélérée, la vitesse reste une vitesse physique',fontsize=16)
    fig.text(.03,.02,'300 frames appariées exactement ; 59,8 s de lecture entre la première et la dernière frame = 604,039 s réelles.\nVitesses issues du fichier fourni ; erreur instrumentale non renseignée. Avancement estimé par trapèzes, seuil de continuité 5 s.',fontsize=9)
    fig.tight_layout(rect=[0,.12,1,.95]);fig.savefig(DEST/'graphiques/12_vitesse_et_horloges.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
