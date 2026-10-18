"""Replay existing observations; compare filters without rerunning the model."""
import argparse,csv,gzip,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from railtsv.smoothing import ProfileSmoother,with_measurement_row
from railtsv.adaptive_smoothing import AdaptiveProfileSmoother
from railtsv.metric_geometry import gauge_scenario
from railtsv.common import sha256


def run(video_folder,out,gauge):
    video_folder,out=Path(video_folder),Path(out);out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((video_folder/'video.json').read_text());width=meta['dimensions'][0]
    physical_clock=meta.get('clock',{}).get('filter_clock')=='physical_utc'
    gap=5.0 if physical_clock else max(.15,2.5/meta['fps'])
    filters={'ema':ProfileSmoother(width,max_gap_seconds=gap),'adaptive':AdaptiveProfileSmoother(width,max_gap_seconds=gap)}
    samples=[];second_points={};repeated=0;observed=0
    with (video_folder/'mesures.csv').open(newline='') as src, gzip.open(out/'profils_adaptatifs.jsonl.gz','wt') as profiles:
        for row in csv.DictReader(src):
            frame=int(row['frame_index']);video_t=float(row['seconds']);t=float(row['physical_seconds'] if physical_clock else row['seconds'])
            if row['repeated_source_frame'].lower() in ['true','1']:
                repeated+=1;continue
            observed+=1;segments=json.loads(row['axis_points_json'])
            raw=dict(segments=segments,points=sum(map(len,segments)))
            section={k:(float(row[k]) if row[k] else None) for k in ['left_inner_x','right_inner_x','center_x_px','apparent_width_px']}
            section.update(status=row['status'],row_y=int(row['row_y']))
            raw=with_measurement_row(raw,section)
            outputs={name:f.update(raw,t) for name,f in filters.items()}
            adapt=outputs['adaptive']
            profiles.write(json.dumps(dict(frame_index=frame,seconds=video_t,filter_seconds=t,physical_seconds=t if physical_clock else None,segments=adapt['segments']),separators=(',',':'))+'\n')
            record=dict(frame_index=frame,seconds=video_t,filter_seconds=t,physical_seconds=t if physical_clock else None,source_seconds=row['source_seconds'],status=row['status'],y_px=section['row_y'])
            for name,prof in outputs.items():
                point=next((p for s in prof['segments'] for p in s if p['y']==section['row_y']),None)
                valid=point is not None and row['status']=='candidate'
                record[name+'_center_px']=point['center_x'] if valid else None
                record[name+'_width_px']=point['apparent_width_px'] if valid else None
                if name=='adaptive':record['adaptive_alpha']=point['smoothing_alpha'] if valid else None
            record['raw_center_px']=section['center_x_px'] if record['adaptive_center_px'] is not None else None
            record['raw_width_px']=section['apparent_width_px'] if record['adaptive_width_px'] is not None else None
            scenario=gauge_scenario(section['left_inner_x'],section['right_inner_x'],width,gauge) if record['raw_center_px'] is not None else dict(metric_status='no_valid_section')
            for k in ['metric_status','assumed_gauge_m','scale_m_per_px_assumed','axis_offset_from_image_midline_m_assumed','measured_gauge_m','ground_x_m','ground_z_m','physical_uncertainty_m']:
                record[k]=scenario.get(k)
            samples.append(record)
            second=int(video_t);distance=abs(video_t-(second+.5))
            if adapt['segments'] and (second not in second_points or distance<second_points[second][0]):
                second_points[second]=(distance,frame,video_t,adapt['segments'])
    with (out/'comparaison_par_image.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(samples[0]));writer.writeheader();writer.writerows(samples)
    fields=['second','frame_index','seconds','segment','y_px','left_raw_px','right_raw_px','center_raw_px','left_adaptive_px','right_adaptive_px','center_adaptive_px','width_adaptive_px','metric_status','assumed_gauge_m','scale_m_per_px_assumed','axis_offset_from_image_midline_m_assumed','measured_gauge_m','ground_x_m','ground_z_m','physical_uncertainty_m']
    with (out/'axe_adaptatif_par_seconde.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for second,(_,frame,t,segments) in sorted(second_points.items()):
            for si,seg in enumerate(segments):
                for p in seg:
                    s=gauge_scenario(p['raw_left_inner_x'],p['raw_right_inner_x'],width,gauge)
                    writer.writerow(dict(second=second,frame_index=frame,seconds=t,segment=si,y_px=p['y'],
                        left_raw_px=p['raw_left_inner_x'],right_raw_px=p['raw_right_inner_x'],center_raw_px=p['raw_center_x'],
                        left_adaptive_px=p['left_inner_x'],right_adaptive_px=p['right_inner_x'],center_adaptive_px=p['center_x'],width_adaptive_px=p['apparent_width_px'],**s))
    pairs=[(a,b) for a,b in zip(samples,samples[1:]) if b['frame_index']==a['frame_index']+1 and 0<b['filter_seconds']-a['filter_seconds']<=gap and a['raw_center_px'] is not None and b['raw_center_px'] is not None]
    triples=[(a,b,c) for a,b,c in zip(samples,samples[1:],samples[2:]) if c['frame_index']==b['frame_index']+1==a['frame_index']+2 and 0<b['filter_seconds']-a['filter_seconds']<=gap and 0<c['filter_seconds']-b['filter_seconds']<=gap and all(r['raw_center_px'] is not None for r in [a,b,c])]
    stats={}
    for name in ['raw','ema','adaptive']:
        delta=np.array([b[name+'_center_px']-a[name+'_center_px'] for a,b in pairs])
        acceleration=np.array([c[name+'_center_px']-2*b[name+'_center_px']+a[name+'_center_px'] for a,b,c in triples])
        shift=np.array([abs(r[name+'_center_px']-r['raw_center_px']) for r in samples if r['raw_center_px'] is not None])
        stats[name]=dict(first_difference_rms_px=float(np.sqrt(np.mean(delta**2))),second_difference_rms_px=float(np.sqrt(np.mean(acceleration**2))),
            mean_abs_shift_from_raw_px=float(shift.mean()),p95_abs_shift_from_raw_px=float(np.percentile(shift,95)))
    summary=dict(source_csv_sha256=sha256(video_folder/'mesures.csv'),source_checkpoint_sha256=meta['checkpoint_sha256'],
        observed_frames=observed,repeated_frames_excluded=repeated,matched_adjacent_pairs=len(pairs),matched_adjacent_triples=len(triples),
        metrics=stats,parameters=dict(min_cutoff_hz=.8,beta_normalized=15,derivative_cutoff_hz=1,ema_tau_seconds=.08),
        reference_gauge_m_assumed=gauge,calibrated=False,model_retrained=False,
        limitation='Temporal variation includes real motion; lower variation is not greater accuracy. No temporally annotated ground truth.',
        filter_clock='physical_utc' if physical_clock else 'playback',units='pixels of original image',source_dimensions=meta['dimensions'],max_gap_seconds=gap,fps=meta['fps'])
    (out/'RESULTATS_STABILITE.json').write_text(json.dumps(summary,indent=2))
    # Standalone scientific figures for the project/presentation.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'raw':'#8b9299','ema':'#c67b25','adaptive':'#177a94'}
    labels={'raw':'Brut','ema':'EMA existant','adaptive':'Adaptatif proposé'}
    fig,axes=plt.subplots(2,1,figsize=(12,7),constrained_layout=True)
    for ax,interval in zip(axes,[(55,65),(135,145)]):
        selected=[r for r in samples if interval[0]<=r['filter_seconds']<=interval[1]]
        for name in colors:
            ax.plot([r['filter_seconds'] for r in selected],[r[name+'_center_px'] if r[name+'_center_px'] is not None else np.nan for r in selected],color=colors[name],label=labels[name],lw=1.4)
        ax.set(xlabel='Temps réel écoulé (s)' if physical_clock else 'Temps vidéo (s)',ylabel=f'Axe à y = {samples[0]["y_px"]} (px)');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Stabilité et suivi des déplacements — mêmes prédictions\nComparaison exploratoire sans référence temporelle',fontsize=14)
    for ext in ['png','svg']:fig.savefig(out/f'01_comparaison_stabilite.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    names=list(colors);values=[stats[n]['first_difference_rms_px'] for n in names]
    bars=ax.bar([labels[n] for n in names],values,color=[colors[n] for n in names]);ax.bar_label(bars,fmt='%.2f px',padding=4)
    ax.set(ylabel='RMS des différences entre images (px)',ylim=(0,max(values)*1.22),title=f'Variation de l’axe — {len(pairs):,} paires consécutives comparables\nLa variation inclut le mouvement réel ; ce n’est pas une erreur de mesure')
    ax.spines[['top','right']].set_visible(False)
    for ext in ['png','svg']:fig.savefig(out/f'02_variation_axe.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video-folder',required=True);p.add_argument('--out',required=True)
    p.add_argument('--assumed-gauge-m',type=float,default=None);a=p.parse_args();run(a.video_folder,a.out,a.assumed_gauge_m)
