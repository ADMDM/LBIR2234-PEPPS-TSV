"""Aligned axis/vector interchange and controlled smoothing/fusion diagnostics."""
from axes_eval_20261018 import *
import gzip,math,shutil,itertools
from railtsv.adaptive_smoothing import AdaptiveProfileSmoother
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def csvrows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))

def vector_points(profile):
    """Unit tangents face forward (towards decreasing image y), segment-local."""
    result={}
    for sid,seg in enumerate(profile['segments']):
        ordered=sorted(seg,key=lambda p:p['y'],reverse=True)
        for i,p in enumerate(ordered):
            a,b=ordered[max(0,i-1)],ordered[min(len(ordered)-1,i+1)]
            dx,dy=b['center_x']-a['center_x'],b['y']-a['y'];norm=math.hypot(dx,dy)
            result[p['y']]=dict(p,segment_id=sid,point_index=i,
                tangent_dx=dx/norm if norm else None,tangent_dy=dy/norm if norm else None)
    return result

def compare_validation():
    dest=OUT/'fusion_validation';dest.mkdir()
    models=list(MODELS);wide=[];long=[];per_image=[];summaries=[]
    errors={n:[] for n in models};spatial_errors={n:[] for n in models}
    refs=json.loads((OUT/'references.json').read_text())
    for row in refs:
        stem=row['image_id'];ref=mask_read(OUT/row['mask_file'],[0,1,255]);h,w=ref.shape
        target=indexed(reference_profile(ref))
        profiles={n:json.loads((OUT/'evaluation'/n/'profiles'/f'{stem}.json').read_text()) for n in models}
        variants={n:{'raw':vector_points(p),'spatial_experimental':vector_points(spatial_profile(p))} for n,p in profiles.items()}
        for y in range(0,h,8):
            r=target.get(y);gt=r['center_x'] if r else None
            item=dict(image_id=stem,image_sha256=row['image_sha256'],image_width_px=w,image_height_px=h,
                y_px=y,reference_x_px=gt,reference_status='annotated' if r else 'not_evaluable',
                video_time_s=None,video_time_status='not_verified')
            for n in models:
                p=variants[n]['raw'].get(y);sp=variants[n]['spatial_experimental'].get(y)
                err=p['center_x']-gt if p and r else None
                item[n+'_x_px']=p['center_x'] if p else None;item[n+'_error_px']=err
                if err is not None:errors[n].append(err)
                if sp and r:spatial_errors[n].append(sp['center_x']-gt)
                for variant in variants[n]:
                    q=variants[n][variant].get(y)
                    long.append(dict(image_id=stem,image_sha256=row['image_sha256'],model_id=n,variant=variant,
                        width_px=w,height_px=h,y_px=y,observed=q is not None,
                        segment_id=q['segment_id'] if q else None,point_index=q['point_index'] if q else None,
                        x_px=q['center_x'] if q else None,left_x_px=q['left_inner_x'] if q else None,
                        right_x_px=q['right_inner_x'] if q else None,
                        tangent_dx=q['tangent_dx'] if q else None,tangent_dy=q['tangent_dy'] if q else None,
                        reference_x_px=gt,error_signed_px=q['center_x']-gt if q and r else None,
                        error_abs_px=abs(q['center_x']-gt) if q and r else None,
                        error_status='measured_vs_polygon' if q and r else 'not_evaluable'))
            a,b=item['ancien_tsv_x_px'],item['ego_tsv_x_px']
            item.update(mean_fusion_x_px=(a+b)/2 if a is not None and b is not None else None,
                        disagreement_px=abs(a-b) if a is not None and b is not None else None)
            item['mean_fusion_error_px']=item['mean_fusion_x_px']-gt if item['mean_fusion_x_px'] is not None and r else None
            wide.append(item)
    write_csv(dest/'AXES_MEMES_IMAGES.csv',wide);write_csv(dest/'VECTEURS_ET_ERREURS.csv',long)
    for n in models:
        s=json.loads((OUT/'evaluation'/n/'summary.json').read_text());s['spatial_mae_px']=stats(spatial_errors[n])['mae_px'];summaries.append(s)
    write_csv(dest/'COMPARAISON_MODELES.csv',summaries)
    common=[r for r in wide if all(r[n+'_error_px'] is not None for n in ['ancien_tsv','ego_tsv'])]
    a=np.array([r['ancien_tsv_error_px'] for r in common]);b=np.array([r['ego_tsv_error_px'] for r in common])
    comparisons=[dict(method=n,**stats(arr)) for n,arr in [('ancien_tsv',a),('ego_tsv',b),('moyenne_50_50',(a+b)/2)]]
    write_csv(dest/'FUSION_SECTIONS_COMMUNES.csv',comparisons)
    corr=float(np.corrcoef(a,b)[0,1]);cov=np.cov(np.array([a,b]),ddof=1).tolist()
    # No hidden fitted correction: descriptive diagnostics, not predictive uncertainty.
    dump(dest/'DIAGNOSTIC_FUSION.json',dict(common_sections=len(common),residual_correlation=corr,
        residual_covariance_px2=cov,covariance_order=['ancien_tsv','ego_tsv'],
        scope='12 development images, correlated rows, no independence claim',
        fusion_weight_training=False,error_calibrated_per_video_frame=False))
    for image in refs:
        rs=[r for r in common if r['image_id']==image['image_id']]
        for n in ['ancien_tsv','ego_tsv','mean_fusion']:
            per_image.append(dict(image_id=image['image_id'],method=n,**stats([r[n+'_error_px'] for r in rs])))
    write_csv(dest/'FUSION_PAR_IMAGE.csv',per_image)
    # A precise, directly usable single-image example, full original resolution.
    example=refs[0]['image_id']
    write_csv(dest/'EXEMPLE_UNE_IMAGE.csv',[r for r in long if r['image_id']==example and r['variant']=='raw'])
    return summaries,comparisons,corr

def series_exports():
    dest=OUT/'equipe_gabarit';dest.mkdir()
    cases={'capture_validation':(BASE/'video',EGO/'video_capture_validation'),
           'autre_capture_60s':(ROOT/'03_RESULTATS/AMELIORATIONS_20261014/video_nouvelle_capture_60s',EGO/'video_autre_capture_60s')}
    allmeta=[]
    for name,(old,new) in cases.items():
        path=dest/name;path.mkdir();ma=json.loads((old/'video.json').read_text());mb=json.loads((new/'video.json').read_text())
        for field in ['source_sha256','fps','dimensions']:assert ma[field]==mb[field]
        assert ma.get('inference_roi_xyxy_exclusive')==mb.get('inference_roi_xyxy_exclusive')
        width,height=mb['dimensions'];fps=mb['fps'];n=mb['frames']
        filters={k:AdaptiveProfileSmoother(width,max_gap_seconds=max(.15,2.5/fps)) for k in ['ancien','ego']}
        point_file=path/'AXES_SYNCHRONISES.csv.gz';vec_file=path/'VECTEURS_PAR_FRAME.csv.gz'
        frames=[];point_count=0;repeats=0
        with (old/'mesures.csv').open(newline='') as fa,(new/'mesures.csv').open(newline='') as fb,\
             gzip.open(point_file,'wt',newline='',encoding='utf-8') as fp,\
             gzip.open(vec_file,'wt',newline='',encoding='utf-8') as fv,\
             (path/'EXTRAIT_10_A_12_SECONDES.csv').open('w',newline='',encoding='utf-8') as fe:
            ra,rb=csv.DictReader(fa),csv.DictReader(fb);pw=vw=ew=None
            for i in range(n):
                a,b=next(ra),next(rb)
                for key in ['frame_index','source_frame_index','seconds','source_seconds','repeated_source_frame']:
                    assert a[key]==b[key],(name,i,key)
                assert int(a['frame_index'])==i
                repeat=a['repeated_source_frame']=='True';repeats+=repeat
                frame=dict(sequence_id=name,frame_index=i,time_s=float(a['seconds']),
                    source_frame_index=int(a['source_frame_index']),source_time_s=float(a['source_seconds']),
                    repeated_source_frame=repeat,width_px=width,height_px=height)
                maps={};vectors={}
                for key,row in [('ancien',a),('ego',b)]:
                    raw=dict(segments=json.loads(row['axis_points_json']))
                    assert not repeat or not raw['segments']
                    smooth=filters[key].update(raw,frame['source_time_s']) if not repeat else dict(segments=[])
                    maps[key]=vector_points(raw);maps[key+'_lisse']=vector_points(smooth)
                    vr=dict(**frame,model_id=key+'_tsv',source_sha256=mb['source_sha256'],
                        checkpoint_sha256=(ma if key=='ancien' else mb)['checkpoint_sha256'],
                        axis_raw_json=json.dumps(raw['segments'],separators=(',',':')),
                        axis_filtered_json=json.dumps(smooth['segments'],separators=(',',':')),
                        observed_points=len(maps[key]),error_status='no_ground_truth',error_px=None)
                    if vw is None:vw=csv.DictWriter(fv,fieldnames=list(vr));vw.writeheader()
                    vw.writerow(vr)
                frame.update(ancien_points=len(maps['ancien']),ego_points=len(maps['ego']))
                frames.append(frame)
                for y in range(0,height,8):
                    r=dict(**frame,y_px=y)
                    for k in maps:
                        p=maps[k].get(y)
                        for outkey,inkey in [('x_px','center_x'),('left_px','left_inner_x'),('right_px','right_inner_x'),
                            ('segment','segment_id'),('tangent_dx','tangent_dx'),('tangent_dy','tangent_dy')]:
                            r[k+'_'+outkey]=p[inkey] if p else None
                    x,z=r['ancien_x_px'],r['ego_x_px']
                    r.update(disagreement_px=abs(x-z) if x is not None and z is not None else None,
                        fusion_mean_raw_x_px=(x+z)/2 if x is not None and z is not None else None,
                        error_status='no_ground_truth',ancien_error_px=None,ego_error_px=None,
                        x_m=None,y_m=None,metric_status='uncalibrated')
                    if pw is None:
                        pw=csv.DictWriter(fp,fieldnames=list(r));pw.writeheader()
                        ew=csv.DictWriter(fe,fieldnames=list(r));ew.writeheader()
                    pw.writerow(r);point_count+=1
                    if 10<=frame['time_s']<12:ew.writerow(r)
                if (i+1)%900==0:print(f'CSV {name}: {i+1}/{n}',flush=True)
        write_csv(path/'INDEX_FRAMES.csv',frames)
        meta=dict(sequence_id=name,source_path=mb['source'],source_sha256=mb['source_sha256'],
            original_dimensions=mb['dimensions'],roi_xyxy_exclusive=mb.get('inference_roi_xyxy_exclusive'),
            source_fps=fps,frames=n,repeated_frames=repeats,point_rows=point_count,
            point_grid_step_px=8,point_file=point_file.name,vector_file=vec_file.name,
            smoothing=dict(method='paired_one_euro',min_cutoff_hz=.8,beta_normalized=15,derivative_cutoff_hz=1,
                           max_gap_s=max(.15,2.5/fps)),
            frame_alignment_checked=True,new_inference=False,
            note='Re-export of predictions from 2026-10-16; both filters replayed identically. No frame-level ground truth.',
            coordinate_system='Original pixels: (0,0) top-left, x right, y down; tangent towards decreasing y',
            model_checkpoints={'ancien_tsv':ma['checkpoint_sha256'],'ego_tsv':mb['checkpoint_sha256']})
        dump(path/'CONTRAT.json',meta);allmeta.append(meta)
    dump(dest/'MANIFEST.json',allmeta)
    return allmeta

def smoothing_tests():
    dest=OUT/'tests_lissage';dest.mkdir()
    configs=[('brut',None,None),('actuel',.8,15),('doux',1.6,15),('fort',.4,15),('reactif',.8,30)]
    result=[];sequences={}
    # Replay real same-row measurements; do not mistake camera motion for noise.
    for name in ['capture_validation','autre_capture_60s']:
        rows=csvrows(EGO/'comparaison'/f'COMPARAISON_{name.upper()}.csv')
        width=1920 if name=='capture_validation' else 1280;fps=30 if width==1920 else 5
        for label,cutoff,beta in configs:
            f=AdaptiveProfileSmoother(width,min_cutoff_hz=cutoff or .8,beta=beta or 0,max_gap_seconds=max(.15,2.5/fps))
            vals=[]
            for r in rows:
                raw=r['egopath_profile_raw_x'];repeated=r['repeated']=='True';t=float(r['source_seconds'])
                if repeated:vals.append(None);continue
                # Same centre motion on fixed-width synthetic edges, not new boundary estimates.
                p=dict(segments=[[dict(y=0,left_inner_x=float(raw)-5,right_inner_x=float(raw)+5,center_x=float(raw))]] if raw else [])
                s=f.update(p,t)
                vals.append(float(raw) if label=='brut' and raw else s['segments'][0][0]['center_x'] if s['segments'] else None)
            sequences[(name,label)]=vals
        valid=[i for i,r in enumerate(rows) if all(sequences[(name,k)][i] is not None for k,_,_ in configs)]
        pairs=[i for i in range(1,len(rows)) if i in valid and i-1 in valid and float(rows[i]['source_seconds'])-float(rows[i-1]['source_seconds'])<=1.5/fps]
        triples=[i for i in pairs if i-1 in pairs]
        for label,cutoff,beta in configs:
            v=sequences[(name,label)];raw=sequences[(name,'brut')]
            result.append(dict(sequence=name,method=label,pairs=len(pairs),triples=len(triples),
                delta_rms_px=float(np.sqrt(np.mean([(v[i]-v[i-1])**2 for i in pairs]))),
                delta2_rms_px=float(np.sqrt(np.mean([(v[i]-2*v[i-1]+v[i-2])**2 for i in triples]))),
                distance_to_raw_mae_px=float(np.mean([abs(v[i]-raw[i]) for i in valid])),
                note='Centre-only replay; temporal variation and deviation from raw, not accuracy or rail-boundary filter'))
    write_csv(dest/'REJEU_CENTRE_VIDEO.csv',result)
    # Known synthetic truth isolates the lag/noise trade-off, no TSV performance claim.
    synthetic=[];rng=np.random.default_rng(42);t=np.arange(600)/30
    truth=960+25*np.sin(2*np.pi*.3*t)+np.where(t>=10,15,0)
    measured=truth+rng.normal(0,2,len(t))
    synthetic_series=[]
    for label,cutoff,beta in configs:
        f=AdaptiveProfileSmoother(1920,min_cutoff_hz=cutoff or .8,beta=beta or 0)
        v=[]
        for tt,x in zip(t,measured):
            p=dict(segments=[[dict(y=864,left_inner_x=x-150,right_inner_x=x+150,center_x=x)]])
            s=f.update(p,float(tt));v.append(float(x) if label=='brut' else s['segments'][0][0]['center_x'])
        arr=np.asarray(v)
        synthetic.append(dict(method=label,truth_known=True,source='synthetic_not_TSV',**stats(arr-truth)))
        for i in range(len(t)):synthetic_series.append(dict(time_s=t[i],method=label,truth_x=truth[i],prediction_x=arr[i]))
    write_csv(dest/'TEST_SYNTHETIQUE_ERREUR_CONNUE.csv',synthetic);write_csv(dest/'TEST_SYNTHETIQUE_SERIES.csv',synthetic_series)
    dump(dest/'DECISION.json',dict(production_filter_changed=False,
        reason='No continuous temporal ground truth. Lower vibration alone cannot select a more accurate filter.',
        synthetic_truth_only_for_implementation_test=True))
    return result,synthetic

def figures(summary,fusion,corr,replay,synthetic):
    dest=OUT/'graphiques';dest.mkdir()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name,caption):
        fig.text(.03,.015,caption,fontsize=9,color='#475569');fig.tight_layout(rect=[0,.1,1,.95]);fig.savefig(dest/(name+'.png'),dpi=170);plt.close(fig)
    order=['public_seul_natif','public_seul_vga','ancien_tsv','ego_tsv'];labels=['Public seul\n640 × 384','Public seul\n640 × 480','RailSem générique\n+ TSV','Ego-path\n+ TSV'];data={s['model']:s for s in summary}
    fig,axs=plt.subplots(1,3,figsize=(14,5.5))
    for ax,key,title,mul in zip(axs,['dice_macro','coverage_pct','rmse_px'],['Dice moyen (%)','Sections détectées (%)','RMSE de l’axe (px)'],[100,1,1]):
        vv=[data[n][key]*mul for n in order];bb=ax.bar(labels,vv,color=['#64748b','#94a3b8','#176b94','#bc7222']);ax.set_title(title);ax.tick_params(axis='x',labelsize=9);ax.set_ylim(0,max(vv)*1.18)
        for b,v in zip(bb,vv):ax.text(b.get_x()+b.get_width()/2,v,f'{v:.2f}',ha='center',va='bottom')
    fig.suptitle('Peut-on utiliser uniquement les annotations publiques ?',fontsize=17)
    save(fig,'08_public_sans_adaptation','12 références TSV figées ; une seule capture, validation de développement.\nRMSE sur les sections disponibles de chaque modèle : lire avec la couverture. Aucune nouvelle annotation ni entraînement.')
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    vals=[r['mae_px'] for r in fusion];axs[0].bar(['Ancien','Ego-path + TSV','Moyenne des deux'],vals,color=['#176b94','#bc7222','#64748b']);axs[0].set_ylim(0,max(vals)*1.2);axs[0].set_ylabel('Erreur absolue moyenne (px)')
    for i,v in enumerate(vals):axs[0].text(i,v,f'{v:.3f}',ha='center',va='bottom')
    rs=csvrows(OUT/'fusion_validation/AXES_MEMES_IMAGES.csv');pairs=[r for r in rs if r['ancien_tsv_error_px'] and r['ego_tsv_error_px']]
    axs[1].scatter([float(r['ancien_tsv_error_px']) for r in pairs],[float(r['ego_tsv_error_px']) for r in pairs],s=7,alpha=.35)
    axs[1].axhline(0,color='grey',lw=.8);axs[1].axvline(0,color='grey',lw=.8);axs[1].set(xlabel='Erreur signée ancien modèle (px)',ylabel='Erreur signée ego-path + TSV (px)',title=f'Corrélation des résidus : {corr:.2f}')
    fig.suptitle('Fusion naïve : une moyenne ne garantit pas un meilleur axe',fontsize=16)
    save(fig,'09_fusion_et_erreurs','Même ensemble de sections observées par les deux modèles ; aucune pondération ajustée.\nLes erreurs voisines d’une même image sont corrélées ; le désaccord entre modèles n’est pas l’erreur réelle.')
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for name,col,lab in [('capture_validation','#bc7222','30 images/s'),('autre_capture_60s','#176b94','5 images/s')]:
        rr=[r for r in replay if r['sequence']==name];axs[0].plot([r['distance_to_raw_mae_px'] for r in rr],[r['delta2_rms_px'] for r in rr],'o-',color=col,label=lab)
        for r in rr:axs[0].annotate(r['method'],(r['distance_to_raw_mae_px'],r['delta2_rms_px']),fontsize=8,xytext=(4,4),textcoords='offset points')
    axs[0].set(xlabel='Écart moyen au signal brut (px)',ylabel='Variation seconde RMS (px)',title='Vidéos : variation, pas précision');axs[0].legend()
    axs[1].bar([r['method'] for r in synthetic],[r['rmse_px'] for r in synthetic],color='#64748b');axs[1].set(ylabel='Erreur RMS par rapport au signal connu (px)',title='Test synthétique : retard et bruit');axs[1].tick_params(axis='x',rotation=20)
    fig.suptitle('Lisser davantage peut accroître l’erreur pendant les mouvements',fontsize=16)
    save(fig,'10_lissage_compromis','Rejeu expérimental du centre seul, paramètres explicités dans les CSV.\nLes deux captures ont des résolutions et cadences différentes ; pas de classement de précision entre elles.')

def main():
    assert (OUT/'evaluation_complete.json').exists()
    summaries,fusion,corr=compare_validation();print('Validation/fusion exported',flush=True)
    replay,synthetic=smoothing_tests();figures(summaries,fusion,corr,replay,synthetic)
    meta=series_exports()
    dump(OUT/'exports_complete.json',dict(complete=True,sequences=len(meta),frames=sum(m['frames'] for m in meta)))

if __name__=='__main__':main()
