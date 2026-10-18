"""Controlled local evaluation and videos for the RailSem19 ego-path experiment.

No training labels are created here. Pixels, timestamps, coverage and missing
predictions remain explicit. A lower frame-to-frame variation is not accuracy.
"""
from pathlib import Path
import argparse, csv, json, os, sys, subprocess, time, math
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'railtsv').is_dir())
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from railtsv.common import dump, sha256, read_manifest, rgb_read, mask_read, device
from railtsv.model import build_model, load_checkpoint
from railtsv.engine import predict_image
from railtsv.metrics import Metrics
from railtsv.axis_metrics import AxisMetrics
from railtsv.geometry import corridor_profile,draw_axis,draw_corridor_limits
from railtsv.adaptive_smoothing import AdaptiveProfileSmoother
from railtsv.video import verify_encoded
from scripts.diagnostic_plots import image_metrics, write_csv
import imageio_ffmpeg

BASE=ROOT/'runs/entrainement_couloir_20261009_143754_101cbf'
NAMES={'actuel':'Modèle actuel','egopath':'Avec ego-path'}
COLORS={'actuel':'#176b94','egopath':'#bc7222'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.color':'#e1e5e8','grid.linewidth':.6,
    'svg.fonttype':'none'})

def readcsv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def number(x):
    return None if x in ('',None) else float(x)

def errstats(values):
    a=np.asarray(values,float)
    if not len(a):return dict(n=0,mae_px=None,rmse_px=None,bias_px=None,p95_abs_px=None)
    return dict(n=len(a),mae_px=float(abs(a).mean()),rmse_px=float(np.sqrt((a*a).mean())),
        bias_px=float(a.mean()),p95_abs_px=float(np.percentile(abs(a),95)))

def evaluate_one(run,name,dev):
    out=run/'comparaison'/'validation'/name
    out.mkdir(parents=True,exist_ok=False)
    (out/'masques').mkdir();(out/'apercus').mkdir()
    checkpoint_path=(BASE if name=='actuel' else run)/'modele/best.pt'
    checkpoint=load_checkpoint(checkpoint_path)
    assert checkpoint['task']=='corridor' and not checkpoint['synthetic_demo']
    rows=read_manifest(run/'dataset/val.csv');assert len(rows)==12
    model=build_model(2);model.load_state_dict(checkpoint['state_dict']);model.to(dev).eval()
    del checkpoint['state_dict']
    metric=Metrics(['background','couloir']);axis=AxisMetrics();per_image=[];points=[]
    for i,r in enumerate(rows,1):
        rgb=rgb_read(r['image']);ref=mask_read(r['mask'],[0,1,255]);stem=Path(r['image']).stem
        pred,prob=predict_image(model,rgb,dev,checkpoint['height'],checkpoint['width'])
        assert pred.shape==ref.shape and np.isfinite(prob).all()
        metric.update(pred,ref);axis.update(pred,prob,ref,stem)
        m,pp,_=image_metrics(pred,ref,prob,stem,r['sequence']);per_image.append(m);points.extend(pp)
        Image.fromarray(pred).save(out/'masques'/f'{stem}.png')
        profile=corridor_profile(pred,prob)
        view=rgb.copy();positive=pred==1
        view[positive]=(view[positive]*.8+np.array([20,190,240])*.2).astype(np.uint8)
        view=draw_axis(draw_corridor_limits(view,profile),profile)
        Image.fromarray(view).resize((640,360)).save(out/'apercus'/f'{stem}.jpg',quality=94)
        print(f'Validation {name} {i}/12',flush=True)
    write_csv(out/'metriques_par_image.csv',per_image);write_csv(out/'erreurs_axe_sections.csv',points)
    result=dict(checkpoint=str(checkpoint_path),checkpoint_sha256=sha256(checkpoint_path),
        selected_epoch=checkpoint['epoch'],evaluation_device=str(dev),network_size=[checkpoint['width'],checkpoint['height']],
        images=len(rows),macro_dice=float(np.mean([p['dice'] for p in per_image])),
        macro_iou=float(np.mean([p['iou'] for p in per_image])),segmentation=metric.result(),axis=axis.result(),
        reference_manifest_sha256=sha256(run/'dataset/val.csv'),
        note='Original-resolution frozen references. Development validation used for model selection, not a final independent test.')
    dump(out/'resultats.json',result)
    del model
    if dev.type=='mps':torch.mps.empty_cache()
    return result

def savefig(fig,out,name,caption):
    fig.text(.02,.015,caption,fontsize=9,color='#45535d',ha='left')
    fig.tight_layout(rect=(0,.08,1,.98))
    fig.savefig(out/(name+'.png'),dpi=170,facecolor='white')
    fig.savefig(out/(name+'.pdf'),facecolor='white')
    plt.close(fig)

def compare_validation(run):
    root=run/'comparaison';out=root/'graphiques';out.mkdir(exist_ok=False)
    results={n:json.loads((root/'validation'/n/'resultats.json').read_text()) for n in NAMES}
    image_rows={n:readcsv(root/'validation'/n/'metriques_par_image.csv') for n in NAMES}
    section_rows={n:readcsv(root/'validation'/n/'erreurs_axe_sections.csv') for n in NAMES}
    indexed={n:{(r['image'],int(r['y_px'])):r for r in rs} for n,rs in section_rows.items()}
    keys=sorted(set(indexed['actuel'])&set(indexed['egopath']))
    common=[dict(image=k[0],y_px=k[1],**{f'{n}_erreur_axe_px':float(indexed[n][k]['center_error_px']) for n in NAMES}) for k in keys]
    write_csv(root/'sections_communes.csv',common)
    summary=[]
    for n,r in results.items():
        c=r['segmentation']['per_class']['couloir'];a=r['axis'];e=errstats([x[f'{n}_erreur_axe_px'] for x in common])
        summary.append(dict(modele=n,dice_macro=r['macro_dice'],iou_macro=r['macro_iou'],dice_micro=c['dice'],
            iou_micro=c['iou'],precision_micro=c['precision'],rappel_micro=c['recall'],
            couverture_pct=a['coverage_pct'],sections_reference=a['reference_points'],sections_detectees=a['matched_points'],
            axe_mae_px=a['mean_absolute_center_error_px'],axe_rmse_px=a['root_mean_square_center_error_px'],
            axe_biais_px=a['signed_center_bias_px'],axe_p95_px=a['p95_absolute_center_error_px'],
            sections_communes=e['n'],axe_mae_sections_communes_px=e['mae_px'],epoch_retenue=r['selected_epoch']))
    write_csv(root/'COMPARAISON_METRIQUES.csv',summary)
    paired=[]
    a={r['image']:r for r in image_rows['actuel']};b={r['image']:r for r in image_rows['egopath']}
    assert set(a)==set(b) and len(a)==12
    for i,key in enumerate(sorted(a),1):
        row=dict(numero=i,image=key)
        for metric in ['dice','iou','axis_mae_px','coverage_pct']:
            for n,d in [('actuel',a),('egopath',b)]:row[f'{metric}_{n}']=number(d[key][metric])
        row['delta_dice_points']=100*(row['dice_egopath']-row['dice_actuel'])
        paired.append(row)
    write_csv(root/'COMPARAISON_PAR_IMAGE.csv',paired)
    caption='12 images d’une seule capture ; validation de développement, utilisée pour sélectionner les modèles.'
    fig,axs=plt.subplots(1,3,figsize=(13.4,5.5))
    for ax,key,title,unit in zip(axs,['dice_macro','axe_mae_sections_communes_px','couverture_pct'],
        ['Recouvrement du couloir','Erreur de l’axe sur les mêmes sections','Couverture des sections annotées'],['Dice moyen (%)','Erreur absolue moyenne (px)','Sections détectées (%)']):
        vals=[r[key]*(100 if key=='dice_macro' else 1) if r[key] is not None else np.nan for r in summary]
        bars=ax.bar([0,1],vals,color=[COLORS[n] for n in NAMES],width=.58)
        ax.set_xticks([0,1],['Actuel','Ego-path']);ax.set_ylabel(unit);ax.set_title(title,fontsize=12)
        ax.set_ylim(0,105 if key!='axe_mae_sections_communes_px' else max([1]+[v for v in vals if np.isfinite(v)])*1.28)
        for bar,v in zip(bars,vals):ax.annotate(f'{v:.2f}' if np.isfinite(v) else 'Non calculable',(bar.get_x()+bar.get_width()/2,v if np.isfinite(v) else .1),xytext=(0,5),textcoords='offset points',ha='center')
    fig.suptitle('Apport du préapprentissage ego-path : comparaison à références identiques',fontsize=16)
    savefig(fig,out,'01_comparaison_globale',caption+'\nL’erreur en pixels ne mesure pas une distance physique. Couverture et erreur doivent se lire ensemble.')
    fig,axs=plt.subplots(3,1,figsize=(12,10),sharex=True)
    for n in NAMES:
        style=dict(color=COLORS[n],marker='o' if n=='actuel' else 's',linestyle='-' if n=='actuel' else '--',label=NAMES[n])
        for ax,key,scale in zip(axs,['dice','axis_mae_px','coverage_pct'],[100,1,1]):
            ax.plot(range(1,13),[p[f'{key}_{n}']*scale if p[f'{key}_{n}'] is not None else np.nan for p in paired],**style)
    for ax,title in zip(axs,['Dice du couloir (%)','Erreur absolue moyenne de l’axe (px)','Couverture des sections (%)']):
        ax.set_ylabel(title);ax.legend(loc='best');ax.set_ylim(bottom=0)
    for ax in [axs[0],axs[2]]:ax.set_ylim(0,105)
    axs[-1].set_xticks(range(1,13));axs[-1].set_xlabel('Image de validation, ordre chronologique (n = 12)')
    fig.suptitle('Résultats par image : repérer les progrès et les régressions',fontsize=16)
    savefig(fig,out,'02_resultats_par_image',caption+'\nMAE calculée sur les sections disponibles de chaque modèle ; les ensembles peuvent différer.')
    fig,axs=plt.subplots(1,2,figsize=(12.5,5.5))
    limit=max(2,2*math.ceil(max((abs(x[f'{n}_erreur_axe_px']) for x in common for n in NAMES),default=0)/2))
    bins=np.linspace(-limit,limit,61)
    for n in NAMES:
        errors=np.array([x[f'{n}_erreur_axe_px'] for x in common]);ordered=np.sort(abs(errors))
        axs[0].plot(ordered,100*np.arange(1,len(ordered)+1)/max(1,len(ordered)),color=COLORS[n],label=NAMES[n],linestyle='-' if n=='actuel' else '--')
        axs[1].hist(errors,bins=bins,histtype='step',color=COLORS[n],label=NAMES[n],linewidth=2,linestyle='-' if n=='actuel' else '--')
    axs[0].set(xlabel='Erreur absolue de l’axe (px)',ylabel='Sections communes cumulées (%)',xlim=(0,None),ylim=(0,101))
    axs[1].set(xlabel='Erreur signée : prédiction − annotation (px)',ylabel='Nombre de sections');axs[1].axvline(0,color='#454b50',linestyle=':')
    for ax in axs:ax.legend()
    fig.suptitle('Distribution des erreurs sur les sections communes',fontsize=16)
    savefig(fig,out,'03_erreurs_sections_communes',f'{len(common)} sections appariées dans 12 images ; les sections ne sont pas des observations indépendantes.\nAucun intervalle de confiance ni pourcentage de précision physique n’est déduit de ce graphique.')
    fig,axs=plt.subplots(1,2,figsize=(12.5,5.5))
    for n in NAMES:
        h=json.loads(((BASE if n=='actuel' else run)/'modele/history.json').read_text())
        x=[r['epoch'] for r in h]
        axs[0].plot(x,[r['train_loss'] for r in h],color=COLORS[n],label=NAMES[n]+' / apprentissage')
        axs[0].plot(x,[r['val_loss'] for r in h],color=COLORS[n],linestyle='--',label=NAMES[n]+' / validation')
        axs[1].plot(x,[r['metrics']['foreground_dice']*100 for r in h],color=COLORS[n],label=NAMES[n],marker='.' if n=='actuel' else 'x')
    axs[0].set(xlabel='Passage sur les 130 images TSV',ylabel='Loss Dice + entropie croisée',ylim=(0,None));axs[0].legend(fontsize=8)
    axs[1].set(xlabel='Passage sur les 130 images TSV',ylabel='Dice de validation à l’entrée du réseau (%)');axs[1].legend()
    for ax in axs:ax.set_xticks([1,5,10,15,20])
    fig.suptitle('Adaptation TSV : mêmes données et budget de 20 passages',fontsize=16)
    savefig(fig,out,'04_apprentissage_tsv','Courbes à 640 × 480 avec marges ignorées ; distinctes des métriques à résolution originale.\nDeux passages supplémentaires sur RailSem19 ego-path précèdent uniquement la nouvelle version.')
    eh=json.loads((run/'preapprentissage_egopath/history.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(11.5,5.5))
    axs[0].plot([r['epoch'] for r in eh],[r['train_loss'] for r in eh],'-o',color=COLORS['actuel'],label='Apprentissage')
    axs[0].plot([r['epoch'] for r in eh],[r['val_loss'] for r in eh],'--s',color=COLORS['egopath'],label='Validation RailSem19')
    axs[0].set(ylabel='Loss Dice + entropie croisée',ylim=(0,None));axs[0].legend()
    axs[1].plot([r['epoch'] for r in eh],[r['metrics']['foreground_dice']*100 for r in eh],'-o',color=COLORS['egopath'])
    axs[1].set(ylabel='Dice ego-path (%)',ylim=(0,100))
    for ax in axs:ax.set(xlabel='Passage sur les 2 796 images ego-path',xticks=[r['epoch'] for r in eh])
    fig.suptitle('Préapprentissage du chemin suivi par le train',fontsize=16)
    savefig(fig,out,'05_apprentissage_egopath','2 796 images d’apprentissage, 466 de validation publique ; entrée 640 × 384.\nCible publique ego-path : pas une mesure certifiée des faces intérieures des rails TSV.')
    font=ImageFont.truetype(findfont('DejaVu Sans'),22)
    for page in range(2):
        canvas=Image.new('RGB',(1920,6*405+75),'white');draw=ImageDraw.Draw(canvas)
        draw.text((18,12),'12 images de validation — références inchangées, comparaison de développement',font=font,fill='#17232d')
        for rowindex,r in enumerate(read_manifest(run/'dataset/val.csv')[page*6:(page+1)*6]):
            stem=Path(r['image']).stem;rgb=rgb_read(r['image']);ref=mask_read(r['mask'],[0,1,255])
            refview=rgb.copy();refview[ref==1]=(refview[ref==1]*.72+np.array([20,190,240])*.28).astype(np.uint8)
            refview[ref==255]=(refview[ref==255]*.65+np.array([230,140,170])*.35).astype(np.uint8)
            views=[Image.fromarray(refview).resize((640,360))]+[Image.open(root/'validation'/n/'apercus'/f'{stem}.jpg') for n in NAMES]
            y=65+rowindex*405
            for col,(view,title) in enumerate(zip(views,['Annotation (rose = ignore)','Modèle actuel','Modèle avec ego-path'])):
                draw.text((col*640+8,y),f'{page*6+rowindex+1:02} | {title}',font=font,fill='#17232d');canvas.paste(view,(col*640,y+30))
        canvas.save(root/f'PLANCHE_VALIDATION_{page+1}.jpg',quality=94)
    dump(root/'RESULTATS.json',dict(models=results,comparison=summary,common_sections=len(common),
        paired_images=12,validation_capture_count=1,final_test=False,
        interpretation='Compare segmentation, axis error and coverage jointly; no claim of generalization or physical gauge accuracy.'))
    return summary

def model_section(row):
    return dict(status=row['status'],reason=row['reason'],center=number(row['center_x_px']),width=number(row['apparent_width_px']),
        score=number(row.get('corridor_model_score')))

def render_pair(old_folder,new_folder,output):
    """Replay equal source frames and raw geometry with identical adaptive filters."""
    if output.exists():raise FileExistsError(output)
    old=json.loads((old_folder/'video.json').read_text());new=json.loads((new_folder/'video.json').read_text())
    for k in ['source_sha256','fps','dimensions']:
        assert old[k]==new[k], f'Comparison mismatch: {k}'
    assert old.get('inference_roi_xyxy_exclusive')==new.get('inference_roi_xyxy_exclusive')
    fps=new['fps'];w,h=new['dimensions'];n=new['frames'];assert old['frames']>=n
    preview_times=[10,60,140,165] if n/fps>=170 else [10,30,50]
    preview_indices={round(t*fps) for t in preview_times if t<n/fps}
    filtered={k:AdaptiveProfileSmoother(w,max_gap_seconds=max(.15,2.5/fps)) for k in NAMES}
    readers={k:csv.DictReader((folder/'mesures.csv').open(newline='')) for k,folder in [('actuel',old_folder),('egopath',new_folder)]}
    ff=imageio_ffmpeg.get_ffmpeg_exe();geometry_rows=[];series=[]
    font=ImageFont.truetype(findfont('DejaVu Sans'),21);small=ImageFont.truetype(findfont('DejaVu Sans'),17)
    with output.with_suffix('.log').open('wb') as log:
        decoder=subprocess.Popen([ff,'-hide_banner','-loglevel','error','-nostdin','-i',new['source'],
            '-an','-vf',f'fps=fps={fps:.12g}:start_time=0:round=near','-frames:v',str(n),'-fps_mode','passthrough',
            '-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=log)
        encoder=subprocess.Popen([ff,'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
            '-s','1600x570','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','20',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE,stderr=log)
        count=0
        try:
            for i in range(n):
                raw=decoder.stdout.read(w*h*3)
                if len(raw)!=w*h*3:raise ValueError(f'Incomplete source frame {i}')
                rgb=np.frombuffer(raw,np.uint8).reshape(h,w,3);canvas=Image.new('RGB',(1600,570),'#111d2b');draw=ImageDraw.Draw(canvas)
                rows={k:next(readers[k]) for k in NAMES}
                for key in ['frame_index','source_frame_index','seconds','source_seconds','repeated_source_frame']:
                    assert rows['actuel'][key]==rows['egopath'][key],f'Frame alignment failed: {i}, {key}'
                repeated=rows['actuel']['repeated_source_frame']=='True'
                joined=dict(frame_index=i,seconds=float(rows['actuel']['seconds']),source_seconds=float(rows['actuel']['source_seconds']),repeated=repeated)
                for panel,key in enumerate(NAMES):
                    r=rows[key];segments=json.loads(r['axis_points_json']);profile=dict(segments=segments,points=sum(map(len,segments)))
                    smooth=filtered[key].last if repeated else filtered[key].update(profile,float(r['source_seconds']))
                    smooth=smooth or dict(segments=[])
                    view=rgb.copy()
                    overlay=view.copy()
                    for seg in smooth['segments']:
                        poly=np.asarray([[p['left_inner_x'],p['y']] for p in seg]+[[p['right_inner_x'],p['y']] for p in seg[::-1]],np.int32)
                        cv2.fillPoly(overlay,[poly],(20,190,240))
                    view=cv2.addWeighted(view,.83,overlay,.17,0)
                    view=draw_axis(draw_corridor_limits(view,smooth),smooth)
                    # Both panels have the same spatial transformation, source frame and smoothing.
                    canvas.paste(Image.fromarray(view).resize((800,450)),(panel*800,46))
                    draw.text((panel*800+18,12),NAMES[key],font=font,fill='white')
                    section=model_section(r)
                    text=(f"Largeur brute : {section['width']:.1f} px | axe x : {section['center']:.1f} px" if section['width'] is not None else 'Section à 80 % : mesure indisponible')
                    draw.text((panel*800+14,505),text,font=small,fill='white')
                    joined.update({f'{key}_{k}':section[k] for k in ['status','reason','center','width','score']})
                    # A fixed image row for like-for-like stability, no extrapolation.
                    y=8*round(.8*h/8);point=next((p for s in smooth['segments'] for p in s if p['y']==y),None)
                    joined[f'{key}_profile_y']=y;joined[f'{key}_profile_raw_x']=point['raw_center_x'] if point and not repeated else None
                    joined[f'{key}_profile_smooth_x']=point['center_x'] if point and not repeated else None
                    if i in preview_indices:
                        geometry_rows.append(dict(frame_index=i,model=key,segments=smooth['segments']))
                draw.text((14,541),f"t = {i/fps:.2f} s | pixels : source {w} × {h} | même lissage | vert : axe médian | mètres non calibrés"+(' | image répétée' if repeated else ''),font=small,fill='#d4e4ef')
                encoder.stdin.write(np.asarray(canvas).tobytes());series.append(joined);count+=1
                if i in preview_indices:canvas.save(output.with_name(output.stem+f'_{int(i/fps):02}s.jpg'),quality=94)
                if count%max(1,round(fps*20))==0:print(f'Comparaison vidéo {output.name}: {count}/{n}',flush=True)
            encoder.stdin.close();decoder.stdout.close()
            if encoder.wait(timeout=60)!=0 or decoder.wait(timeout=60)!=0:raise RuntimeError('Video encoding/decoding failed')
        finally:
            for p in [encoder,decoder]:
                if p.poll() is None:p.terminate();p.wait(timeout=15)
    verify_encoded(output,count,fps,[1600,570])
    write_csv(output.with_suffix('.csv'),series)
    dump(output.with_suffix('.json'),dict(old_checkpoint_sha256=old['checkpoint_sha256'],new_checkpoint_sha256=new['checkpoint_sha256'],
        source_sha256=new['source_sha256'],frames=count,fps=fps,duration_seconds=count/fps,
        comparison='same source frames, same raw geometry extraction and paired adaptive display filter',
        metric_calibration=None,accuracy_ground_truth_available=False,
        units='Original source image pixels, not the 800-pixel display panels',
        source_dimensions=[w,h],roi=new.get('inference_roi_xyxy_exclusive'),repeats=sum(s['repeated'] for s in series),
        preview_times_seconds=preview_times))
    return series

def video_comparisons(run):
    out=run/'comparaison';out.mkdir(exist_ok=True)
    old_novel=ROOT/'03_RESULTATS/AMELIORATIONS_20261014/video_nouvelle_capture_60s'
    cases=[('capture_validation',BASE/'video',0),('autre_capture_60s',old_novel,300)]
    for label,oldfolder,maxframes in cases:
        meta=json.loads((oldfolder/'video.json').read_text());newfolder=run/('video_'+label)
        cmd=[str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/infer_video.py'),'--input',meta['source'],
            '--checkpoint',str(run/'modele/best.pt'),'--out',str(newfolder),'--device','mps','--max-frames',str(maxframes),
            '--smooth-display','--smoothing-method','adaptive']
        if meta.get('inference_roi_xyxy_exclusive'):cmd+=['--roi',*map(str,meta['inference_roi_xyxy_exclusive'])]
        dump(out/f'commande_video_{label}.json',cmd)
        if not (newfolder/'video.json').exists():
            with (out/f'inference_{label}.log').open('x') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        series=render_pair(oldfolder,newfolder,out/f'COMPARAISON_{label.upper()}.mp4')
        compare_stability(series,out,label,meta['fps'])

def compare_stability(series,out,label,fps):
    """Compare only common consecutive observations; motion is not labelled error."""
    fields=[f'{n}_profile_{kind}_x' for n in NAMES for kind in ['raw','smooth']]
    def valid(row):return not row['repeated'] and all(row[k] is not None for k in fields)
    pairs=[(a,b) for a,b in zip(series,series[1:]) if valid(a) and valid(b)
        and b['frame_index']==a['frame_index']+1 and b['source_seconds']-a['source_seconds']<=1.5/fps]
    triples=[(a,b,c) for a,b,c in zip(series,series[1:],series[2:]) if all(valid(r) for r in [a,b,c])
        and c['frame_index']==b['frame_index']+1==a['frame_index']+2
        and max(b['source_seconds']-a['source_seconds'],c['source_seconds']-b['source_seconds'])<=1.5/fps]
    stats=[]
    for name in NAMES:
        for kind in ['raw','smooth']:
            key=f'{name}_profile_{kind}_x';delta=np.array([b[key]-a[key] for a,b in pairs]);acc=np.array([c[key]-2*b[key]+a[key] for a,b,c in triples])
            stats.append(dict(model=name,mode=kind,pairs=len(pairs),triples=len(triples),
                first_difference_rms_px=float(np.sqrt(np.mean(delta**2))) if len(delta) else None,
                second_difference_rms_px=float(np.sqrt(np.mean(acc**2))) if len(acc) else None))
    write_csv(out/f'STABILITE_{label}.csv',stats)
    dump(out/f'STABILITE_{label}.json',dict(metrics=stats,source_fps=fps,
        y_px=series[0]['actuel_profile_y'],common_observations=sum(valid(r) for r in series),
        missing_or_repeated_observations=len(series)-sum(valid(r) for r in series),
        note='Common consecutive fixed-row observations only. Variation includes actual track/camera motion; lower variation is not greater accuracy. No temporal ground truth.'))
    seconds=[]
    for sec in range(math.ceil(len(series)/fps)):
        rows=[r for r in series if sec<=r['seconds']<sec+1];observed=[r for r in rows if not r['repeated']]
        common=[r for r in rows if valid(r)]
        summary=dict(second=sec,frames=len(rows),observations=len(observed),common_profile_observations=len(common))
        for n in NAMES:
            summary[n+'_candidate_frames']=sum(r[n+'_status']=='candidate' for r in observed)
            summary[n+'_axis_raw_mean_px_common']=float(np.mean([r[n+'_profile_raw_x'] for r in common])) if common else None
            summary[n+'_axis_smooth_mean_px_common']=float(np.mean([r[n+'_profile_smooth_x'] for r in common])) if common else None
        seconds.append(summary)
    write_csv(out/f'COMPARAISON_PAR_SECONDE_{label}.csv',seconds)
    fig,axs=plt.subplots(2,1,figsize=(12,7.8),sharex=True)
    for ax,n in zip(axs,NAMES):
        t=[r['seconds'] for r in series]
        raw=[r[n+'_profile_raw_x'] if not r['repeated'] and r[n+'_profile_raw_x'] is not None else np.nan for r in series]
        smooth=[r[n+'_profile_smooth_x'] if not r['repeated'] and r[n+'_profile_smooth_x'] is not None else np.nan for r in series]
        ax.plot(t,raw,color='#929aa1',linewidth=.8,label='Axe brut')
        ax.plot(t,smooth,color=COLORS[n],linewidth=1.1,label='Même filtre adaptatif')
        ax.set(title=NAMES[n],ylabel=f"Axe x à y = {series[0]['actuel_profile_y']} (px)");ax.legend(loc='best')
    finite=[r[f'{n}_profile_{kind}_x'] for r in series for n in NAMES for kind in ['raw','smooth'] if r[f'{n}_profile_{kind}_x'] is not None]
    if finite:
        lo,hi=min(finite),max(finite);margin=max(2,.05*(hi-lo))
        for ax in axs:ax.set_ylim(lo-margin,hi+margin)
    axs[-1].set_xlabel('Temps vidéo (s)');axs[-1].set_xlim(0,(len(series)-1)/fps)
    fig.suptitle('Stabilité de l’axe — '+('capture de validation' if label=='capture_validation' else 'autre capture, 60 secondes'),fontsize=16)
    savefig(fig,out/'graphiques',f'06_stabilite_{label}',f'{len(pairs)} paires consécutives communes pour les statistiques ; {fps:g} images/s ; coordonnées de l’image source.\nLe mouvement réel contribue à la variation. Une courbe plus lisse ne prouve pas une meilleure précision.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--phase',choices=['baseline','new','plots','videos','all'],default='all')
    p.add_argument('--device',default='auto');a=p.parse_args();run=a.run.resolve()
    torch.set_num_threads(4);dev=device(a.device)
    if a.phase in ('baseline','all'):evaluate_one(run,'actuel',dev)
    if a.phase in ('new','all'):evaluate_one(run,'egopath',dev)
    if a.phase in ('plots','all'):compare_validation(run)
    if a.phase in ('videos','all'):video_comparisons(run)

if __name__=='__main__':main()
