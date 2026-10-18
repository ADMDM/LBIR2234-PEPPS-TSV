"""Scientific figures for corridor diagnostics; training and validation kept explicit."""
import argparse, csv, json, os, time
from datetime import datetime
from pathlib import Path
import sys, uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from railtsv.common import device, dump, new_dir, read_manifest, rgb_read, mask_read, sha256
from railtsv.model import build_model,load_checkpoint
from railtsv.engine import predict_image
from railtsv.geometry import corridor_profile
from railtsv.metrics import Metrics
from railtsv.data import letterbox
from railtsv.annotation_config import active_validation
from scripts.export_labelme import rasterize

COLORS=['#176b94','#bc7222','#815f8b','#637b36']
GROUPS=['video_03383','nordland_ete','photos_20220803']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':14,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
    'grid.color':'#dfe5e8','grid.linewidth':.65,'svg.fonttype':'none'})


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def compare_profiles(pred,prob,ref,image_id,group):
    refprob=np.stack([(ref!=1).astype(np.float32),(ref==1).astype(np.float32)])
    rp=corridor_profile(ref,refprob);pp=corridor_profile(pred,prob)
    targets={p['y']:p for segment in rp['segments'] for p in segment
             if not np.any(ref[p['y'],max(0,p['left_inner_x']-1):p['right_inner_x']+2]==255)}
    estimates={p['y']:p for segment in pp['segments'] for p in segment}
    rows=[];h,w=ref.shape
    for y,r in targets.items():
        if y not in estimates:continue
        p=estimates[y];center=p['center_x']-r['center_x']
        rows.append({'image':image_id,'group':group,'width_px':w,'height_px':h,'y_px':y,'y_relative':y/h,
            'center_ref_px':r['center_x'],'center_pred_px':p['center_x'],'center_error_px':center,
            'center_error_pct_width':100*center/w,
            'left_error_px':p['left_inner_x']-r['left_inner_x'],
            'right_error_px':p['right_inner_x']-r['right_inner_x'],
            'width_ref_px':r['apparent_width_px'],'width_pred_px':p['apparent_width_px'],
            'width_error_px':p['apparent_width_px']-r['apparent_width_px']})
    return rows,len(targets)


def image_metrics(pred,ref,prob,stem,group):
    metric=Metrics(['background','couloir']);metric.update(pred,ref);m=metric.result()
    tn,fp,fn,tp=np.asarray(m['confusion']).ravel()
    points,nref=compare_profiles(pred,prob,ref,stem,group)
    errors=np.array([p['center_error_px'] for p in points],float)
    row={'image':stem,'group':group,'width_px':ref.shape[1],'height_px':ref.shape[0],
         'dice':m['per_class']['couloir']['dice'],'iou':m['per_class']['couloir']['iou'],
         'precision':float(tp/(tp+fp)) if tp+fp else None,
         'recall':float(tp/(tp+fn)) if tp+fn else None,
         'reference_points':nref,'matched_points':len(points),
         'coverage_pct':100*len(points)/nref if nref else None,
         'axis_mae_px':float(np.abs(errors).mean()) if len(errors) else None,
         'axis_mae_pct_width':float(100*np.abs(errors).mean()/ref.shape[1]) if len(errors) else None}
    return row,points,m['confusion']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--scope',choices=['training','validation'],default='training')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();run=args.run.resolve();checkpoint_path=run/'modele/last.pt'
    if not checkpoint_path.exists():checkpoint_path=run/'modele/best.pt'
    checkpoint=load_checkpoint(checkpoint_path)
    if checkpoint['task']!='corridor' or checkpoint['synthetic_demo']:raise ValueError('Modèle couloir réel requis')
    if args.scope=='training':rows=read_manifest(run/'dataset/train.csv')
    elif (run/'dataset/val.csv').is_file():
        rows=read_manifest(run/'dataset/val.csv')
        for row in rows:
            annotation=run/'dataset/json'/f'{Path(row["image"]).stem}.json'
            data=json.loads(annotation.read_text())
            with Image.open(row['image']) as im:ref=rasterize(data,im.size,task='corridor')
            assert np.array_equal(ref,mask_read(row['mask'],[0,1,255])), 'Frozen validation mask differs from annotation'
            row['annotation']=str(annotation)
        if len(rows)!=12:raise ValueError('Les 12 références de validation sont requises')
    else:
        session=active_validation(ROOT);rows=[]
        for image in sorted((session/'images').glob('*.png')):
            path=session/'json'/f'{image.stem}.json'
            if not path.exists():raise ValueError('Les 12 images de validation doivent être annotées et reviewed avant les graphiques de validation')
            data=json.loads(path.read_text())
            with Image.open(image) as im:rasterize(data,im.size,task='corridor')
            rows.append({'image':str(image),'annotation':str(path),'sequence':'video_03676'})
        if len(rows)!=12:raise ValueError('Les 12 références de validation sont requises')
    suffix=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    out=new_dir(args.out or ROOT/'runs'/('figures_modele_'+suffix));figdir=out/'figures';figdir.mkdir()
    if args.scope=='validation':
        (out/'masks').mkdir()
        for row in rows:
            data=json.loads(Path(row['annotation']).read_text())
            with Image.open(row['image']) as im:ref=rasterize(data,im.size,task='corridor')
            dest=out/'masks'/f'{Path(row["image"]).stem}.png';Image.fromarray(ref).save(dest);row['mask']=str(dest)
    names={x['image']:x.get('sequence','UNKNOWN') for x in rows}
    torch.set_num_threads(4);dev=device('auto');model=build_model(2);model.load_state_dict(checkpoint['state_dict']);model.to(dev).eval()
    per_image=[];all_points=[];confusion=np.zeros((2,2),np.int64)
    samples_p=[];samples_t=[];rng=np.random.default_rng(42)
    example=None
    for i,row in enumerate(rows,1):
        image=rgb_read(row['image']);ref=mask_read(row['mask'],[0,1,255]);stem=Path(row['image']).stem
        pred,prob=predict_image(model,image,dev,checkpoint['height'],checkpoint['width'])
        metrics,points,c=image_metrics(pred,ref,prob,stem,row['sequence']);per_image.append(metrics);all_points.extend(points);confusion+=np.asarray(c)
        valid=np.flatnonzero(ref.ravel()!=255);selected=rng.choice(valid,min(20000,len(valid)),replace=False)
        samples_p.append(prob[1].ravel()[selected].copy());samples_t.append(ref.ravel()[selected].copy())
        if example is None and row['sequence'].startswith('video_'):
            entropy=-(prob*np.log2(np.clip(prob,1e-9,1))).sum(0)
            example=(image.copy(),entropy.copy(),stem)
        print(f'Diagnostic {args.scope} : {i}/{len(rows)}',flush=True)
        del image,ref,pred,prob
    write_csv(out/'metriques_par_image.csv',per_image);write_csv(out/'erreurs_axe_sections.csv',all_points)
    sample_p=np.concatenate(samples_p);sample_t=np.concatenate(samples_t)
    np.savez_compressed(out/'echantillon_probabilites.npz',p=sample_p,target=sample_t)
    # Compare three network sizes on the same deterministic group-stratified images.
    chosen=[]
    for group,n in [('video_03383',6),('nordland_ete',3),('photos_20220803',3)]:
        members=[row for row in rows if row['sequence']==group]
        if members:chosen.extend(members[int(i)] for i in np.linspace(0,len(members)-1,min(n,len(members))))
    if not chosen:chosen=rows[:12]
    sensitivity=[];(out/'entrees_vga').mkdir()
    for height,width in [(256,320),(480,640),(960,1280)]:
        with torch.inference_mode():model(torch.zeros(1,3,height,width,device=dev))
        if dev.type=='mps':torch.mps.synchronize()
        for row in chosen:
            image=rgb_read(row['image']);ref=mask_read(row['mask'],[0,1,255]);stem=Path(row['image']).stem
            start=time.perf_counter();pred,prob=predict_image(model,image,dev,height,width);elapsed=time.perf_counter()-start
            m,_,_=image_metrics(pred,ref,prob,stem,row['sequence'])
            sensitivity.append({'image':stem,'group':row['sequence'],'network_width':width,'network_height':height,
                                'dice':m['dice'],'iou':m['iou'],'axis_mae_pct_width':m['axis_mae_pct_width'],
                                'coverage_pct':m['coverage_pct'],'inference_seconds':elapsed})
            if width==640:
                vga,_,_=letterbox(image,480,640,ref);Image.fromarray(vga).save(out/'entrees_vga'/f'{stem}.png')
                # Explicit smoke inference on a real 640x480 input; no original GT is fabricated.
                vm,vp=predict_image(model,vga,dev,480,640);assert vm.shape==(480,640) and np.isfinite(vp).all()
            del image,ref,pred,prob
        print(f'Comparaison de résolution terminée : {width}×{height}',flush=True)
    write_csv(out/'sensibilite_resolution.csv',sensitivity)
    scope_note=(f'Diagnostic sur {len(rows)} images utilisées pour apprendre — pas une mesure de généralisation' if args.scope=='training'
                else 'Validation TSV : 12 images de la seconde capture — pas un test final multi-captures')
    entries=[]
    def figure(number,title,caption,scope=None,size=(9,5.4)):
        fig,ax=plt.subplots(figsize=size);fig.suptitle(title,x=.08,ha='left',fontsize=15,fontweight='bold')
        fig.text(.08,.925,scope or scope_note,fontsize=9,color='#5c6870');fig.subplots_adjust(left=.10,right=.96,bottom=.14,top=.83)
        def finish():
            name=f'{number:02d}';fig.savefig(figdir/f'{name}.png',dpi=170);fig.savefig(figdir/f'{name}.svg');plt.close(fig)
            entries.append({'id':name,'title':title,'caption':caption,'scope':scope or scope_note})
        return fig,ax,finish
    def col(key):return np.array([p[key] if p[key] is not None else np.nan for p in per_image],float)
    fig,ax,done=figure(1,'Les images du lot analysé','Les groupes sont des origines de capture ; les lots ne sont pas des captures indépendantes.')
    groups=sorted({x['group'] for x in per_image});counts=[sum(x['group']==g for x in per_image) for g in groups]
    fig.subplots_adjust(left=.30)
    display={'nordland_ete':'Nordland','photos_20220803':'Photos complémentaires','video_03383':'Frames TSV'}
    ax.barh([display.get(g,g) for g in groups],counts,color=COLORS[0]);ax.set_xlabel('Nombre d’images');ax.set_xlim(0,max(counts)*1.15)
    for i,n in enumerate(counts):ax.text(n+.2,i,str(n),va='center');
    done()
    history=json.loads((run/'modele/history.json').read_text())
    fig,ax,done=figure(2,'Apprentissage TSV : évolution de la loss','Dice + CrossEntropy. La loss d’apprentissage ne donne pas la précision sur une autre vidéo.',scope='Historique de l’apprentissage du modèle TSV, issu du run sélectionné')
    ax.plot([x['epoch'] for x in history],[x['train_loss'] for x in history],'-o',color=COLORS[0],label='Apprentissage')
    if all(x.get('val_loss') is not None for x in history):ax.plot([x['epoch'] for x in history],[x['val_loss'] for x in history],'--s',color=COLORS[1],label='Validation')
    ax.legend();ax.set(xlabel='Passage',ylabel='Loss moyenne',ylim=(0,None));ax.set_xticks([x['epoch'] for x in history]);done()
    init=json.loads((run/'initialisation.json').read_text());railrun=Path(init['checkpoint']).parent.parent;rh=json.loads((railrun/'modele/history.json').read_text())
    fig,ax,done=figure(3,'RailSem19 : apprentissage et validation','Deux passages terminés. Le troisième a été interrompu.',scope='RailSem19 : 3000 images train / 500 validation, séparation de l’article')
    ax.plot([x['epoch'] for x in rh],[x['train_loss'] for x in rh],'-o',color=COLORS[0],label='Apprentissage');ax.plot([x['epoch'] for x in rh],[x['val_loss'] for x in rh],'--s',color=COLORS[1],label='Validation');ax.set(xlabel='Passage terminé',ylabel='Loss',ylim=(0,None));ax.set_xticks([x['epoch'] for x in rh]);ax.legend();done()
    fig,ax,done=figure(4,'RailSem19 : Dice et IoU du domaine ferroviaire','Ces scores ne sont pas ceux des vidéos TSV.',scope='RailSem19 : 500 images de validation, zone de voie générique')
    x=np.arange(len(rh));ax.bar(x-.18,[100*r['metrics']['per_class']['track_area']['dice'] for r in rh],.36,color=COLORS[0],label='Dice');ax.bar(x+.18,[100*r['metrics']['per_class']['track_area']['iou'] for r in rh],.36,color=COLORS[1],label='IoU');ax.set(xticks=x,xticklabels=[str(r['epoch']) for r in rh],xlabel='Passage terminé',ylabel='Score (%)',ylim=(0,100));ax.legend();done()
    fig,ax,done=figure(5,'Matrice de confusion pixel par pixel','Les pixels ignore sont exclus ; les lignes sont normalisées par la classe de référence.')
    norm=confusion/confusion.sum(1,keepdims=True);ax.imshow(norm,cmap='Blues',vmin=0,vmax=1);ax.grid(False);ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Fond','Couloir'],yticklabels=['Fond','Couloir'],xlabel='Prédiction',ylabel='Annotation')
    for i in range(2):
        for j in range(2):ax.text(j,i,f'{100*norm[i,j]:.2f}%\n{confusion[i,j]:,} pixels',ha='center',va='center',color='white' if norm[i,j]>.6 else '#152634')
    done()
    fig,ax,done=figure(6,'Distribution des scores par image','Chaque image compte une fois. Des scores élevés sur l’apprentissage peuvent refléter une mémorisation.')
    ax.hist(100*col('dice'),bins=np.linspace(0,100,21),alpha=.65,color=COLORS[0],label='Dice');ax.hist(100*col('iou'),bins=np.linspace(0,100,21),histtype='step',linewidth=2,color=COLORS[1],label='IoU');ax.set(xlabel='Score (%)',ylabel='Nombre d’images',xlim=(0,100));ax.legend();done()
    fig,ax,done=figure(7,'Dice selon l’origine des images','Les boîtes décrivent la dispersion observée ; elles ne sont pas des intervalles de confiance.')
    ax.boxplot([[100*x['dice'] for x in per_image if x['group']==g and x['dice'] is not None] for g in groups],tick_labels=groups);ax.set(ylabel='Dice (%)',ylim=(0,100));done()
    fig,ax,done=figure(8,'Erreur moyenne de l’axe, image par image','Erreur normalisée par la largeur pour comparer les résolutions originales différentes.',size=(10,12))
    ranked=sorted([x for x in per_image if x['axis_mae_pct_width'] is not None],key=lambda x:x['axis_mae_pct_width'])
    write_csv(out/'classement_erreur_axe.csv',[dict(rank=i+1,**x) for i,x in enumerate(ranked)])
    aliases=[f'{i+1:02d}' for i in range(len(ranked))];ax.barh(aliases,[x['axis_mae_pct_width'] for x in ranked],color=COLORS[0]);ax.set(xlabel='MAE de l’axe (% de la largeur de l’image)',ylabel='Image (identifiants détaillés dans le CSV)');done()
    errors=np.array([x['center_error_pct_width'] for x in all_points]);abs_err=np.abs(errors)
    fig,ax,done=figure(9,'Distribution cumulée des écarts de l’axe','Sections appariées seulement. Le 95e percentile empirique n’est pas un intervalle de confiance.')
    z=np.sort(abs_err);ax.plot(z,np.arange(1,len(z)+1)/len(z),color=COLORS[0]);q=np.percentile(z,95);ax.axvline(q,color='#404a53',linestyle='--',label=f'P95 observé : {q:.3f}%');ax.set(xlabel='Écart absolu (% de largeur)',ylabel='Part des sections appariées',ylim=(0,1));ax.legend();done()
    fig,ax,done=figure(10,'Erreur signée de l’axe selon la hauteur','La hauteur relative de l’image ne représente pas une distance en mètres.')
    bins=np.linspace(0,1,6);values=[[x['center_error_pct_width'] for x in all_points if a<=x['y_relative']<b] for a,b in zip(bins,bins[1:])]
    ax.boxplot([v or [np.nan] for v in values],tick_labels=[f'{a:.1f}–{b:.1f}' for a,b in zip(bins,bins[1:])],showfliers=False);ax.axhline(0,color='#404a53');ax.set(xlabel='Hauteur y / hauteur image (0 = haut)',ylabel='Écart signé (% de largeur)');done()
    fig,ax,done=figure(11,'Largeur prédite et largeur du couloir annoté','Largeur horizontale apparente, normalisée par l’image ; ce n’est pas l’écartement physique.')
    sample=all_points[::max(1,len(all_points)//5000)];xx=[100*x['width_ref_px']/x['width_px'] for x in sample];yy=[100*x['width_pred_px']/x['width_px'] for x in sample];lim=max(xx+yy)
    ax.scatter(xx,yy,s=8,alpha=.25,color=COLORS[0]);ax.plot([0,lim],[0,lim],'--',color='#404a53');ax.set(xlabel='Largeur annotée (% image)',ylabel='Largeur prédite (% image)',xlim=(0,lim),ylim=(0,lim));done()
    fig,ax,done=figure(12,'Écarts des deux limites et du milieu','Dispersion des sections appariées ; les erreurs des limites peuvent être corrélées.')
    ax.boxplot([[100*x[k]/x['width_px'] for x in all_points] for k in ['left_error_px','right_error_px','center_error_px']],tick_labels=['Limite gauche','Limite droite','Axe'],showfliers=False);ax.axhline(0,color='#404a53');ax.set_ylabel('Écart signé (% de largeur)');done()
    fig,ax,done=figure(13,'Couverture des références exploitables','Une faible erreur sur les seules portions détectées ne suffit pas : les portions manquantes comptent dans la couverture.')
    ax.hist(col('coverage_pct'),bins=np.linspace(0,100,21),color=COLORS[0]);ax.set(xlabel='Sections annotées ayant une prédiction (%)',ylabel='Nombre d’images',xlim=(0,100));done()
    predicted=(sample_p>=.5).astype(np.uint8);confidence=np.maximum(sample_p,1-sample_p);correct=(predicted==sample_t)
    reliability=[]
    for a,b in zip(np.linspace(.5,1,11)[:-1],np.linspace(.5,1,11)[1:]):
        sel=(confidence>=a)&(confidence<=b if b==1 else confidence<b)
        if sel.any():reliability.append({'bin_start':float(a),'bin_end':float(b),'count':int(sel.sum()),'confidence':float(confidence[sel].mean()),'correct_fraction':float(correct[sel].mean())})
    write_csv(out/'diagramme_fiabilite.csv',reliability)
    fig,ax,done=figure(14,'Score interne et concordance aux annotations','Échantillon uniforme de 20000 pixels valides par image ; fond inclus. Diagnostic, pas calibration garantie.')
    ax.plot([50,100],[50,100],'--',color='#404a53',label='Concordance idéale');ax.plot([100*x['confidence'] for x in reliability],[100*x['correct_fraction'] for x in reliability],'-o',color=COLORS[0]);ax.set(xlabel='Score moyen (%)',ylabel='Pixels concordants (%)',xlim=(50,100),ylim=(0,100));ax.legend();done()
    thresholds=np.linspace(.01,.99,99);precision=[];recall=[]
    for threshold in thresholds:
        pred=sample_p>=threshold;tp=np.sum(pred&(sample_t==1));fp=np.sum(pred&(sample_t==0));fn=np.sum((~pred)&(sample_t==1));precision.append(tp/(tp+fp) if tp+fp else np.nan);recall.append(tp/(tp+fn) if tp+fn else np.nan)
    fig,ax,done=figure(15,'Seuil pixel : compromis précision et rappel','Échantillon du lot analysé ; le seuil pixel diffère du seuil de contrôle géométrique.')
    ax.plot(thresholds,100*np.array(precision),color=COLORS[0],label='Précision des pixels couloir');ax.plot(thresholds,100*np.array(recall),'--',color=COLORS[1],label='Rappel des pixels couloir');ax.set(xlabel='Seuil de probabilité pixel',ylabel='Taux (%)',ylim=(0,100));ax.legend();done()
    if example:
        fig,ax,done=figure(16,'Entropie du réseau : où hésite-t-il ?','Une faible entropie peut aussi accompagner une prédiction fausse. Elle n’est pas une incertitude métrique.',size=(10,5.8))
        image,entropy,stem=example;ax.imshow(image);im=ax.imshow(entropy,cmap='magma',vmin=0,vmax=1,alpha=.65);ax.grid(False);ax.set_axis_off();fig.colorbar(im,ax=ax,label='Entropie binaire (bits)',shrink=.8);done()
    video_path=run/'video/mesures.csv'
    if video_path.is_file():
        video=[]
        with video_path.open() as f:
            for row in csv.DictReader(f):
                if row['status']=='candidate' and row['repeated_source_frame'] not in ['1','True','true']:
                    video.append(row)
        video_report=json.loads((video_path.parent/'video.json').read_text())
        source_name=Path(video_report['source']).name
        duration=video_report['duration_seconds']
        for number,key,title,ylabel in [(17,'apparent_width_px','Vidéo : stabilité de la largeur apparente','Largeur à y=0,8 hauteur (px origine)'),(18,'center_x_px','Vidéo : déplacement de l’axe à une hauteur fixe','Centre x à y=0,8 hauteur (px origine)')]:
            fig,ax,done=figure(number,title,'Les variations combinent mouvement, perspective et détection ; elles ne mesurent pas seules l’incertitude.',scope=f'Vidéo {source_name}, {duration:.1f} s ; observations candidates, répétitions exclues')
            ax.plot([float(x['seconds']) for x in video],[float(x[key]) for x in video],color=COLORS[0]);ax.set(xlabel='Temps vidéo (s)',ylabel=ylabel);ax.text(.01,.02,'Échelle verticale détaillée',transform=ax.transAxes,fontsize=9,color='#5c6870');done()
    sizes=sorted({x['network_width'] for x in sensitivity})
    fig,ax,done=figure(19,'Sensibilité à la résolution : segmentation','Même modèle, mêmes 12 images ; pas de réentraînement ni preuve de généralisation. Le cas 640×480 est VGA.',scope='Échantillon déterministe de 12 images du lot analysé')
    for width in sizes:
        values=[100*x['dice'] for x in sensitivity if x['network_width']==width];ax.scatter([width]*len(values),values,color=COLORS[0],alpha=.3)
    ax.plot(sizes,[np.mean([100*x['dice'] for x in sensitivity if x['network_width']==w]) for w in sizes],'-s',color=COLORS[1],label='Dice moyen');ax.set(xticks=sizes,xticklabels=['320×256','640×480','1280×960'],xlabel='Entrée réseau, proportions conservées',ylabel='Dice (%)',ylim=(0,100));ax.legend();done()
    fig,ax,done=figure(20,'Résolution : erreur, couverture de l’axe et coût de calcul','Sections appariées seulement pour la MAE ; lire la couverture en même temps. Moyennes de couverture par image, temps médians hors encodage.',scope='Les mêmes 12 images ; pixels d’origine pour l’évaluation',size=(13,5.2))
    fig.delaxes(ax)
    axes=fig.subplots(1,3)
    fig.subplots_adjust(left=.06,right=.97,bottom=.17,top=.80,wspace=.35)
    data=[
        [np.nanmean([x['axis_mae_pct_width'] for x in sensitivity if x['network_width']==w and x['axis_mae_pct_width'] is not None]) for w in sizes],
        [np.nanmean([x['coverage_pct'] for x in sensitivity if x['network_width']==w and x['coverage_pct'] is not None]) for w in sizes],
        [np.median([x['inference_seconds'] for x in sensitivity if x['network_width']==w]) for w in sizes]]
    for ax,values,ylabel,title,color in zip(axes,data,['MAE axe (% de largeur)','Couverture moyenne par image (%)','Temps médian / image (s)'],['Erreur sur sections appariées','Sections de référence retrouvées','Prédiction seule, hors encodage'],COLORS):
        ax.plot(sizes,values,'-o',color=color)
        ax.set(xticks=sizes,xticklabels=['320×256','640×480','1280×960'],xlabel='Entrée réseau',ylabel=ylabel,title=title,ylim=(0,100) if ylabel.startswith('Couverture') else (0,None))
        ax.tick_params(axis='x',labelsize=9)
    done()
    fig,ax,done=figure(21,'VGA et mètres : la perspective change l’échelle','Illustration : focale fictive 500 px, objet large de 1 m. Même image VGA, tailles projetées différentes.',scope='Exemple théorique fictif — aucune mesure de la voie TSV')
    depth=np.linspace(5,50,200);ax.plot(depth,500/depth,color=COLORS[0]);ax.scatter([5,10,20],[100,50,25],color=COLORS[1]);ax.set(xlabel='Distance supposée à la caméra (m)',ylabel='Largeur projetée de l’objet (px)',ylim=(0,110));done()
    fig,ax,done=figure(22,'Propagation d’une erreur de tracé vers les mètres','Hypothèses fictives : référence 1 m, erreurs indépendantes de 1/2/5 px par bord. Calibration et biais exclus.',scope='Exemple illustratif — pas une barre d’erreur validée pour TSV')
    separation=np.linspace(10,400,200)
    for sigma,color in zip([1,2,5],COLORS):ax.plot(separation,100*(1/separation)*(sigma/np.sqrt(2)),color=color,label=f'σ bord = {sigma} px')
    ax.set(xlabel='Largeur de la référence dans l’image (px)',ylabel='Incertitude-type du milieu (cm)',ylim=(0,40));ax.legend();done()
    global_metrics=Metrics(['background','couloir']);global_metrics.confusion=confusion
    payload={'scope':args.scope,'scope_note':scope_note,'images':len(rows),'validation_reviewed_at_build':sum(json.loads(Path(r['annotation']).read_text()).get('flags',{}).get('reviewed') is True for r in rows) if args.scope=='validation' else None,
        'checkpoint':str(checkpoint_path),'checkpoint_sha256':sha256(checkpoint_path),'network_trained_size':[checkpoint['width'],checkpoint['height']],
        'device':str(dev),'micro_pixel_metrics':global_metrics.result(),'macro_image_dice':float(np.nanmean(col('dice'))),
        'macro_image_iou':float(np.nanmean(col('iou'))),'axis_sections_matched':len(all_points),
        'axis_mean_absolute_pct_width':float(abs_err.mean()),'axis_p95_absolute_pct_width':float(np.percentile(abs_err,95)),
        'axis_reference_points':sum(x['reference_points'] for x in per_image),
        'axis_matched_points':sum(x['matched_points'] for x in per_image),
        'uncertainty_interval':None,'metric_scale':None,'confidence_sample_seed':42,'confidence_sample_pixels_per_image':20000,
        'resolution_sample':[Path(x['image']).stem for x in chosen],'figures':entries,
        'training_images':len(read_manifest(run/'dataset/train.csv')),
        'validation_used_for_model_selection':not checkpoint.get('training_only_demo',False),
        'sources':[{'image':x['image'],'image_sha256':sha256(x['image']),'mask':x['mask'],'mask_sha256':sha256(x['mask']),
                    'annotation':x.get('annotation'),'annotation_sha256':sha256(x['annotation']) if x.get('annotation') else None} for x in rows]}
    dump(out/'mesures_et_sources.json',payload)
    text=['# Figures du modèle : erreurs, VGA et incertitude','',scope_note+'.',
          '',f'{len(entries)} figures exportées en PNG et SVG. Chaque CSV garde les mesures détaillées.',
          '', 'Les 12 images de validation doivent être tracées avec couloir/ignore, reviewed puis Save avant la comparaison hors apprentissage.',
          'VGA signifie 640×480 pixels, pas une échelle métrique. Le document propose une calibration sur l’écartement constant, mais sa valeur spécifique reste à confirmer.',
          'Les écarts à vos polygones ne constituent pas une incertitude par rapport à la voie réelle. Une double annotation et des références physiques permettront de séparer ces contributions.',
          '', '## Comment lire les figures']
    for entry in entries:
        text += ['',f'### {entry["id"]}. {entry["title"]}',entry['scope'],entry['caption'],f'![{entry["title"]}](figures/{entry["id"]}.png)',f'[Version SVG](figures/{entry["id"]}.svg)']
    text+=['','## Données et sources','metriques_par_image.csv ; erreurs_axe_sections.csv ; sensibilite_resolution.csv ; diagramme_fiabilite.csv ; mesures_et_sources.json.',
           'Images VGA préparées dans entrees_vga (640×480, ratio conservé et bandes ajoutées). Le test logiciel a exécuté une inférence sur ces 12 entrées.',
           '[Calibration OpenCV](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) ; [homographie OpenCV](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html) ; [GUM/JCGM](https://www.bipm.org/en/committees/jc/jcgm/publications).']
    (out/'LIRE_LES_FIGURES.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('Figures terminées : '+str(out),flush=True)


if __name__=='__main__':main()
