"""Compare two real TSV models on the same frozen references and package outputs."""
import argparse,csv,html,json,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.diagnostic_plots import plt,COLORS
from railtsv.common import sha256,dump
from railtsv.model import load_checkpoint


def csv_rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def build(run,baseline):
    run=Path(run).resolve();baseline=Path(baseline).resolve()
    diagnostic=run/'diagnostic_validation'
    new=json.loads((diagnostic/'RESUME_VALIDATION.json').read_text())
    old=json.loads((baseline/'RESUME_VALIDATION.json').read_text())
    assert new['same_reference_masks_as_baseline']
    assert new['checkpoint_sha256']==sha256(run/'modele/best.pt')
    out=run/'comparaison_50_130';out.mkdir(exist_ok=False)
    figures=out/'figures';figures.mkdir()
    old_rows={x['image']:x for x in csv_rows(baseline/'metriques_par_image.csv')}
    new_rows={x['image']:x for x in csv_rows(diagnostic/'metriques_par_image.csv')}
    assert set(old_rows)==set(new_rows) and len(new_rows)==12
    stems=sorted(new_rows)
    annotations=json.loads((run/'annotations.json').read_text())
    history=json.loads((run/'modele/history.json').read_text())
    checkpoint=load_checkpoint(run/'modele/best.pt')
    video=json.loads((run/'video/video.json').read_text())
    assert video['checkpoint_sha256']==new['checkpoint_sha256']
    init=json.loads((run/'initialisation.json').read_text())
    ref_verified=[]
    for stem in stems:
        from PIL import Image
        new_mask=np.array(Image.open(diagnostic/'masks'/f'{stem}.png'))
        old_mask=np.array(Image.open(baseline/'masks'/f'{stem}.png'))
        assert np.array_equal(new_mask,old_mask)
        ref_verified.append(dict(stem=stem,new_mask_sha256=sha256(diagnostic/'masks'/f'{stem}.png'),old_mask_sha256=sha256(baseline/'masks'/f'{stem}.png')))
    plot_entries=[]
    def plot(number,title,caption,size=(9,5.2)):
        fig,ax=plt.subplots(figsize=size)
        fig.suptitle(title,x=.08,ha='left',fontsize=15,fontweight='bold')
        fig.text(.08,.925,'12 mêmes références, vidéo 03676 · annotations humaines · validation utilisée pour le choix du modèle',fontsize=8.5,color='#59656d')
        fig.subplots_adjust(left=.11,right=.97,bottom=.16,top=.82)
        def save():
            name=f'{number:02d}'
            fig.savefig(figures/f'{name}.png',dpi=180);fig.savefig(figures/f'{name}.svg');plt.close(fig)
            plot_entries.append(dict(id=name,title=title,caption=caption))
        return ax,save
    def paired(ax,labels,keys,scale=1,ylabel='',upper=None):
        x=np.arange(len(keys));w=.32
        b=ax.bar(x-w/2,[scale*old[k] for k in keys],w,label='Premier modèle : 50 images',color=COLORS[0])
        n=ax.bar(x+w/2,[scale*new[k] for k in keys],w,label='Nouveau modèle : 130 images',color=COLORS[1])
        ax.bar_label(b,fmt='%.2f',padding=3,fontsize=10);ax.bar_label(n,fmt='%.2f',padding=3,fontsize=10)
        ax.set(xticks=x,xticklabels=labels,ylabel=ylabel,ylim=(0,upper));ax.legend(loc='upper center',bbox_to_anchor=(.5,1.12),ncols=2,fontsize=10)
    ax,save=plot(1,'Segmentation : comparaison des deux modèles','Dice et IoU moyennés par image, pixels ignore exclus. Entrée 640×384 pour le premier modèle, 640×480 pour le nouveau ; 10 contre 20 passages. Le gain ne peut pas être attribué à la seule quantité d’annotations.')
    paired(ax,['Dice moyen','IoU moyenne'],['macro_image_dice','macro_image_iou'],100,'Score (%)',100);save()
    ax,save=plot(2,'Axe médian : erreurs sur les sections détectées','Erreur en pixels à une même hauteur de l’image originale. Les ensembles de sections détectées peuvent différer ; lire aussi la couverture.')
    paired(ax,['Erreur moyenne absolue','95e percentile observé'],['axis_mae_px','axis_p95_absolute_px'],1,'Écart absolu (pixels origine)',max(old['axis_p95_absolute_px'],new['axis_p95_absolute_px'])*1.45);save()
    ax,save=plot(3,'Couverture : les portions de référence retrouvées','Dénominateur : toutes les sections annotées exploitables, soit '+str(new['reference_sections'])+'. Un score interne élevé ne rend pas une section correcte.')
    paired(ax,['Couverture de l’axe'],['coverage_pct'],1,'Sections appariées (%)',100);save()
    ax,save=plot(4,'Dice sur chacune des 12 images','Les numéros correspondent aux comparaisons du dossier diagnostic_validation. Les références sont identiques entre les deux évaluations.')
    xx=np.arange(1,13)
    ax.plot(xx,[100*float(old_rows[s]['dice']) for s in stems],'-o',color=COLORS[0],label='50 images')
    ax.plot(xx,[100*float(new_rows[s]['dice']) for s in stems],'--s',color=COLORS[1],label='130 images')
    ax.set(xticks=xx,xlabel='Image de validation',ylabel='Dice (%)',ylim=(0,100));ax.legend();save()
    ax,save=plot(5,'Écart de l’axe, image par image','Erreur moyenne sur les sections appariées. Une image peut perdre des sections difficiles : la couverture par image figure dans le CSV.')
    ax.plot(xx,[float(old_rows[s]['axis_mae_px']) for s in stems],'-o',color=COLORS[0],label='50 images')
    ax.plot(xx,[float(new_rows[s]['axis_mae_px']) for s in stems],'--s',color=COLORS[1],label='130 images')
    ax.set(xticks=xx,xlabel='Image de validation',ylabel='Erreur moyenne absolue (px origine)',ylim=(0,None));ax.legend();save()
    ax,save=plot(6,'Données utilisées et données laissées de côté','Tous les dessins inclus sont reviewed et valides structurellement. Les 12 images de validation viennent d’une autre capture.')
    names=sorted(annotations['lots']);valid=[annotations['lots'][n]['reviewed_valid'] for n in names]
    excluded=[annotations['lots'][n]['expected']-v for n,v in zip(names,valid)]
    ax.bar(names,valid,color=COLORS[0],label='Incluses');ax.bar(names,excluded,bottom=valid,color='#c8d0d5',label='Exclues de cet essai')
    for i,(v,e) in enumerate(zip(valid,excluded)):
        ax.text(i,v/2,str(v),ha='center',va='center',color='white',fontsize=13)
        if e:ax.text(i,v+e/2,str(e),ha='center',va='center',color='#192f3b',fontsize=13)
    ax.set(xlabel='Lot attribué',ylabel='Nombre d’images',ylim=(0,60));ax.legend(loc='upper center',ncols=2);save()
    before_points={(r['image'],int(r['y_px'])):r for r in csv_rows(baseline/'erreurs_axe_sections.csv')}
    after_points={(r['image'],int(r['y_px'])):r for r in csv_rows(diagnostic/'erreurs_axe_sections.csv')}
    common=sorted(before_points.keys() & after_points.keys())
    assert common
    before_errors=np.array([abs(float(before_points[k]['center_error_px'])) for k in common])
    after_errors=np.array([abs(float(after_points[k]['center_error_px'])) for k in common])
    common_stats=dict(sections=len(common),before_mae_px=float(before_errors.mean()),after_mae_px=float(after_errors.mean()),before_p95_px=float(np.percentile(before_errors,95)),after_p95_px=float(np.percentile(after_errors,95)))
    with (out/'ERREURS_SUR_SECTIONS_COMMUNES.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['image','y_px','before_error_px','after_error_px']);w.writeheader()
        w.writerows(dict(image=k[0],y_px=k[1],before_error_px=before_points[k]['center_error_px'],after_error_px=after_points[k]['center_error_px']) for k in common)
    keys=['macro_image_dice','macro_image_iou','axis_mae_px','axis_p95_absolute_px','coverage_pct']
    comparison={k:dict(before=old[k],after=new[k],difference=new[k]-old[k]) for k in keys}
    report=dict(training_images=annotations['training_annotations'],validation_images=12,completed_epochs=len(history),selected_epoch=checkpoint['epoch'],network_size=new['network_size'],initialization=init,comparison=comparison,common_axis_sections=common_stats,reference_masks=ref_verified,validation_used_for_model_selection=True,third_capture_test_available=False,units='original_image_pixels',meters=None,old_report=str(baseline),new_report=str(diagnostic),checkpoint_sha256=new['checkpoint_sha256'],video_source=video['source'],video_seconds=video['duration_seconds'],video_frames=video['frames'],figures=plot_entries)
    dump(out/'COMPARAISON.json',report)
    comparison_rows=[]
    for i,stem in enumerate(stems,1):
        r=dict(image_number=i,image=stem)
        for label,source in [('before',old_rows),('after',new_rows)]:
            for key in ['dice','iou','axis_mae_px','coverage_pct','matched_points','reference_points']:r[label+'_'+key]=source[stem][key]
        comparison_rows.append(r)
    with (out/'COMPARAISON_PAR_IMAGE.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(comparison_rows[0]));w.writeheader();w.writerows(comparison_rows)
    with (run/'IMAGES_EXCLUES_DE_CET_ESSAI.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['lot','stem','reason']);w.writeheader();w.writerows(annotations['exclusions'])
    def fmt(v):return f'{v:.2f}'.replace('.',',')
    labels=[('Dice moyen','macro_image_dice',100,'%'),('IoU moyenne','macro_image_iou',100,'%'),('Erreur moyenne de l’axe','axis_mae_px',1,'px'),('95e percentile de l’erreur','axis_p95_absolute_px',1,'px'),('Couverture de l’axe','coverage_pct',1,'%'),('Écart maximal observé','axis_max_absolute_px',1,'px')]
    table='<table><tr><th>Mesure</th><th>50 images</th><th>130 images</th></tr>'+''.join(f'<tr><td>{label}</td><td>{fmt(scale*old[key])} {unit}</td><td>{fmt(scale*new[key])} {unit}</td></tr>' for label,key,scale,unit in labels)+'</table>'
    style='<style>body{font:17px/1.55 system-ui;background:#f3f6f8;color:#17343f;margin:0}main{max-width:1150px;padding:28px;margin:auto}section,article{background:white;padding:22px;margin:18px 0;border:1px solid #dce5e8;border-radius:12px}img,video{width:100%;height:auto}td,th{padding:12px;text-align:left;border-bottom:1px solid #dce5e8}a{color:#076a8b}.note{border-left:5px solid #bc7222}</style>'
    page=['<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nouvel essai : 130 annotations</title>',style,'<main><h1>Nouvel essai : 130 annotations</h1>',f'<p>Lot A : 50 · lot B : 30 · lot C : 50. DeepLabV3+ / ResNet50 · entrée VGA 640×480 · MPS. {len(history)} passages terminés ; modèle du passage {checkpoint["epoch"]} sélectionné.</p>','<section><h2>1. Ce qui a changé</h2><p>Le modèle repart du préapprentissage RailSem19 : 2 passages terminés sur 3 000 images génériques. Il apprend ensuite le couloir entre les bords intérieurs des rails à partir de vos 130 annotations. Les 20 images restantes du lot B sont exclues de cet essai.</p></section>','<section><h2>2. Comparaison sur les mêmes 12 références</h2>',table,f'<p>Sur les {len(common)} sections où les deux modèles ont un axe : erreur moyenne {common_stats["before_mae_px"]:.2f} px avant, {common_stats["after_mae_px"]:.2f} px après. <a href="comparaison_50_130/ERREURS_SUR_SECTIONS_COMMUNES.csv">CSV sur ces sections communes</a>.</p>','<p>Les erreurs sont exprimées en pixels des images originales 1920×1080. Dice et IoU sont des moyennes par image. Le 95e percentile décrit les écarts observés ; ce n’est pas un intervalle de confiance.</p><p><a href="diagnostic_validation/VOIR_LA_VALIDATION.html">Voir les 12 annotations face aux prédictions et les figures détaillées</a></p></section>','<section class="note"><h2>Comment interpréter ces résultats</h2><p>La vidéo 03676 est séparée des captures utilisées pour apprendre. Ses 12 annotations ont toutefois servi à sélectionner le meilleur passage : elles constituent une validation, et pas un test final indépendant. Il reste à évaluer une troisième capture. Le nombre d’images, la résolution et le nombre de passages ont tous changé ; cette comparaison ne mesure pas l’effet isolé des lots B et C.</p><p><a href="POINT_A_RELIRE_IMAGE_10.md">Point de relecture : fermeture basse de l’image 10</a>. Le plus grand écart brut est affecté par cette fermeture oblique ; les scores conservés restent relatifs aux annotations humaines.</p><p>Les scores comparent vos polygones aux prédictions. Une section détectée ou un score interne élevé peut encore correspondre à une erreur. Le lissage stabilise l’affichage, avec un possible retard dans les virages. Les mètres et une incertitude physique restent indisponibles sans calibration.</p></section>',f'<section><h2>3. Nouvelle vidéo et données pour l’équipe gabarit</h2><p>Vidéo muette de {video["duration_seconds"]:.1f} s. Fond cyan : couloir brut ; axe vert : milieu des limites lissées ; panneau : largeur apparente en pixels, score interne non calibré et statut.</p><video controls muted playsinline preload="metadata" src="video/rails_annotes.mp4"></video><p><a href="video/VOIR_VIDEO_ET_MESURES.html">Vidéo et tous les CSV</a> · <a href="video/DONNEES_POUR_EQUIPE_GABARIT.zip">ZIP à envoyer à l’équipe gabarit</a> · <a href="video/axe_brut_et_lisse_par_seconde.csv">Axe brut et lissé par seconde</a> · <a href="video/mesures_par_seconde.csv">Mesures par seconde</a></p></section>','<section><h2>4. Graphiques comparatifs</h2><p><a href="comparaison_50_130/COMPARAISON_PAR_IMAGE.csv">Données de comparaison par image</a> · <a href="IMAGES_EXCLUES_DE_CET_ESSAI.csv">Les 20 images exclues</a></p></section>']
    for entry in plot_entries:page.append(f'<article><h3>{entry["title"]}</h3><p>{entry["caption"]}</p><img loading="lazy" src="comparaison_50_130/figures/{entry["id"]}.png" alt="{html.escape(entry["title"])}"><a href="comparaison_50_130/figures/{entry["id"]}.svg">Version SVG</a></article>')
    page+=['<section><h2>5. La suite</h2><p>Regarder la vidéo et les images où l’axe ou la couverture se dégrade. Compléter les 20 annotations manquantes du lot B, puis refaire un entraînement dans un nouveau dossier. Réserver ensuite une troisième capture, sans l’utiliser pour régler le modèle, afin de mesurer sa généralisation. La calibration en mètres viendra après.</p><p>Le dossier dataset contient les copies utilisées pour cet essai ; modele/best.pt est le modèle retenu ; diagnostic_validation contient les scores et comparaisons ; video contient la vidéo et les CSV ; comparaison_50_130 contient les graphiques comparatifs. Les sources originales et les précédents essais sont conservés.</p></section></main></html>']
    (run/'VOIR_LE_NOUVEL_ESSAI.html').write_text('\n'.join(page),encoding='utf-8')
    lines=['# Essai avec les annotations disponibles','',f'130 images pour apprendre : A50 + B30 + C50. Validation : 12 références de la vidéo 03676.',f'{len(history)} passages terminés, meilleur passage : {checkpoint["epoch"]}. Initialisation RailSem19 (2 passages génériques terminés). Entrée : VGA 640×480, calcul MPS.','','| Mesure | Premier modèle (50 images) | Nouveau modèle (130 images) |','|---|---:|---:|']
    lines += [f'| {label} | {fmt(scale*old[key])} {unit} | {fmt(scale*new[key])} {unit} |' for label,key,scale,unit in labels]
    lines += ['','Moyennes Dice/IoU par image. Erreurs de l’axe sur les sections appariées, pixels de l’image originale. Lire la couverture en même temps.',f'Sur {len(common)} sections communes aux deux modèles : MAE avant {common_stats["before_mae_px"]:.2f} px, après {common_stats["after_mae_px"]:.2f} px.',
    'Les mêmes références ont été utilisées ; elles servent au choix du modèle. Une troisième capture est encore nécessaire pour un test final.','', '[Ouvrir le résultat](VOIR_LE_NOUVEL_ESSAI.html).','[Voir la vidéo et les CSV](video/VOIR_VIDEO_ET_MESURES.html).','', '[Point de relecture de l’image 10](POINT_A_RELIRE_IMAGE_10.md). Le grand écart au bas de cette image est affecté par la fermeture du polygone ; aucune annotation source n’a été changée.',
    'Les champs en mètres restent vides. Le score interne n’est pas un pourcentage de précision réelle. Le lissage peut retarder les mouvements et ne corrige pas une mauvaise voie détectée.', '', 'Les 20 fichiers exclus sont listés dans IMAGES_EXCLUES_DE_CET_ESSAI.csv. Les annotations et essais précédents sont conservés.']
    (run/'LIRE_ESSAI_130_IMAGES.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ['training_images','completed_epochs','selected_epoch','comparison']},indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    a=p.parse_args();build(a.run,a.baseline)
