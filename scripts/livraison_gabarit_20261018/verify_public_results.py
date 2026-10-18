"""Independent arithmetic check of frozen public results and exportable figures."""
from pathlib import Path
import csv,json,math,statistics,argparse,os

def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);a=ap.parse_args();p=a.folder
    assert json.loads((p/'TERMINE.json').read_text())['complete']
    summary=rows(p/'METRIQUES.csv');verified=[];details={};ims={}
    for row in summary:
        name=row['model'];images=rows(p/f'{name}_PAR_IMAGE.csv');points=rows(p/f'{name}_ERREURS.csv')
        assert len(images)==419 and len({r['image'] for r in images})==419
        errors=[float(r['center_error_px']) for r in points]
        recomputed=dict(dice_macro=statistics.mean(float(r['dice']) for r in images),
            iou_macro=statistics.mean(float(r['iou']) for r in images),
            coverage_pct=100*sum(int(r['matched_points']) for r in images)/sum(int(r['reference_points']) for r in images),
            mae_px=statistics.mean(abs(v) for v in errors),rmse_px=math.sqrt(statistics.mean(v*v for v in errors)),
            bias_px=statistics.mean(errors))
        assert len(points)==sum(int(r['matched_points']) for r in images)==int(row['n'])
        for k,v in recomputed.items():assert math.isclose(v,float(row[k]),abs_tol=1e-8),(name,k)
        verified.append(dict(model=name,**recomputed));details[name]={(r['image'],r['y_px']):float(r['center_error_px']) for r in points}
        ims[name]={r['image']:r for r in images}
    common=set(details['ancien_tsv'])&set(details['ego_tsv'])
    for r in rows(p/'COMPARAISON_AXE_SECTIONS_COMMUNES.csv'):
        assert int(r['n'])==len(common)
        assert math.isclose(float(r['mae_px']),statistics.mean(abs(details[r['model']][k]) for k in common),abs_tol=1e-8)
    comparison=[]
    for name in sorted(ims['ancien_tsv']):
        keys=[k for k in common if k[0]==name]
        row=dict(image=name,common_sections=len(keys),old_dice=float(ims['ancien_tsv'][name]['dice']),
            ego_dice=float(ims['ego_tsv'][name]['dice']))
        if keys:
            row.update(old_common_mae_px=statistics.mean(abs(details['ancien_tsv'][k]) for k in keys),
                ego_common_mae_px=statistics.mean(abs(details['ego_tsv'][k]) for k in keys))
            row['improvement_px']=row['old_common_mae_px']-row['ego_common_mae_px']
        comparison.append(row)
    with (p/'COMPARAISON_APPARIEE_PAR_IMAGE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['image','common_sections','old_dice','ego_dice','old_common_mae_px','ego_common_mae_px','improvement_px']);w.writeheader();w.writerows(comparison)
    check=dict(arithmetic_ok=True,images=419,models=verified,common_axis_sections=len(common),
        images_with_common_sections=sum(bool(r['common_sections']) for r in comparison),
        images_with_lower_common_axis_mae=sum(r.get('improvement_px',0)>0 for r in comparison),
        image_errors_are_correlated_within_scenes=True,independent_final_tsv_test=False,
        no_statistical_confidence_claim=True)
    (p/'VERIFICATION_INDEPENDANTE.json').write_text(json.dumps(check,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#68788B','#C87530','#718747'];labels=['Ancien TSV','Ego-path + TSV','Public ego-path seul']
    fig,axes=plt.subplots(1,3,figsize=(14,5.6),layout='constrained')
    for ax,key,title in zip(axes,['dice_macro','iou_macro','coverage_pct'],['Dice moyen','IoU moyenne','Sections de référence retrouvées']):
        vals=[r[key]*(1 if key=='coverage_pct' else 100) for r in verified]
        bars=ax.bar(range(3),vals,color=colors,width=.65)
        ax.set_xticks(range(3),['Ancien\nTSV','Ego-path\n+ TSV','Public\nego-path seul']);ax.set_ylim(0,110);ax.set_yticks(range(0,101,20));ax.set_ylabel('%');ax.set_title(title,fontsize=12)
        ax.bar_label(bars,labels=[f'{v:.1f} %' for v in vals],padding=5)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('419 images publiques réservées · modèles inchangés\nEntrée 640 × 480 ; ce test ne valide pas les mètres ni les aiguillages TSV',fontsize=15)
    fig.savefig(p/'01_TEST_PUBLIC_419.png',dpi=170);fig.savefig(p/'01_TEST_PUBLIC_419.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,7.5),layout='constrained')
    ax.scatter([100*r['old_dice'] for r in comparison],[100*r['ego_dice'] for r in comparison],s=14,alpha=.5,color=colors[1])
    ax.plot([0,100],[0,100],'--',color='#39434B',lw=1.4);ax.set(xlim=(0,100),ylim=(0,100),xlabel='Dice ancien TSV (%)',ylabel='Dice ego-path + TSV (%)')
    ax.set_aspect('equal');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    ax.set_title('Chaque point représente une des 419 images\nAu-dessus de la diagonale : meilleur accord avec le masque public',fontsize=12)
    fig.savefig(p/'02_COMPARAISON_PAR_IMAGE.png',dpi=170);plt.close(fig)
    print(json.dumps(check,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
