"""Frozen 60-image public holdout diagnostic, sampled before any predictions."""
from axes_eval_20261018 import *
from PIL import ImageDraw,ImageFont
from matplotlib.font_manager import findfont

def main():
    dest=OUT/'test_public_60';dest.mkdir(exist_ok='--resume' in sys.argv)
    if (dest/'COMPLETE.json').exists():raise FileExistsError('Completed diagnostic already exists')
    prepared=ROOT/'data/prepared/railsem_egopath_20261014'
    rows=read_manifest(prepared/'test.csv');assert len(rows)==479
    annotations=json.loads((ROOT/'data/raw/railsem19_egopath_20261014/rs19_egopath.json').read_text())
    ranked=[]
    for r in rows:
        a=annotations[Path(r['image']).name]
        l=np.array(a['left_rail'],float);rr=np.array(a['right_rail'],float)
        lo=max(l[:,1].min(),rr[:,1].min());hi=min(l[:,1].max(),rr[:,1].max())
        if hi-lo<24:continue
        yy=np.linspace(lo,hi,80);l=l[l[:,1].argsort()];rr=rr[rr[:,1].argsort()]
        xx=(np.interp(yy,l[:,1],l[:,0])+np.interp(yy,rr[:,1],rr[:,0]))/2
        line=np.polyval(np.polyfit(yy,xx,1),yy)
        with Image.open(r['image']) as im:width=im.width
        score=float(np.sqrt(np.mean((xx-line)**2))/width)
        ranked.append(dict(r,curvature_proxy=score))
    ranked.sort(key=lambda r:r['curvature_proxy']);rng=np.random.default_rng(20261018);selected=[]
    for group,indices in zip(['faible','moyenne','forte'],np.array_split(np.arange(len(ranked)),3)):
        for i in sorted(rng.choice(indices,20,replace=False)):
            row=dict(ranked[int(i)],group=group);row['image_sha256']=sha256(row['image']);selected.append(row)
    # Byte-identical leakage gate against all public/TSV training and validation inputs.
    manifests=[prepared/'train.csv',prepared/'val.csv',EGO/'dataset/train.csv',EGO/'dataset/val.csv']
    paths={r['image'] for p in manifests for r in read_manifest(p)}
    selection_hashes={r['image_sha256'] for r in selected}
    collision=[]
    for p in paths:
        if sha256(p) in selection_hashes:collision.append(p)
    assert not collision,'Identical images in train/val and holdout'
    dump(dest/'selection.json',dict(seed=20261018,images=selected,test_pool=479,remaining_untouched=419,
        curvature_definition='RMS departure of annotated image-space midpoint from its best straight line / image width',
        grouping='20 from each rank tertile; not labels of physical radius or switches',
        source_video_disjointness_proven=False,byte_identical_leakage_found=False,
        selection_before_predictions=True,training_or_tuning_on_this_subset=False))
    selected_models=['ancien_tsv','ego_tsv','public_seul_natif'];dev=device('auto');torch.set_num_threads(4)
    summaries=[];allimages=[];allpoints=[]
    for name in selected_models:
        path,h,w=MODELS[name];ck=load_checkpoint(path);model=build_model(2);model.load_state_dict(ck['state_dict']);model.to(dev).eval()
        viewdir=dest/name;viewdir.mkdir(exist_ok='--resume' in sys.argv);images=[];points=[]
        for i,row in enumerate(selected,1):
            rgb=rgb_read(row['image']);ref=mask_read(row['mask'],[0,1,255]);stem=Path(row['image']).stem
            pred,prob=predict_image(model,rgb,dev,h,w);m,pp,_=image_metrics(pred,ref,prob,stem,row['group'])
            images.append(dict(model=name,**m));points.extend(dict(model=name,**p) for p in pp)
            if i in [1,21,41,50,55,60]:
                view=rgb.copy();view[pred==1]=(view[pred==1]*.8+np.array([20,190,240])*.2).astype(np.uint8)
                view=draw_axis(draw_corridor_limits(view,corridor_profile(pred,prob)),corridor_profile(pred,prob))
                Image.fromarray(view).resize((640,360)).save(viewdir/f'{stem}.jpg',quality=92)
                gt=rgb.copy();gt[ref==1]=(gt[ref==1]*.7+np.array([20,190,240])*.3).astype(np.uint8)
                Image.fromarray(gt).resize((640,360)).save(dest/f'{stem}_reference.jpg',quality=92)
            if (i%10)==0:print(f'Public holdout {name}: {i}/60',flush=True)
        for group in ['toutes','faible','moyenne','forte']:
            ms=[r for r in images if group=='toutes' or r['group']==group];ps=[p for p in points if group=='toutes' or p['group']==group]
            nref=sum(r['reference_points'] for r in ms)
            summaries.append(dict(model=name,group=group,images=len(ms),dice_macro=float(np.mean([r['dice'] for r in ms])),
                coverage_pct=100*len(ps)/nref if nref else None,**stats([p['center_error_px'] for p in ps])))
        allimages.extend(images);allpoints.extend(points);del ck,model
        if dev.type=='mps':torch.mps.empty_cache()
    write_csv(dest/'METRIQUES.csv',summaries);write_csv(dest/'PAR_IMAGE.csv',allimages);write_csv(dest/'ERREURS.csv',allpoints)
    font=ImageFont.truetype(findfont('DejaVu Sans'),20)
    selected_views=[row for i,row in enumerate(selected,1) if i in [1,21,41,50,55,60]]
    for page in range(2):
        canvas=Image.new('RGB',(2560,3*400),'white');draw=ImageDraw.Draw(canvas)
        for i,row in enumerate(selected_views[page*3:page*3+3]):
            stem=Path(row['image']).stem
            for j,(title,path) in enumerate([('Référence publique',dest/f'{stem}_reference.jpg')]+[(n,dest/n/f'{stem}.jpg') for n in selected_models]):
                draw.text((j*640+8,i*400+5),f'{stem} | {title}',font=font,fill='#17232d');canvas.paste(Image.open(path),(j*640,i*400+35))
        canvas.save(dest/f'EXEMPLES_{page+1}.jpg',quality=93)
    dump(dest/'COMPLETE.json',dict(complete=True,models=selected_models,images=60,new_annotations=0))

if __name__=='__main__':main()
