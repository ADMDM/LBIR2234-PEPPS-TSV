from axes_eval_20261018 import *

dest=OUT/'test_public_60';selection=json.loads((dest/'selection.json').read_text())['images']
assert len(selection)==60 and not (dest/'CONTROLE_VGA.csv').exists()
path,h,w=MODELS['public_seul_vga'];ck=load_checkpoint(path);dev=device('auto');torch.set_num_threads(4)
model=build_model(2);model.load_state_dict(ck['state_dict']);model.to(dev).eval();images=[];points=[]
for i,row in enumerate(selection,1):
    rgb=rgb_read(row['image']);ref=mask_read(row['mask'],[0,1,255]);stem=Path(row['image']).stem
    pred,prob=predict_image(model,rgb,dev,h,w);m,pp,_=image_metrics(pred,ref,prob,stem,row['group'])
    images.append(dict(model='public_seul_vga',**m));points.extend(dict(model='public_seul_vga',**p) for p in pp)
    if i%15==0:print(f'Public VGA control: {i}/60',flush=True)
summary=[]
for group in ['toutes','faible','moyenne','forte']:
    ms=[r for r in images if group=='toutes' or r['group']==group];ps=[r for r in points if group=='toutes' or r['group']==group]
    summary.append(dict(model='public_seul_vga',group=group,images=len(ms),dice_macro=float(np.mean([r['dice'] for r in ms])),
        coverage_pct=100*len(ps)/sum(r['reference_points'] for r in ms),**stats([r['center_error_px'] for r in ps])))
write_csv(dest/'CONTROLE_VGA.csv',summary);write_csv(dest/'CONTROLE_VGA_PAR_IMAGE.csv',images);write_csv(dest/'CONTROLE_VGA_ERREURS.csv',points)
