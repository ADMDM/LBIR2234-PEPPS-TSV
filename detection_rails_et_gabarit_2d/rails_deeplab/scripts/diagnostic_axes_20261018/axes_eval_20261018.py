"""Evaluate public-only ego-path and TSV models on frozen references; no training."""
from pathlib import Path
import os,sys,json,csv,argparse,copy
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'railtsv').is_dir())
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import torch
from PIL import Image
from railtsv.common import read_manifest,rgb_read,mask_read,dump,sha256,device
from railtsv.model import build_model,load_checkpoint
from railtsv.engine import predict_image
from railtsv.geometry import corridor_profile,draw_axis,draw_corridor_limits
from scripts.diagnostic_plots import image_metrics,write_csv
EGO=ROOT/'runs/experience_egopath_20261016_141426'
BASE=ROOT/'runs/entrainement_couloir_20261009_143754_101cbf'
OUT=ROOT/'runs/fusion_axes_20261018'
MODELS={
 'ancien_tsv':(BASE/'modele/best.pt',480,640),
 'ego_tsv':(EGO/'modele/best.pt',480,640),
 'public_seul_vga':(EGO/'preapprentissage_egopath/best.pt',480,640),
 'public_seul_natif':(EGO/'preapprentissage_egopath/best.pt',384,640)}

def indexed(profile):
    return {p['y']:dict(p,segment_id=s) for s,seg in enumerate(profile['segments']) for p in seg}

def reference_profile(ref):
    p=corridor_profile(ref,np.stack([ref!=1,ref==1]).astype(np.float32))
    # Break at ignored boundaries; never connect across ignored reference rows.
    segs=[]
    for seg in p['segments']:
        cur=[]
        for x in seg:
            if np.any(ref[x['y'],max(0,x['left_inner_x']-1):x['right_inner_x']+2]==255):
                if cur:segs.append(cur)
                cur=[]
            else:cur.append(x)
        if cur:segs.append(cur)
    return dict(segments=segs)

def spatial_profile(profile):
    """Local quadratic fit, seven sampled rows, within each existing segment."""
    out=copy.deepcopy(profile)
    for seg in out['segments']:
        if len(seg)<7:continue
        ys=np.array([p['y'] for p in seg],float)
        edges=np.array([[p['left_inner_x'],p['right_inner_x']] for p in seg],float)
        for i,p in enumerate(seg):
            a,b=max(0,i-3),min(len(seg),i+4)
            fitted=np.polynomial.polynomial.polyfit(ys[a:b]-ys[i],edges[a:b],2)[0]
            if fitted[0]<fitted[1]:
                p.update(left_inner_x=float(fitted[0]),right_inner_x=float(fitted[1]),
                         center_x=float(fitted.mean()),apparent_width_px=float(np.diff(fitted)[0]))
    return out

def stats(errors):
    a=np.asarray(errors,float)
    return dict(n=len(a),mae_px=float(abs(a).mean()) if len(a) else None,
        rmse_px=float(np.sqrt((a*a).mean())) if len(a) else None,
        bias_px=float(a.mean()) if len(a) else None,
        p95_abs_px=float(np.percentile(abs(a),95)) if len(a) else None)

def main():
    OUT.mkdir(exist_ok=False);(OUT/'evaluation').mkdir();(OUT/'references').mkdir()
    rows=read_manifest(EGO/'dataset/val.csv');assert len(rows)==12
    dev=device('auto');torch.set_num_threads(4)
    manifest=[]
    for row in rows:
        stem=Path(row['image']).stem
        import shutil
        shutil.copy2(row['image'],OUT/'references'/f'{stem}.png')
        shutil.copy2(row['mask'],OUT/'references'/f'{stem}_mask.png')
        manifest.append(dict(image_id=stem,image_sha256=sha256(row['image']),mask_sha256=sha256(row['mask']),
            sequence=row['sequence'],image_file=f'references/{stem}.png',mask_file=f'references/{stem}_mask.png',
            video_time_verified=False,note='Do not join to decoded video timestamps by filename alone'))
    dump(OUT/'references.json',manifest)
    dump(OUT/'protocol.json',dict(date='2026-10-18',device=str(dev),training=False,new_manual_annotations=False,
        validation_images=12,validation_captures=1,independent_test=False,
        source_manifest_sha256=sha256(EGO/'dataset/val.csv'),
        models={n:dict(checkpoint=str(p),checkpoint_sha256=sha256(p),network_height=h,network_width=w) for n,(p,h,w) in MODELS.items()},
        spatial_filter='Exploratory local quadratic, 7 points at 8px spacing; no segment bridging',
        fusion='Equal mean on shared observations, no fitted weights'))
    for name,(path,h,w) in MODELS.items():
        dest=OUT/'evaluation'/name;dest.mkdir();(dest/'profiles').mkdir();(dest/'apercus').mkdir()
        ck=load_checkpoint(path);model=build_model(2);model.load_state_dict(ck['state_dict']);model.to(dev).eval()
        per_image=[];details=[]
        for i,row in enumerate(rows,1):
            rgb=rgb_read(row['image']);ref=mask_read(row['mask'],[0,1,255]);stem=Path(row['image']).stem
            pred,prob=predict_image(model,rgb,dev,h,w)
            m,pp,_=image_metrics(pred,ref,prob,stem,row['sequence']);per_image.append(m);details.extend(pp)
            profile=corridor_profile(pred,prob)
            dump(dest/'profiles'/f'{stem}.json',profile)
            view=rgb.copy();view[pred==1]=(view[pred==1]*.8+np.array([20,190,240])*.2).astype(np.uint8)
            view=draw_axis(draw_corridor_limits(view,profile),profile)
            Image.fromarray(view).resize((960,540)).save(dest/'apercus'/f'{stem}.jpg',quality=93)
            print(f'{name}: {i}/12 Dice={m["dice"]:.4f} coverage={m["coverage_pct"]:.1f}%',flush=True)
        write_csv(dest/'images.csv',per_image);write_csv(dest/'errors.csv',details)
        dump(dest/'summary.json',dict(model=name,checkpoint_sha256=sha256(path),
            dice_macro=float(np.mean([r['dice'] for r in per_image])),iou_macro=float(np.mean([r['iou'] for r in per_image])),
            coverage_pct=100*sum(r['matched_points'] for r in per_image)/sum(r['reference_points'] for r in per_image),
            **stats([r['center_error_px'] for r in details])))
        del model,ck
        if dev.type=='mps':torch.mps.empty_cache()
    dump(OUT/'evaluation_complete.json',dict(complete=True,models=list(MODELS)))

if __name__=='__main__':main()
