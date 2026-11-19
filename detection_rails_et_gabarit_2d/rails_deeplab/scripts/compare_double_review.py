"""Compare two independent reviewed Labelme corridors; neither is ground truth."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'railtsv').is_dir());sys.path.insert(0,str(ROOT))
from scripts.export_labelme import rasterize
from railtsv.common import sha256

def compare(a,b):
    a,b=Path(a),Path(b);result=[]
    names={p.name for p in a.glob('*.json')}
    if not names or names!={p.name for p in b.glob('*.json')}:raise ValueError('Les deux dossiers doivent contenir exactement les mêmes JSON')
    for name in sorted(names):
        pair=[];hashes=[]
        for folder in [a,b]:
            data=json.loads((folder/name).read_text());im=(folder/data['imagePath']).resolve()
            hashes.append(sha256(im))
            with Image.open(im) as image:pair.append(rasterize(data,image.size,task='corridor'))
        if len(set(hashes))!=1 or pair[0].shape!=pair[1].shape:raise ValueError('Images différentes : '+name)
        ma,mb=pair;valid=(ma!=255)&(mb!=255);pa=(ma==1)&valid;pb=(mb==1)&valid
        den=int(pa.sum()+pb.sum());dice=2*np.count_nonzero(pa&pb)/den if den else None
        errors=[]
        for y in range(0,ma.shape[0],8):
            ends=[]
            for m in pair:
                xs=np.flatnonzero(m[y]==1)
                if len(xs)<2 or np.any(np.diff(xs)>1) or xs[0]==0 or xs[-1]==m.shape[1]-1 or m[y,xs[0]-1]!=0 or m[y,xs[-1]+1]!=0:break
                ends.append((xs[0],xs[-1]))
            if len(ends)==2:
                dl=ends[1][0]-ends[0][0];dr=ends[1][1]-ends[0][1];errors.append([dl,dr,(dl+dr)/2])
        e=np.asarray(errors)
        result.append(dict(image=name,image_sha256=hashes[0],common_sections=len(errors),dice_between_reviewers=dice,
            left_mae_px=float(abs(e[:,0]).mean()) if len(e) else None,right_mae_px=float(abs(e[:,1]).mean()) if len(e) else None,
            axis_mae_px=float(abs(e[:,2]).mean()) if len(e) else None,axis_bias_B_minus_A_px=float(e[:,2].mean()) if len(e) else None,
            interpretation='Disagreement between two reviewers; no reviewer is declared true; not a model error.'))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--a',required=True);p.add_argument('--b',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    rows=compare(args.a,args.b)
    with Path(args.out).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Comparaison créée ; désaccord humain, pas vérité terrain absolue :',args.out)
