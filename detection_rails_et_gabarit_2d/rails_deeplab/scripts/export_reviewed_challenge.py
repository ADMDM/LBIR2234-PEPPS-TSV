"""Freeze reviewed reinforcement/control labels without changing the active model."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.export_labelme import rasterize
from railtsv.common import read_manifest,write_manifest,check_disjoint,sha256
from railtsv.validation_protocol import final_split_guard


def export(batch,out,final_registry=None):
    batch,out=Path(batch).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    data=json.loads((batch/'MANIFESTE.json').read_text())
    accepted=[];excluded=[];pending=[]
    for r in data['cases']:
        image=ROOT/r['image'];annotation=ROOT/r['json'];label=json.loads(annotation.read_text())
        if sha256(image)!=r['image_sha256']:raise ValueError('Image modifiée : '+str(image))
        if label.get('flags',{}).get('reviewed') is not True:pending.append(annotation.name);continue
        if label.get('flags',{}).get('exclude_from_training'):
            excluded.append(dict(id=r['id'],role=r['role'],reason='human_excluded'));continue
        with Image.open(image) as im:mask=rasterize(label,im.size,task='corridor')
        if not np.any(mask!=255):
            excluded.append(dict(id=r['id'],role=r['role'],reason='all_ignore'));continue
        accepted.append((r,mask))
    if pending:raise ValueError(f'{len(pending)} images restent à relire : '+', '.join(pending[:6]))
    run=ROOT/json.loads((ROOT/'configs/diagnostic_target.json').read_text())['run']
    rows={'train':read_manifest(run/'dataset/train.csv'),'val':read_manifest(run/'dataset/val.csv'),'test':[]}
    if not any(r['role']=='renfort_a_annoter' for r,m in accepted) or not any(r['role']=='controle_nouvelle_capture' for r,m in accepted):
        raise ValueError('Au moins un renfort et un contrôle annotés sont nécessaires')
    # Historical control frames have already been inspected and used to tune
    # crop/filter parameters. They remain development data by default.
    controls=[r for r,m in accepted if r['role']=='controle_nouvelle_capture']
    final_verification=None
    if final_registry:
        registry=json.loads(Path(final_registry).read_text())
        final_verification=final_split_guard(registry,sorted({r['sequence'] for r in controls}))
        known={r['capture_id']:r for r in registry}
        for r in controls:
            if sha256(ROOT/r['source'])!=known[r['sequence']]['source_sha256']:
                raise ValueError('Source vidéo différente du registre final')
    else:
        rows.pop('test');rows['challenge_dev']=[]
    out.mkdir(parents=True);(out/'masks').mkdir();(out/'annotations_figees').mkdir()
    for r,mask in accepted:
        image=ROOT/r['image'];label=ROOT/r['json'];target=out/'masks'/(image.stem+'.png')
        Image.fromarray(mask).save(target);shutil.copy2(label,out/'annotations_figees'/label.name)
        split='train' if r['role']=='renfort_a_annoter' else ('test' if final_registry else 'challenge_dev')
        rows[split].append(dict(image=str(image),mask=str(target),sequence=r['sequence']))
    control_split='test' if final_registry else 'challenge_dev'
    for a,b in [('train','val'),('train',control_split),('val',control_split)]:check_disjoint(rows[a],rows[b])
    for key,records in rows.items():write_manifest(out/(key+'.csv'),records)
    (out/'EXPORT.json').write_text(json.dumps(dict(counts={k:len(v) for k,v in rows.items()},excluded=excluded,
        training_performed=False,validation='Original second capture for model selection',
        control_split=control_split,final_verification=final_verification,
        control_scope='Final only with verified frozen capture registry; otherwise development challenge, not independent test',
        source_manifest_sha256=sha256(batch/'MANIFESTE.json')),indent=2))
    print('Dataset figé :',out,{k:len(v) for k,v in rows.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--out',required=True);p.add_argument('--final-registry',help='Explicit verified capture registry; omission exports challenge_dev, never test');a=p.parse_args();export(a.batch,a.out,a.final_registry)
