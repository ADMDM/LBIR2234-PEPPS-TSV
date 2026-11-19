"""Suggest contours without replacing human work or checking reviewed."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import cv2
from PIL import Image
import numpy as np
from railtsv.annotation_config import active_batch
from railtsv.common import TASKS,device,dump,new_dir,rgb_read,sha256
from railtsv.engine import predict_image
from railtsv.model import build_model,load_checkpoint
from railtsv.drafts import draft_annotation


def targets(batch,limit):
    rows=[]
    for lot in sorted(batch.glob('lot_*')):
        for row in json.loads((lot/'manifest.json').read_text())['images']:
            rows.append((lot,row))
    if not limit:return rows
    # A pilot sees all sources before filling an entire batch with suggestions.
    groups={}
    for item in rows:groups.setdefault(item[1]['sequence'],[]).append(item)
    chosen=[]
    while len(chosen)<limit and any(groups.values()):
        for values in groups.values():
            if values and len(chosen)<limit:chosen.append(values.pop(0))
    return chosen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,default=active_batch(ROOT))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--device',choices=['auto','mps','cpu'],default='auto')
    p.add_argument('--limit',type=int,default=0,help='0=all images; 9=a pilot across sources')
    p.add_argument('--apply',action='store_true',help='Fill only absent JSON files with reviewed=false')
    args=p.parse_args()
    if args.limit<0:p.error('limit >= 0')
    task=json.loads((args.batch/'preparation.json').read_text()).get('task','tsv')
    if task not in ('tsv','corridor'):raise ValueError('Type de lot incompatible')
    checkpoint=args.checkpoint
    if checkpoint is None:
        corridor_models=list(ROOT.glob('runs/entrainement_couloir_*/modele/best.pt')) if task=='corridor' else []
        candidates=corridor_models or [*(p for p in ROOT.glob('runs/*/modele/best.pt')
            if not p.parent.parent.name.startswith('entrainement_couloir_')),
            *ROOT.glob('runs/essai_10_images_*/modele/last.pt')]
        if not candidates:raise ValueError('Entraîner un premier modèle avant de proposer les contours')
        checkpoint=max(candidates,key=lambda v:v.stat().st_mtime_ns)
    cp=load_checkpoint(checkpoint)
    source_task=cp.get('task')
    if source_task not in ('tsv','corridor') or cp['classes']!=TASKS[source_task] or cp.get('synthetic_demo'):
        raise ValueError('Utiliser un modèle de rails ou de couloir entraîné sur des images réelles')
    if task=='tsv' and source_task!='tsv':raise ValueError('Le modèle du couloir ne fournit pas les surfaces métalliques des rails')
    torch.set_num_threads(2);dev=device(args.device)
    model=build_model(len(cp['classes']));model.load_state_dict(cp['state_dict']);model.to(dev).eval()
    digest=sha256(checkpoint)
    suffix=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    out=new_dir(ROOT/'runs'/('propositions_annotation_'+suffix));rows=[]
    for lot,row in targets(args.batch,args.limit):
        path=lot/'images'/(row['stem']+'.png');human=lot/'json'/(row['stem']+'.json')
        if human.exists():
            rows.append(dict(stem=row['stem'],lot=lot.name,status='existing_json_preserved'));continue
        rgb=rgb_read(path);mask,prob=predict_image(model,rgb,dev,cp['height'],cp['width'])
        data=draft_annotation(mask,prob,path.name,digest,task=source_task)
        if data and task=='corridor' and source_task=='tsv':
            from scripts.prepare_corridor_annotations import convert_rails_annotation
            data=convert_rails_annotation(data,(rgb.shape[1],rgb.shape[0]))
            if not any(shape['label']=='couloir' for shape in data['shapes']):data=None
        view=rgb.copy()
        if data:
            for shape in data['shapes']:
                color=(0,220,255) if shape['label'] in ('rail_left','couloir') else (255,150,0)
                cv2.polylines(view,[np.rint(shape['points']).astype(np.int32)],True,color,3)
            dump(out/(row['stem']+'.json'),data)
            if args.apply:
                # Exclusive creation is essential even if another annotator saves meanwhile.
                with human.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
        preview=Image.fromarray(view);preview.thumbnail((960,960));preview.save(out/(row['stem']+'.jpg'))
        rows.append(dict(stem=row['stem'],lot=lot.name,sequence=row['sequence'],
            status='suggested' if data else 'manual_required',applied=bool(data and args.apply),
            polygons=len(data['shapes']) if data else 0))
        display='Proposition à corriger' if data else 'À tracer manuellement'
        print(f'{len(rows)} : {row["stem"]} — {display}',flush=True)
    dump(out/'rapport.json',dict(batch_id=args.batch.name,checkpoint_sha256=digest,device=str(dev),
        task=task,source_model_task=source_task,applied=args.apply,images=rows,
        note='Model suggestions are unreviewed, even with a high score; any rail-to-corridor conversion is a draft'))
    print('Suggestions à vérifier : '+str(out),flush=True)
    return out

if __name__=='__main__':main()
