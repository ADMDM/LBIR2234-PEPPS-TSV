"""Clone assigned rail lots into unreviewed corridor suggestions, without editing sources."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
from railtsv.common import dump,new_dir,sha256
from scripts.export_labelme import rasterize
from scripts.prepare_team_annotations import make_lot_files,write_csv


def convert_rails_annotation(data,size):
    """Return a Labelme corridor DRAFT from observed rail inner pixel boundaries.

    Missing/ambiguous rows remain empty; gaps and ignore areas are not bridged.
    Temporary reviewed=True enables internal rasterization only, never acceptance
    of source predictions as ground truth. Original data is not mutated.
    """
    source=copy.deepcopy(data)
    source.setdefault('flags',{})['reviewed']=True
    rails=rasterize(source,size,task='tsv')
    corridor=np.zeros(rails.shape,np.uint8)
    for y,row in enumerate(rails):
        left=np.flatnonzero(row==1);right=np.flatnonzero(row==2)
        if not len(left) or not len(right):continue
        if np.any(np.diff(left)>1) or np.any(np.diff(right)>1):continue
        x0=int(left[-1]);x1=int(right[0])
        if x1<=x0:continue
        corridor[y,x0:x1+1]=255
    corridor[rails==255]=0
    contours,_=cv2.findContours(corridor,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    shapes=[]
    for contour in sorted(contours,key=lambda c:(int(c[:,:,1].min()),int(c[:,:,0].min()))):
        if cv2.contourArea(contour)<2:continue
        points=cv2.approxPolyDP(contour,.75,True).reshape(-1,2).tolist()
        if len(points)<3:continue
        shapes.append(dict(label='couloir',points=points,group_id=None,shape_type='polygon',flags={},
                           description='Suggestion depuis les bords intérieurs observés ; vérifier toute la zone.'))
    shapes.extend(copy.deepcopy(shape) for shape in data.get('shapes',[]) if shape.get('label')=='ignore')
    flags=dict(data.get('flags',{}),reviewed=False)
    return dict(version=data.get('version','7.7.0'),task='corridor',flags=flags,shapes=shapes,
                imagePath=data.get('imagePath'),imageData=None,imageHeight=size[1],imageWidth=size[0],
                description='PROPOSITION non vérifiée : conversion rail_left/rail_right vers couloir. '
                            'Limites intérieures, arrêt à la dernière portion où les deux rails sont observés.')


def _link_image(source,target):
    if not source.is_file():raise FileNotFoundError(source)
    target.symlink_to(os.path.relpath(source.resolve(),target.parent))


def _clone_session(root,source,dest,stems,task='corridor'):
    for name in ('images','json'):(dest/name).mkdir(parents=True)
    with (source/'sequences.csv').open(newline='') as handle:
        rows=[row for row in csv.DictReader(handle) if row['stem'] in stems]
    if {row['stem'] for row in rows}!=set(stems):raise ValueError('Séance source incomplète')
    conversions=[]
    for row in rows:
        image=source/'images'/(row['stem']+'.png');_link_image(image,dest/'images'/image.name)
        path=source/'json'/(row['stem']+'.json')
        if path.exists():
            with Image.open(image) as im:data=convert_rails_annotation(json.loads(path.read_text()),im.size)
            data['imagePath']='../images/'+image.name
            dump(dest/'json'/path.name,data)
            conversions.append(dict(stem=row['stem'],source_json_sha256=sha256(path),reviewed=False))
    write_csv(dest/'sequences.csv',rows,['stem','sequence'])
    (dest/'labels.txt').write_text('couloir\nignore\n',encoding='utf-8')
    shutil.copy2(root/'configs/labelme_corridor.yaml',dest/'labelme.yaml')
    dump(dest/'session.json',dict(task=task,images=len(rows),source_session=str(source),conversions=conversions,
        note='Images linked read-only by convention; JSON distinct. Every converted annotation requires human review.'))
    return len(rows)


def prepare(root=ROOT,source=None,output=None):
    root=Path(root).resolve()
    source=Path(source or root/'data/annotation_team_v3_150_20261005_final').resolve()
    output=Path(output or root/'data/annotation_corridor_v1_150_20261006').resolve()
    if output.exists():raise FileExistsError(output)
    if not output.is_relative_to(root/'data') or source==output:raise ValueError('Nouveau lot requis dans data/')
    preparation=json.loads((source/'preparation.json').read_text())
    if preparation.get('task','tsv')!='tsv':raise ValueError('La source doit contenir les deux rails TSV')
    manifests={}
    seen=set()
    for lot_id in 'ABC':
        manifest=json.loads((source/f'lot_{lot_id}/manifest.json').read_text())
        rows=manifest['images']
        if manifest['batch_id']!=source.name or manifest['lot_id']!=lot_id or len(rows)!=50:
            raise ValueError('Trois lots de 50 images, attribués sans ambiguïté, requis')
        stems={row['stem'] for row in rows}
        if len(stems)!=50 or seen&stems:raise ValueError('Attribution en double')
        seen|=stems;manifests[lot_id]=manifest
    originals={path:sha256(path) for path in source.glob('lot_*/json/*.json')}
    old=root/'data/annotation_videos_v1'
    originals.update({path:sha256(path) for path in (old/'json').glob('*.json')})
    out=new_dir(output);rows_all=[];conversions=[]
    for lot_id,manifest in manifests.items():
        lot=out/f'lot_{lot_id}';(lot/'images').mkdir(parents=True)
        rows=copy.deepcopy(manifest['images'])
        make_lot_files(lot,lot_id,rows,out.name,task='corridor')
        for row in rows:
            image=source/f'lot_{lot_id}/images'/(row['stem']+'.png')
            if sha256(image)!=row['image_sha256']:raise ValueError('Image source modifiée : '+image.name)
            _link_image(image,lot/'images'/image.name)
            path=source/f'lot_{lot_id}/json'/(row['stem']+'.json')
            if path.exists():
                with Image.open(image) as im:data=convert_rails_annotation(json.loads(path.read_text()),im.size)
                data['imagePath']='../images/'+image.name
                dump(lot/'json'/path.name,data)
                conversions.append(dict(stem=row['stem'],lot_id=lot_id,source_json_sha256=originals[path],
                                        polygons=sum(s['label']=='couloir' for s in data['shapes']),reviewed=False))
        rows_all+=rows
    with (old/'sequences.csv').open(newline='') as handle:old_rows=list(csv.DictReader(handle))
    train_sequences=set(preparation.get('train_sequences',[preparation['sequence']]))
    heldout={row['stem'] for row in old_rows if row['sequence'] not in train_sequences}
    training_old={row['stem'] for row in old_rows if row['sequence'] in train_sequences and (old/'json'/(row['stem']+'.json')).exists()}
    if len(heldout)!=12:raise ValueError('Les 12 images de la seconde capture sont requises pour la validation')
    _clone_session(root,old,out/'validation',heldout)
    _clone_session(root,old,out/'supplemental',training_old)
    report=copy.deepcopy(preparation)
    report.update(task='corridor',batch_id=out.name,source_batch=str(source),images=rows_all,
        validation_session=str((out/'validation').relative_to(root)),
        supplemental_session=str((out/'supplemental').relative_to(root)),
        annotation_status='Unreviewed suggestions only; human review required for all150 +12validation',
        conversions=conversions,storage='Image symlinks; portable ZIP must dereference them',
        supplemental_required=False)
    dump(out/'preparation.json',report)
    write_csv(out/'repartition.csv',rows_all,['stem','assignee','sequence','category','image_sha256'])
    if any(sha256(path)!=digest for path,digest in originals.items()):raise RuntimeError('Annotation source modifiée')
    dump(out/'sources_preserved.json',dict(files_checked=len(originals),json_unchanged=True,
        all_canonical_image_hashes_verified=True,active_pointer_modified=False))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    result=prepare(source=args.source,output=args.out)
    print(f'150 images couloir prêtes ; {len(result["conversions"])} suggestions non reviewed. Pointeur actif inchangé.')
