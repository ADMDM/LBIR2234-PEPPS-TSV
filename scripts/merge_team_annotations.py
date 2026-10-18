"""Validate ownership and merge returned Labelme lots into a NEW session."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from railtsv.common import dump,new_dir,sha256
from scripts.export_labelme import rasterize
from scripts.prepare_team_annotations import write_csv


def inspect(batch,returned=None):
    batch=Path(batch).resolve()
    returned=returned or {}
    seen=set();items=[];report={}
    preparation=json.loads((batch/'preparation.json').read_text())
    task=preparation.get('task','tsv')
    lot_ids=sorted(p.name.removeprefix('lot_') for p in batch.glob('lot_*') if (p/'manifest.json').is_file())
    if len(lot_ids)<2:raise ValueError('Au moins deux lots attribués sont requis')
    for lot_id in lot_ids:
        lot=batch/f'lot_{lot_id}'
        manifest=json.loads((lot/'manifest.json').read_text())
        if manifest['lot_id']!=lot_id or manifest['batch_id']!=batch.name:
            raise ValueError('Manifest de lot incompatible')
        if manifest.get('task','tsv')!=task:
            raise ValueError('Modes d’annotation incompatibles entre le lot et la préparation')
        rows=manifest['images']
        expected={r['stem']:r for r in rows}
        if len(expected)!=len(rows) or seen & set(expected):
            raise ValueError('Attribution en double entre les lots')
        seen.update(expected)
        incoming=Path(returned.get(lot_id,lot/'json')).resolve()
        if (incoming/'json').is_dir():incoming=incoming/'json'
        if not incoming.is_dir():raise FileNotFoundError(incoming)
        annotations={p.stem:p for p in incoming.glob('*.json')}
        unknown=set(annotations)-set(expected)
        if unknown:raise ValueError(f'Lot {lot_id} : fichiers non attribués {sorted(unknown)}')
        invalid={};valid=0
        for stem,path in annotations.items():
            row=expected[stem]
            canonical=lot/'images'/f'{stem}.png'
            if row['assignee']!=f'lot_{lot_id}' or sha256(canonical)!=row['image_sha256']:
                raise ValueError('Image canonique ou attribution modifiée')
            try:
                data=json.loads(path.read_text())
                with Image.open(canonical) as im:rasterize(data,im.size,task=task)
            except (ValueError,KeyError,TypeError) as error:
                invalid[stem]=str(error)
                continue
            items.append((lot_id,row,canonical,path,data));valid+=1
        report[lot_id]=dict(expected=len(rows),received=len(annotations),reviewed_valid=valid,
                            missing=sorted(set(expected)-set(annotations)),invalid=invalid)
    return items,report


def merge(batch,output,returned=None,allow_incomplete=False):
    batch=Path(batch).resolve()
    preparation=json.loads((batch/'preparation.json').read_text())
    task=preparation.get('task','tsv')
    items,report=inspect(batch,returned)
    if not allow_incomplete and any(r['missing'] or r['invalid'] for r in report.values()):
        raise ValueError('Tous les lots doivent être complets et reviewed : '+json.dumps(report,ensure_ascii=False))
    if not items:raise ValueError('Aucune annotation reviewed valide à réunir')
    out=new_dir(output)
    (out/'images').mkdir();(out/'json').mkdir()
    rows=[];provenance=[]
    for lot_id,row,image,path,data in sorted(items,key=lambda item:item[1]['stem']):
        shutil.copy2(image,out/'images'/image.name)
        received=out/'received'/f'lot_{lot_id}'
        received.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,received/path.name)
        portable=dict(data,imagePath='../images/'+image.name,imageData=None)
        dump(out/'json'/path.name,portable)
        rows.append(row)
        provenance.append(dict(stem=row['stem'],lot_id=lot_id,returned_json_sha256=sha256(path),
                               merged_json_sha256=sha256(out/'json'/path.name),image_sha256=row['image_sha256']))
    write_csv(out/'sequences.csv',rows,['stem','sequence'])
    write_csv(out/'sources.csv',rows,['stem','sequence','source_frame_index','seconds','image_sha256'])
    (out/'labels.txt').write_text('couloir\nignore\n' if task=='corridor' else 'rail_left\nrail_right\nignore\n')
    videos=preparation.get('sources') or [dict(sequence=preparation['sequence'],
        video=os.path.relpath(preparation['source'],out),sha256=preparation['source_sha256'])]
    dump(out/'session.json',dict(task=task,images=len(items),videos=videos,
        grouping='Keep original capture groups; assignment lots are not independent sequences',
        annotation_status='Reviewed polygon checks passed; visual quality still needs review'))
    dump(out/'merge.json',dict(task=task,source_batch=str(batch),lots=report,images=len(items),provenance=provenance,
                             note='Sources and original returned JSON preserved; no overwrite'))
    print(f'{len(items)} annotations réunies dans {out}',flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',required=True)
    parser.add_argument('--lot-a',help='Dossier json reçu du lot A, sinon copie locale')
    parser.add_argument('--lot-b',help='Dossier json reçu du lot B, sinon copie locale')
    parser.add_argument('--lot-c',help='Dossier json reçu du lot C, sinon copie locale')
    parser.add_argument('--out')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args()
    returned={k:v for k,v in [('A',args.lot_a),('B',args.lot_b),('C',args.lot_c)] if v}
    if args.check_only:
        _,report=inspect(args.batch,returned)
        print(json.dumps(report,indent=2,ensure_ascii=False))
    else:
        if not args.out:parser.error('--out requis pour une nouvelle session')
        merge(args.batch,args.out,returned,args.allow_incomplete)
