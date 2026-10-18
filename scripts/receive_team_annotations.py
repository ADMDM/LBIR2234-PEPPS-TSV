"""Receive a collaborator's JSON archive without extracting arbitrary ZIP paths."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import sys
import uuid
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from railtsv.common import dump,new_dir
from scripts.merge_team_annotations import inspect,merge
from railtsv.annotation_config import active_batch,latest_returns


def returned_json(incoming,expected):
    incoming=Path(incoming).resolve()
    files={}
    if incoming.is_file() and incoming.suffix.lower()=='.zip':
        with zipfile.ZipFile(incoming) as z:
            total=0
            for info in z.infolist():
                path=Path(info.filename)
                if info.is_dir() or '__MACOSX' in path.parts or path.suffix.lower()!='.json':continue
                if path.stem not in expected:
                    raise ValueError(f'Ce fichier ne fait pas partie du lot attribué : {path.name}')
                total+=info.file_size
                if info.file_size>20*1024**2 or total>256*1024**2:
                    raise ValueError('Archive JSON trop volumineuse : renvoyer les annotations sans imageData')
                if path.name in files:raise ValueError('Deux fichiers portent le même nom dans le retour')
                files[path.name]=z.read(info)
    elif incoming.is_dir():
        if (incoming/'json').is_dir():incoming=incoming/'json'
        for path in incoming.glob('*.json'):
            if path.stem not in expected:raise ValueError(f'Fichier non attribué au lot : {path.name}')
            files[path.name]=path.read_bytes()
    else:raise ValueError('Choisir le ZIP du dossier json ou le dossier json décompressé')
    if not files:raise ValueError('Aucune annotation JSON dans ce retour')
    for raw in files.values():
        if not isinstance(json.loads(raw),dict):raise ValueError('Format JSON invalide')
    return files


def receive(batch,incoming,out,lot_id='B'):
    batch=Path(batch).resolve()
    manifest=json.loads((batch/f'lot_{lot_id}/manifest.json').read_text())
    expected={r['stem'] for r in manifest['images']}
    files=returned_json(incoming,expected)
    staged=new_dir(out)
    (staged/'json').mkdir()
    for name,raw in files.items():(staged/'json'/name).write_bytes(raw)
    _,report=inspect(batch,{lot_id:staged/'json'})
    dump(staged/'reception.json',dict(source=str(Path(incoming).resolve()),batch_id=batch.name,lot_id=lot_id,lots=report))
    return staged,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,default=active_batch(ROOT))
    parser.add_argument('--lot',choices=['A','B','C'],default='B')
    parser.add_argument('--incoming',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.check_only:
        _,report=inspect(args.batch)
        print(json.dumps({lot:{k:r[k] for k in ['expected','received','reviewed_valid']} for lot,r in report.items()},indent=2))
        return
    incoming=args.incoming
    if incoming is None:
        print('Glisse ici le ZIP json reçu de ton binôme, puis appuie sur Entrée.',flush=True)
        pieces=shlex.split(input('Fichier ou dossier : ').strip())
        if len(pieces)!=1:raise ValueError('Glisser un seul fichier ZIP ou un seul dossier json')
        incoming=Path(pieces[0])
    suffix=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    staged,report=receive(args.batch,incoming,ROOT/'data/retours'/(f'lot_{args.lot}_'+suffix),args.lot)
    returned=latest_returns(ROOT,args.batch)
    _,report=inspect(args.batch,returned)
    print(f'Retour enregistré : {staged}',flush=True)
    for lot,r in report.items():
        print(f'Lot {lot} : {r["reviewed_valid"]} / {r["expected"]} annotations reviewed valides',flush=True)
        for stem,why in r['invalid'].items():
            if not why.startswith('Annotation non validée'):
                print(f'À vérifier : {stem} — {why}',flush=True)
    if any(r['missing'] or r['invalid'] for r in report.values()):
        print('Retour conservé. Compléter et vérifier tous les lots avant de les réunir.',flush=True)
        return
    combined=ROOT/'data'/('annotations_equipe_reunies_'+suffix)
    merge(args.batch,combined,returned)
    print('Les annotations sont réunies dans une nouvelle session. Les originaux sont conservés.',flush=True)


if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError,zipfile.BadZipFile) as error:
        print(f'Réception interrompue : {error}',file=sys.stderr)
        sys.exit(1)
