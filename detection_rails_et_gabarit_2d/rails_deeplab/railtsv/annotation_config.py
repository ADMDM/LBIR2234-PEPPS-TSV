"""Portable pointer to the active annotation batch, with legacy fallback."""
import json
from pathlib import Path

def _data_path(root, value):
    path=(root/value).resolve()
    if not path.is_relative_to(root/'data'):raise ValueError('La séance d’annotation doit être dans data/')
    return path

def active_batch(root):
    root=Path(root).resolve();path=root/'configs/annotation_batch.json'
    if not path.exists():return root/'data/annotation_team_v2_50'
    value=json.loads(path.read_text())['batch']
    return _data_path(root,value)


def active_task(root):
    root=Path(root).resolve();batch=active_batch(root)
    preparation=json.loads((batch/'preparation.json').read_text())
    task=preparation.get('task','tsv')
    if task not in ('tsv','corridor'):raise ValueError('Mode d’annotation actif inconnu')
    pointer=root/'configs/annotation_batch.json'
    if pointer.exists():
        declared=json.loads(pointer.read_text()).get('task',task)
        if declared!=task:raise ValueError('Le mode actif contredit la préparation du lot')
    return task


def active_validation(root):
    root=Path(root).resolve();batch=active_batch(root)
    preparation=json.loads((batch/'preparation.json').read_text())
    value=preparation.get('validation_session')
    if value:return _data_path(root,value)
    if active_task(root)=='corridor':raise ValueError('Séance de validation couloir manquante')
    return root/'data/annotation_videos_v1'


def latest_returns(root, batch):
    """Newest complete return folder per lot, filtered by capture batch first."""
    root=Path(root).resolve();batch=Path(batch).resolve();selected={}
    for lot in sorted(batch.glob('lot_*')):
        lot_id=lot.name.removeprefix('lot_')
        candidates=[]
        for path in (root/'data/retours').glob(f'lot_{lot_id}_*/json'):
            receipt=path.parent/'reception.json'
            if not receipt.is_file():continue
            data=json.loads(receipt.read_text())
            if data.get('batch_id')!=batch.name or data.get('lot_id',lot_id)!=lot_id:continue
            candidates.append(path)
        if candidates:selected[lot_id]=max(candidates,key=lambda p:p.parent.stat().st_mtime_ns)
    return selected
