"""Revoir les dix anciens tracés convertis, comme complément facultatif."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from railtsv.annotation_config import active_batch,active_task

if __name__=='__main__':
    batch=active_batch(ROOT)
    if active_task(ROOT)!='corridor':raise ValueError('Complément réservé au mode couloir')
    value=json.loads((batch/'preparation.json').read_text()).get('supplemental_session')
    if not value:raise ValueError('Aucune séance complémentaire préparée')
    session=(ROOT/value).resolve()
    if not session.is_relative_to(ROOT/'data'):raise ValueError('Séance complémentaire hors de data/')
    subprocess.run([str(ROOT/'.labelme-venv/bin/python'),'-m','labelme',str(session/'images'),
        '--output',str(session/'json'),'--config',str(session/'labelme.yaml'),
        '--labels',str(session/'labels.txt'),'--validate-label','exact','--no-auto-save','--no-sort-labels'],check=True)
