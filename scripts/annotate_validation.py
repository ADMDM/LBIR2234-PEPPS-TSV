"""Open validation annotations for the active task, keeping training separate."""
from pathlib import Path
import csv
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from railtsv.annotation_config import active_validation,active_task,active_batch

if __name__=='__main__':
    task=active_task(ROOT);session=active_validation(ROOT)
    preparation=__import__('json').loads((active_batch(ROOT)/'preparation.json').read_text())
    train_sequences=set(preparation.get('train_sequences',[preparation['sequence']]))
    view=ROOT/'.local'/('validation_images_'+session.parent.name+'_'+task)
    view.mkdir(parents=True,exist_ok=True)
    with (session/'sequences.csv').open(newline='') as f:
        rows=[row for row in csv.DictReader(f) if row['sequence'] not in train_sequences]
    expected={row['stem']+'.png' for row in rows}
    if not expected:raise ValueError('Images de validation introuvables')
    if {p.name for p in view.iterdir()}-expected:raise ValueError('Vue de validation avec fichiers inattendus')
    for name in sorted(expected):
        source=(session/'images'/name).resolve();dest=view/name
        if not source.is_file():raise FileNotFoundError(source)
        if dest.is_symlink():
            if dest.resolve()!=source:raise ValueError('Lien de validation incorrect')
        elif dest.exists():raise FileExistsError(dest)
        else:dest.symlink_to(source)
    config=session/'labelme.yaml'
    if not config.is_file():config=ROOT/'configs'/('labelme_corridor.yaml' if task=='corridor' else 'labelme.yaml')
    subprocess.run([str(ROOT/'.labelme-venv/bin/python'),'-m','labelme',str(view),'--output',str(session/'json'),
        '--config',str(config),'--labels',str(session/'labels.txt'),
        '--validate-label','exact','--no-auto-save','--no-sort-labels'],check=True)
