"""Evaluate the existing pilot and create its readable validation report."""
from pathlib import Path
from datetime import datetime
import json,subprocess,sys,uuid
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    config=json.loads((ROOT/'configs/diagnostic_target.json').read_text())
    run=(ROOT/config['run']).resolve()
    if not run.is_relative_to(ROOT/'runs'):raise ValueError('Modèle local dans runs requis')
    out=ROOT/'runs'/('figures_modele_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    subprocess.run([sys.executable,str(ROOT/'scripts/diagnostic_plots.py'),'--run',str(run),'--scope','validation','--out',str(out)],cwd=ROOT,check=True)
    command=[sys.executable,str(ROOT/'scripts/summarize_validation.py'),'--out',str(out)]
    baseline=config.get('baseline_validation')
    if baseline:
        path=(ROOT/baseline).resolve()
        if not path.is_relative_to(ROOT/'runs'):raise ValueError('Référence précédente dans runs requise')
        command+=['--baseline',str(path)]
    note=config.get('validation_annotation_note')
    if note:command+=['--annotation-note',note]
    subprocess.run(command,cwd=ROOT,check=True)
