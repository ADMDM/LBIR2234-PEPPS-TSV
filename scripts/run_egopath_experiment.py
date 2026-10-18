"""Launch an isolated, reproducible ego-path transfer experiment."""
from pathlib import Path
import csv, datetime, json, os, shutil, subprocess, sys, traceback

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'railtsv').is_dir())
BASE = ROOT/'runs/entrainement_couloir_20261009_143754_101cbf'
GENERIC = ROOT/'runs/preapprentissage_railsem19_20261007_172310_ca812b/modele/best.pt'
EGO = ROOT/'data/prepared/railsem_egopath_20261014'
PY = ROOT/'.venv/bin/python'
sys.path.insert(0, str(ROOT))
from railtsv.common import dump, sha256, read_manifest, write_manifest, check_disjoint

def main():
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT/'runs'/('experience_egopath_'+stamp)
    out.mkdir(exist_ok=False)
    (out/'logs').mkdir()
    (out/'dataset').mkdir()
    shutil.copy2(__file__,out/'REPRODUIRE.py')
    protocol = dict(created=datetime.datetime.now().isoformat(),baseline=str(BASE),
        baseline_checkpoint_sha256=sha256(BASE/'modele/best.pt'),
        generic_checkpoint_sha256=sha256(GENERIC),ego_preparation=json.loads((EGO/'preparation.json').read_text()),
        experiment='Generic RailSem19 -> ego-path RailSem19 -> TSV corridor',
        ego=dict(train=2796,validation=466,epochs=2,batch_size=2,height=384,width=640,lr=.0001,seed=42),
        tsv=dict(train=130,validation=12,epochs=20,batch_size=1,height=480,width=640,lr=.0001,seed=42),
        comparison='Same frozen TSV train and validation data, augmentation, seed, fine-tuning budget and image geometry as baseline. Additional public supervised pretraining is the intended difference.',
        checkpoint_selection='Foreground Dice on respective development validation sets. TSV validation is not an untouched test.',
        public_test_used=False,new_pending_annotations_used=False,
        pretrained_TEP_Net_weights_used=False,metric_calibration=None,
        limitations=['One TSV validation capture, 12 images, correlated sections.',
                    'Ego-path public boundaries need not be TSV inner rail faces.',
                    'RailSem19 official split not established source-video-disjoint.',
                    'Single seed, exploratory training budget, no causal significance claim.',
                    'No independent camera or physical gauge calibration.'])
    dump(out/'PROTOCOLE.json',protocol)
    for split,n in [('train',130),('val',12)]:
        rows=read_manifest(BASE/'dataset'/f'{split}.csv')
        assert len(rows)==n
        write_manifest(out/'dataset'/f'{split}.csv',rows)
    check_disjoint(read_manifest(out/'dataset/train.csv'),read_manifest(out/'dataset/val.csv'))
    dump(out/'ETAT.json',dict(status='preparing',started=stamp))
    print('EXPERIMENT='+str(out),flush=True)
    def execute(stage,args):
        dump(out/'ETAT.json',dict(status='running',stage=stage,updated=datetime.datetime.now().isoformat()))
        cmd=[str(PY),'-u',*map(str,args)]
        dump(out/'logs'/f'{stage}_commande.json',cmd)
        print('START '+stage,flush=True)
        with (out/'logs'/f'{stage}.log').open('x') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
                           env={**os.environ,'PYTHONUNBUFFERED':'1','NO_ALBUMENTATIONS_UPDATE':'1'})
        print('DONE '+stage,flush=True)
    try:
        execute('01_egopath',['scripts/train.py','--train',EGO/'train.csv','--val',EGO/'val.csv',
            '--out',out/'preapprentissage_egopath','--task','railsem_egopath','--init',GENERIC,
            '--height','384','--width','640','--epochs','2','--batch-size','2','--seed','42',
            '--lr','0.0001','--device','mps','--progress-every','50'])
        execute('02_adaptation_tsv',['scripts/train.py','--train',out/'dataset/train.csv','--val',out/'dataset/val.csv',
            '--out',out/'modele','--task','corridor','--init',out/'preapprentissage_egopath/best.pt',
            '--height','480','--width','640','--epochs','20','--batch-size','1','--seed','42',
            '--lr','0.0001','--device','mps','--progress-every','25'])
        execute('03_evaluation',['scripts/evaluate.py','--checkpoint',out/'modele/best.pt',
            '--manifest',out/'dataset/val.csv','--out',out/'evaluation_validation','--device','mps'])
        dump(out/'ETAT.json',dict(status='training_and_validation_complete',updated=datetime.datetime.now().isoformat(),
            next='Matched comparative videos and detailed comparison.'))
    except BaseException as e:
        dump(out/'ETAT.json',dict(status='failed',error=str(e),traceback=traceback.format_exc()))
        raise

if __name__=='__main__': main()
