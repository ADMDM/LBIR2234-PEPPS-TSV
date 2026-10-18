"""Prepare official RailSem19 paper splits, pretrain and register the checkpoint."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from railtsv.common import dump, sha256
from railtsv.railsem import prepare_track_dataset, write_pretraining_state, PAPER_NOTE


def main():
    from railtsv.settings import processing_resolution
    default_height, default_width = processing_resolution(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['prepare', 'train', 'status'], default='train')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--height', type=int, default=default_height)
    parser.add_argument('--width', type=int, default=default_width)
    parser.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    args = parser.parse_args()
    pointer = ROOT / 'configs/railsem_pretraining.json'
    if args.action == 'status':
        print(pointer.read_text() if pointer.exists() else 'RailSem19 non préparé')
        return
    if args.epochs < 1 or min(args.height, args.width) < 32 or args.height % 32 or args.width % 32:
        parser.error('epochs positif et dimensions multiples de 32')
    lock_path = ROOT / '.local/railsem_pretraining.lock'
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Une préparation ou un apprentissage RailSem19 est déjà en cours')
        dataset = ROOT / 'data/prepared/railsem_track_paper_v1'
        if not dataset.exists():
            prepare_track_dataset(ROOT/'data/raw/railsem19/rs19_val',
                                  ROOT/'data/raw/railsem19/rs19_splits4000', dataset,
                                  expected={'train': 3000, 'val': 500, 'test': 500})
        report = json.loads((dataset/'preparation.json').read_text())
        if report['task'] != 'railsem_track':
            raise ValueError('Préparation RailSem19 incompatible')
        for role, digest in report['manifest_sha256'].items():
            if sha256(dataset/f'{role}.csv') != digest:
                raise ValueError('Manifeste RailSem19 modifié : ' + role)
        if args.action == 'prepare':
            if not pointer.exists():
                write_pretraining_state(ROOT, {'status': 'prepared', 'dataset': str(dataset.relative_to(ROOT)),
                                              'split_note': PAPER_NOTE})
            print('RailSem19 prêt : 3000 train, 500 validation, 500 test réservés', flush=True)
            return
        suffix = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
        run = ROOT/'runs'/('preapprentissage_railsem19_' + suffix)
        run.mkdir()
        state = {'status': 'running', 'run': str(run.relative_to(ROOT)),
                 'dataset': str(dataset.relative_to(ROOT)), 'epochs_requested': args.epochs,
                 'split_note': PAPER_NOTE, 'test_used_for_selection': False}
        write_pretraining_state(ROOT, state)
        dump(run/'railsem19.json', state)
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        command = [sys.executable, str(ROOT/'scripts/train.py'), '--train', str(dataset/'train.csv'),
                   '--val', str(dataset/'val.csv'), '--out', str(run/'modele'),
                   '--task', 'railsem_track', '--pretrained', '--device', args.device,
                   '--height', str(args.height), '--width', str(args.width),
                   '--epochs', str(args.epochs), '--batch-size', '1', '--progress-every', '100']
        print(f'Préapprentissage RailSem19 : {args.epochs} passages sur 3000 images. Validation : 500. Test réservé : 500.', flush=True)
        print(f'Suivi : {run / "APPRENTISSAGE.log"}', flush=True)
        try:
            with (run/'APPRENTISSAGE.log').open('w', buffering=1) as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1, cwd=ROOT)
                for line in process.stdout:
                    log.write(line)
                    print(line, end='', flush=True)
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)
            checkpoint = run/'modele/best.pt'
            if not checkpoint.is_file():
                raise ValueError('Préapprentissage sans checkpoint best.pt')
            state.update(status='completed', checkpoint=str(checkpoint.relative_to(ROOT)),
                         checkpoint_sha256=sha256(checkpoint))
            write_pretraining_state(ROOT, state)
            dump(run/'railsem19.json', state)
            (run/'LIRE_LE_RESULTAT.md').write_text('# Préapprentissage RailSem19 terminé\n\n'
                'Le bouton 05 utilisera ce checkpoint pour initialiser encodeur et décodeur TSV ; la tête est réinitialisée. '
                'Les scores RailSem19 ne mesurent pas la précision sur les vidéos TSV. '
                'Les 500 images test ne servent ni à apprendre ni à choisir ce modèle.\n', encoding='utf-8')
            print('Terminé : RailSem19 est enregistré comme départ du futur entraînement TSV.', flush=True)
        except BaseException as error:
            state.update(status='failed', error=str(error))
            write_pretraining_state(ROOT, state)
            dump(run/'railsem19.json', state)
            raise


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print('RailSem19 interrompu : ' + str(error), file=sys.stderr)
        sys.exit(1)
