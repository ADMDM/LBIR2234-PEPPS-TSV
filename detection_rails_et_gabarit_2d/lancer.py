"""One documented entry point; configuration contains paths, never shell code.

Examples: python lancer.py --config ../configurations/projet.local.json --stage check
          python lancer.py --config ../configurations/projet.local.json --stage predictions
          python lancer.py --config ../configurations/projet.local.json --stage presentation
Completed stages may be reused only when their source hashes still match.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def execute(config, stage, selected=None):
    config = Path(config).resolve()
    cfg = json.loads(config.read_text())
    if cfg.get('schema_version') != 1: raise ValueError('schema_version=1 required')
    def path(v):
        p = Path(v).expanduser()
        return p if p.is_absolute() else (config.parent/p).resolve()
    def python_command(key):
        command = [str(path(cfg.get(key, sys.executable)))]
        if cfg.get('python_arch'):
            if sys.platform != 'darwin' or cfg['python_arch'] not in ('arm64', 'x86_64'):
                raise ValueError('Explicit architecture is only supported on macOS')
            command = ['/usr/bin/arch', '-'+cfg['python_arch']] + command
        return command
    jobs = [v for v in cfg['videos'] if not selected or v['video_id'] in selected]
    if selected and len(jobs) != len(set(selected)): raise ValueError('Unknown video ID')
    for job in jobs:
        src = path(job['source'])
        if not src.is_file() or sha(src) != job['source_sha256']:
            raise ValueError('Missing or changed source: '+job['video_id'])
        if stage == 'check':
            print('Source vérifiée : '+job['video_id'], flush=True)
            continue
        if stage in ('predictions', 'all') and job.get('new_predictions'):
            for name, marker in [('rails', 'SOURCE_ET_PARAMETRES.json'), ('objects', 'MANIFESTE.json')]:
                out = path(job[name+'_result'])
                if out.exists():
                    m = json.loads((out/marker).read_text())
                    if m.get('status') != 'complete' or m.get('source_sha256') != job['source_sha256']:
                        raise ValueError('Existing stage is incomplete or has another source: '+str(out))
                    if name == 'rails' and m.get('checkpoint_sha256') != sha(path(cfg['rail_checkpoint'])):
                        raise ValueError('Existing rail stage used another checkpoint')
                    if name == 'objects':
                        for filename, digest in m.get('inputs_sha256', {}).items():
                            if sha(filename) != digest:
                                raise ValueError('An input of the completed object stage has changed: '+filename)
                    print('Étape déjà terminée : '+str(out), flush=True)
                    continue
                if name == 'rails':
                    cmd = [str(ROOT/'rails_deeplab/presentation/infer_rails.py'), '--source', str(src), '--checkpoint', str(path(cfg['rail_checkpoint'])), '--out', str(out), '--video-id', job['video_id']]
                else:
                    cmd = [str(ROOT/'gabarit_2d/infer_gabarit_2d.py'), '--rail-delivery', str(path(job['rails_result'])), '--public-model', str(path(cfg['public_objects_checkpoint'])), '--litter-model', str(path(cfg['litter_checkpoint'])), '--out', str(out), '--public-imgsz', '960', '--litter-imgsz', '960', '--public-conf', '0.25', '--litter-conf', '0.15']
                out.parent.mkdir(exist_ok=True, parents=True)
                with out.with_suffix('.log').open('x') as log:
                    print('Calcul : '+job['video_id']+' / '+name, flush=True)
                    subprocess.run(python_command('inference_python') + cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        if stage in ('presentation', 'all'):
            cmd = python_command('presentation_python') + [str(ROOT/'rails_deeplab/presentation/build_video.py'), '--config', str(config), '--video-id', job['video_id']]
            subprocess.run(cmd, check=True)
    return {'status': 'complete', 'stage': stage, 'videos': [j['video_id'] for j in jobs]}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--stage', choices=['check', 'predictions', 'presentation', 'all'], default='all')
    p.add_argument('--video-id', action='append')
    a = p.parse_args()
    print(json.dumps(execute(a.config, a.stage, a.video_id), indent=2, ensure_ascii=False))
