"""Portable launcher for one assigned Labelme lot (macOS, Windows, Linux)."""
from pathlib import Path
import argparse
import subprocess
import sys
import venv


def labelme_python(root):
    relative = 'Scripts/python.exe' if sys.platform=='win32' else 'bin/python'
    # Reuse the project's existing annotation environment on the owner's Mac.
    for parent in [root,*root.parents][:6]:
        candidate=parent/'.labelme-venv'/relative
        if candidate.is_file():return candidate
    candidate=root/'.annotation-env'/relative
    if candidate.is_file():return candidate
    if not (3,12) <= sys.version_info[:2] <= (3,14):
        raise RuntimeError('Installer Python 3.12, 3.13 ou 3.14 pour ouvrir ce lot.')
    env=root/'.annotation-env'
    venv.EnvBuilder(with_pip=True).create(env)
    return env/relative


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    python=labelme_python(root)
    check=subprocess.run([str(python),'-m','labelme','--version'],capture_output=True,text=True)
    if check.returncode:
        if args.check_only:raise RuntimeError('Labelme absent : lancer ANNOTER pour l’installation initiale.')
        print('Installation initiale de Labelme 7.7.0 dans ce lot…',flush=True)
        subprocess.run([str(python),'-m','pip','install','labelme==7.7.0'],check=True)
    if args.check_only:
        print(f'Labelme disponible : {python}')
        return
    subprocess.run([str(python),'-m','labelme',str(root/'images'),'--output',str(root/'json'),
                    '--config',str(root/'labelme.yaml'),'--labels',str(root/'labels.txt'),
                    '--validate-label','exact','--no-auto-save','--no-sort-labels'],check=True)


if __name__=='__main__':
    try:main()
    except (RuntimeError,subprocess.CalledProcessError) as error:
        print(f'Ouverture impossible : {error}',file=sys.stderr)
        sys.exit(1)
