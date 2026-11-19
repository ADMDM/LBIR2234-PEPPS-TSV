"""Show the current scientific figure folder from the verified project pointer."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
config=json.loads((ROOT/'configs/diagnostic_target.json').read_text())
folder=(ROOT/config.get('latest_figures',config['training_figures'])/'figures').resolve()
if not folder.is_relative_to(ROOT/'runs') or not folder.is_dir():raise ValueError('Dossier de figures locales introuvable')
print(folder)
if sys.platform=='darwin':subprocess.run(['open',str(folder)],check=True)
