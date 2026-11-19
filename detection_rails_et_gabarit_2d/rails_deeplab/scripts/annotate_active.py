"""Open an assigned lot from the active annotation batch."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from railtsv.annotation_config import active_batch

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lot',choices=['A','B','C'],default='A')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args();lot=active_batch(ROOT)/('lot_'+args.lot)
    cmd=[sys.executable,str(lot/'start_annotation.py')]
    if args.check_only:cmd.append('--check-only')
    subprocess.run(cmd,check=True,cwd=ROOT)
