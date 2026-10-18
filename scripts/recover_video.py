"""Récupérer les prédictions d'une vidéo dont les horodatages ont des intervalles manquants."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from railtsv.video import recover_video

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keep-audio',action='store_true',help='Conserver le son ; vidéo muette par défaut')
    for name in ['input','annotated','out']:
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    recover_video(args.input,args.annotated,args.out,args.keep_audio)
