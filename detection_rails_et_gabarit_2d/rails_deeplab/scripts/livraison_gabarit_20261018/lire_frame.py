"""Extraire une frame du nouveau CSV de points, sans bibliothèque externe."""
import argparse,csv,gzip,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('--frame',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();op=gzip.open if a.input.suffix=='.gz' else open
    found=[]
    with op(a.input,'rt',encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            i=int(r['frame_index'])
            if i==a.frame:found.append(r)
            if i>a.frame:break
    if not found:raise SystemExit('Frame absente du CSV')
    with a.out.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(found[0]));w.writeheader();w.writerows(found)
    r=found[0]
    print(json.dumps(dict(frame=a.frame,lecture_s=r['video_time_s'],temps_reel_s=r['real_elapsed_s'],
        vitesse_kmh=r['train_speed_km_h'],points_observes=sum(bool(r['raw_x_px']) for r in found),
        note='Champs vides = indisponibles. Ne pas raccorder deux segments distincts.'),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
