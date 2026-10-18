"""Extract one synchronized frame for fusion, using only Python's standard library."""
import argparse,csv,gzip,json
from pathlib import Path

def extract_frame(input_path,frame_index):
    p=Path(input_path);opener=gzip.open if p.suffix=='.gz' else open
    with opener(p,'rt',encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        for row in reader:
            i=int(row['frame_index'])
            if i==frame_index:yield row
            elif i>frame_index:break

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path,help='AXES_SYNCHRONISES.csv.gz')
    parser.add_argument('--frame',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();rows=list(extract_frame(args.input,args.frame))
    if not rows:raise ValueError('Frame absente du fichier')
    with args.out.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(frame=args.frame,time_s=rows[0]['time_s'],rows=len(rows),
        ego_observed_points=sum(r['ego_x_px']!='' for r in rows),
        note='Empty coordinate = unavailable. Tangents face decreasing y. Errors require ground truth.'),indent=2))

if __name__=='__main__':main()
