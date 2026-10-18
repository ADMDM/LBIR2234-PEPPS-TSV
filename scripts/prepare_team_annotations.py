"""Extract new source images and prepare two disjoint portable Labelme lots."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image
from railtsv.common import dump,new_dir,sha256
from railtsv.video import probe_timing

ROOT=Path(__file__).resolve().parents[1]


def plan_indices(timestamps,count,excluded=()):
    times=np.asarray(timestamps,dtype=float)
    if count < 2 or count%2 or len(times) < count or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Nombre pair et chronologie monotone suffisamment longue requis')
    excluded={int(x) for x in excluded if 0<=int(x)<len(times)}
    span=times[-1]-times[0]
    margin=min(1.,span/(4*count))
    spacing=span/count*.35
    available=np.ones(len(times),bool)
    for i in excluded:available &= np.abs(times-times[i])>margin
    chosen=[]
    for target in np.linspace(times[0]+span/count/2,times[-1]-span/count/2,count):
        candidates=np.flatnonzero(available)
        if not len(candidates):raise ValueError('Pas assez d’images nouvelles après exclusion')
        index=int(candidates[np.argmin(np.abs(times[candidates]-target))])
        chosen.append(index)
        available &= np.abs(times-times[index])>spacing
    return sorted(chosen)


def write_csv(path,rows,fields):
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def make_lot_files(lot,lot_id,rows,batch_id,task='tsv'):
    if task not in ('tsv','corridor'):raise ValueError('Mode d’annotation inconnu')
    (lot/'json').mkdir()
    (lot/'labels.txt').write_text('couloir\nignore\n' if task=='corridor' else 'rail_left\nrail_right\nignore\n',encoding='utf-8')
    shutil.copy2(ROOT/'configs'/('labelme_corridor.yaml' if task=='corridor' else 'labelme.yaml'),lot/'labelme.yaml')
    # Version 7.7 stores imageData only when explicitly requested; make it explicit.
    with (lot/'labelme.yaml').open('a') as f:f.write('\nwith_image_data: false\n')
    shutil.copy2(ROOT/'examples/team_annotation/start_annotation.py',lot/'start_annotation.py')
    (lot/'ANNOTER.command').write_text('''#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ -x ../../../.labelme-venv/bin/python ]; then
    exec ../../../.labelme-venv/bin/python start_annotation.py
fi
exec python3 start_annotation.py
''')
    (lot/'ANNOTER.command').chmod(0o755)
    (lot/'ANNOTER_WINDOWS.bat').write_text('''@echo off\r
cd /d "%~dp0"\r
py -3 start_annotation.py\r
pause\r
''')
    write_csv(lot/'sequences.csv',rows,['stem','sequence'])
    write_csv(lot/'sources.csv',rows,['stem','sequence','source_frame_index','seconds','image_sha256'])
    dump(lot/'manifest.json',{'task':task,'batch_id':batch_id,'lot_id':lot_id,'images':rows,
                             'source_grouping':'Preserve each image capture sequence. Assignment lots do not define independent captures.'})
    (lot/'LIRE_AVANT_ANNOTATION.md').write_text(f'''# Lot {lot_id} : {len(rows)} images attribuées

Travaille uniquement sur les images de CE dossier. Ton binôme a un autre lot.
Les noms de fichiers doivent rester identiques. Ne déplace pas images/ ni json/.

1. Décompresser entièrement le ZIP avant de travailler. Garder une copie locale.
2. Sur le Mac du projet, double-cliquer sur ANNOTER.command : Labelme est déjà installé.
3. Sur un autre ordinateur, installer Python 3.12–3.14 si nécessaire, puis lancer
   ANNOTER.command (Mac) ou ANNOTER_WINDOWS.bat (Windows). La première ouverture
   installe Labelme 7.7.0 dans un environnement local au lot et exige Internet.
   Sous Linux : python3 start_annotation.py. Le poste du binôme n'a pas été testé.
4. {"Tracer un POLYGONE couloir entre les BORDS INTÉRIEURS des deux rails de la voie parcourue. Le couloir contient les traverses et le ballast entre les rails, pas l’extérieur. Ajouter des points dans les virages et ignore pour les portions incertaines. Arrêter là où les deux bords ne sont plus visibles ensemble. Les suggestions automatiques sont à corriger et ne sont jamais une vérité validée." if task=='corridor' else "Tracer des POLYGONES rail_left, rail_right et ignore pour les parties incertaines. Gauche/droite correspondent à l’image. Suivre les surfaces métalliques visibles de la voie parcourue, sans ballast. Ajouter des points aux courbes, ne pas inventer les portions cachées."}
   Ne pas utiliser line. Arrêter quand les bords sont indécidables.
5. Vérifier, cocher reviewed et enregistrer (Save) sur chaque image. Les annotations
   sont les fichiers JSON dans json/. Relancer le même bouton pour reprendre.
6. Quand tes {len(rows)} images sont finies, compresser seulement le dossier json/
   et l'envoyer au responsable du projet, en indiquant lot {lot_id}. Pas besoin
   d'envoyer .annotation-env/ ni les images ni la vidéo.

Une sauvegarde n'est pas une synchronisation automatique avec le binôme. Ne
travaillez pas tous les deux sur une copie partagée du même fichier JSON.
Comparez visuellement vos premiers contours pour utiliser la même convention.
''',encoding='utf-8')


def prepare(source,sequence,output,exclude_sessions,count=50):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',sequence):raise ValueError('Nom de séquence invalide')
    source=Path(source).resolve()
    protected={p.resolve():sha256(p) for session in exclude_sessions for sub in ['images','json']
               for p in (Path(session)/sub).glob('*') if p.is_file()}
    existing_images={digest for p,digest in protected.items() if p.suffix.lower()=='.png'}
    old_indices=[]
    for p in protected:
        match=re.fullmatch(re.escape(sequence)+r'_frame_(\d+)',p.stem)
        if match:old_indices.append(int(match.group(1)))
    cap=cv2.VideoCapture(str(source))
    try:
        if not cap.isOpened():raise ValueError('Vidéo source illisible')
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:cap.release()
    timing=probe_timing(source,fps)
    indices=plan_indices(timing['timestamps_seconds'],count,old_indices)
    out=new_dir(output)
    for name in ['lot_A','lot_B']:(out/name/'images').mkdir(parents=True)
    source_hash=sha256(source)
    batch_id=out.name
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    selected='+'.join(f'eq(n\\,{i})' for i in indices)
    rows=[]
    with (out/'extraction.log').open('wb') as log:
        reader=subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-i',str(source),
            '-map','0:v:0','-an','-vf','select='+selected,'-fps_mode','passthrough','-f','rawvideo',
            '-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=log)
        try:
            for position,index in enumerate(indices):
                raw=reader.stdout.read(width*height*3)
                if len(raw)!=width*height*3:raise ValueError('Image sélectionnée non décodée')
                lot_id='A' if position%2==0 else 'B'
                stem=f'{sequence}_frame_{index:09d}'
                path=out/f'lot_{lot_id}'/'images'/f'{stem}.png'
                Image.fromarray(np.frombuffer(raw,np.uint8).reshape(height,width,3)).save(path)
                digest=sha256(path)
                if digest in existing_images or any(r['image_sha256']==digest for r in rows):
                    raise ValueError('Image identique à une image existante ou au nouveau lot')
                rows.append(dict(stem=stem,sequence=sequence,assignee=f'lot_{lot_id}',
                    source_frame_index=index,seconds=timing['timestamps_seconds'][index],
                    image_sha256=digest,image=f'images/{stem}.png'))
            if reader.stdout.read(1) or reader.wait(timeout=60):raise ValueError('Décodage de sélection incomplet')
        finally:
            reader.stdout.close()
            if reader.poll() is None:reader.terminate();reader.wait(timeout=10)
    for lot_id in ['A','B']:
        subset=[r for r in rows if r['assignee']==f'lot_{lot_id}']
        lot=out/f'lot_{lot_id}'
        make_lot_files(lot,lot_id,subset,batch_id)
        zip_path=out/f'PARTAGER_LOT_{lot_id}_{len(subset)}_IMAGES.zip'
        with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(lot.rglob('*')):
                if path.is_file():z.write(path,Path(lot.name)/path.relative_to(lot))
            z.writestr(f'{lot.name}/json/','')
    write_csv(out/'repartition.csv',rows,['stem','assignee','sequence','source_frame_index','seconds','image_sha256'])
    unchanged=all(p.is_file() and sha256(p)==digest for p,digest in protected.items())
    if not unchanged:raise RuntimeError('Un fichier existant a changé pendant la préparation')
    report=dict(batch_id=batch_id,source=str(source),source_sha256=source_hash,sequence=sequence,
        new_images=len(rows),lot_A=len(rows)//2,lot_B=len(rows)//2,
        source_decoded_frames=timing['decoded_frames'],timestamp_method='Original decoded presentation timestamps',
        excluded_source_indices=sorted(set(old_indices)),old_files_verified_unchanged=len(protected),
        grouping='All new images remain in the original capture group; reserve the second video for validation',
        annotation_status='Unannotated; no model predictions used as labels',images=rows)
    dump(out/'preparation.json',report)
    print(f'{len(rows)} nouvelles images prêtes dans {out}',flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',required=True)
    parser.add_argument('--sequence',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--exclude-session',action='append',default=[])
    parser.add_argument('--count',type=int,default=50)
    args=parser.parse_args()
    prepare(args.video,args.sequence,args.out,args.exclude_session,args.count)
