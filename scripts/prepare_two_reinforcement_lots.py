"""Two disjoint, portable 30-image lots from already verified Drive media.

The independent control capture is referenced for future export, never sampled.
Existing labels and baseline snapshots are read only. No model prediction is
used as annotation, no reviewed flag is set, and no training is launched.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import cv2
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / '04_ANNOTATIONS/RENFORT_2_LOTS_30_20261014'
SHARE = ROOT / '02_A_PARTAGER/RENFORT_2_LOTS_30_20261014'
PREVIOUS = ROOT / '03_RESULTATS/AMELIORATIONS_20261014/nouveaux_exemples'
NAMES = ['LOT_1_MOI_30', 'LOT_2_BINOME_30']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def selections():
    long_times = [4.4,8.4,12.4,16.4,20.4,24.4,27.4,30.4,33.4,36.4,
                  39.8,42.4,45.8,48.4,51.8,54.4,57.4,60.4,63.4,66.4,
                  69.4,72.4,75.8,78.4,81.4,88.4,100.4,112.4,124.4,
                  130.4,134.4,138.4,142.4,146.4,150.4,154.4,158.4,162.4,166.4,172.4]
    groups = [('data/raw/drive_20261005/video_03383.AVI', 'video_03383', long_times),
              ('data/raw/drive_20261014/20220803_114407.mp4', 'photos_20220803', [1.5,3.5,5.5,8.,10.5,13.5]),
              ('data/raw/drive_20261014/20220803_134951.mp4', 'photos_20220803', [.5,1.8,3.1,4.4,5.7,7.]),
              ('data/raw/drive_20261014/20220803_133227.mp4', 'photos_20220803', [.7,2.3,5.7,7.8,13.7,14.7]),
              ('data/raw/drive_20261014/20220803_134155.mp4', 'photos_20220803', [.6,9.2])]
    return [(NAMES[i % 2], source, sequence, seconds)
            for source, sequence, times in groups for i, seconds in enumerate(times)]


def contact(rows, target):
    for start in range(0, len(rows), 15):
        sheet = Image.new('RGB', (1440, 5*225), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, r in enumerate(rows[start:start+15]):
            x, y = (i % 3)*480, (i // 3)*225
            im = Image.open(ROOT/r['image']).convert('RGB')
            thumb = ImageOps.contain(im, (474, 195))
            sheet.paste(thumb, (x+(474-thumb.width)//2, y+26))
            draw.text((x+4,y+5), f"{r['number']:02d} | {Path(r['source']).stem} | {r['decoded_seconds']:.2f} s", fill='black')
        sheet.save(target/f'APERCU_{start+1:02d}_{min(start+15,len(rows)):02d}.jpg', quality=92)


RETURN_SCRIPT = r'''"""Verify and package the assigned JSON files; Python standard library only."""
import argparse, datetime, hashlib, json, math, zipfile
from pathlib import Path

def verify(root):
    manifest=json.loads((root/'manifest.json').read_text())
    expected={r['stem']+'.json' for r in manifest['images']}
    found={p.name for p in (root/'json').glob('*.json')}
    if found!=expected:raise ValueError('Le dossier json doit contenir exactement les 30 noms attribués.')
    pending=[]
    for r in manifest['images']:
        image=root/'images'/(r['stem']+'.png')
        if hashlib.sha256(image.read_bytes()).hexdigest()!=r['image_sha256']:
            raise ValueError('Image modifiée : '+image.name)
        p=root/'json'/(r['stem']+'.json');d=json.loads(p.read_text())
        if (d.get('imageWidth'),d.get('imageHeight'))!=(r['width'],r['height']):
            raise ValueError('Dimensions incorrectes : '+p.name)
        if d.get('imagePath','').replace(chr(92),'/').split('/')[-1]!=image.name:
            raise ValueError('Image incorrecte : '+p.name)
        if d.get('flags',{}).get('reviewed') is not True:pending.append(p.name)
        for s in d.get('shapes',[]):
            pts=s.get('points',[])
            if s.get('label') not in ['couloir','ignore'] or s.get('shape_type')!='polygon' or len(pts)<3:
                raise ValueError('Polygone couloir/ignore invalide : '+p.name)
            for pt in pts:
                if len(pt)!=2 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in pt) or not (0<=pt[0]<r['width'] and 0<=pt[1]<r['height']):
                    raise ValueError('Sommet hors image : '+p.name)
        excluded=d.get('flags',{}).get('exclude_from_training') is True
        if d.get('flags',{}).get('reviewed') is True and not excluded and not d.get('shapes'):
            raise ValueError('Image relue sans polygone ni exclusion : '+p.name)
    return manifest, pending

def main():
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parent;manifest,pending=verify(root)
    print(f"{30-len(pending)}/30 fichiers enregistrés avec reviewed.")
    if pending:
        print('A terminer : '+', '.join(pending[:5]))
        raise ValueError('Finir les 30 relectures et enregistrer avant de préparer le retour.')
    if a.check_only:return
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    archive=root/f"RETOUR_{manifest['lot_id']}_{stamp}.zip"
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        z.write(root/'manifest.json','manifest.json')
        for r in manifest['images']:
            q=root/'json'/(r['stem']+'.json');z.write(q,'json/'+q.name)
    print('Envoyer ce fichier ZIP : '+str(archive))

if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError) as e:
        print('Retour non créé :',e);raise SystemExit(1)
'''


def launcher(script):
    return f'''#!/bin/sh
set -eu
cd "$(dirname "$0")"
for py in ../../../.labelme-venv/bin/python ../../.labelme-venv/bin/python .annotation-env/bin/python; do
    if [ -x "$py" ]; then exec "$py" {script}; fi
done
exec python3 {script}
'''


def windows(script):
    return f'''@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel% equ 0 (py -3 {script}) else (python {script})
pause
'''


def instructions(lot):
    who = 'Toi' if lot == NAMES[0] else 'Ton binôme'
    return f'''# {lot} — 30 images de renfort

Attribution : {who}. Ce lot complète les anciens lots A/B/C ; il ne les remplace pas.
Il appartient à la série RENFORT_2_LOTS_30_20261014. Les 20 images de contrôle restent ailleurs.

## Ouvrir

1. Extraire complètement le ZIP avant de travailler. Garder tous les fichiers ensemble.
2. Mac : double-cliquer sur ANNOTER.command. Windows : ANNOTER_WINDOWS.bat.
3. Sur le Mac du projet, l’environnement Labelme existant est réutilisé.
   Sur un autre ordinateur, Python 3.12 à 3.14 doit être installé ; le lanceur
   installe Labelme 7.7.0 dans ce lot à sa première utilisation (Internet requis).
   Le code d’entraînement, PyTorch et RailSem19 ne sont pas nécessaires.
4. Si Labelme est déjà installé mais ouvert directement, préférer malgré tout
   le lanceur : il charge les bons drapeaux reviewed, virage et aiguillage.
   Le dossier à ouvrir est images ; les sauvegardes doivent aller dans json.

## Annoter chaque image

- Créer un **polygone couloir** entre les **faces intérieures** des deux rails
  de la voie suivie. L’espace entre les rails est inclus, l’épaisseur du métal non.
- Dans un virage, ajouter des sommets pour suivre la courbe. Ne pas dessiner
  deux rails séparés, ni une simple ligne centrale.
- Arrêter la partie certaine quand les bords ne sont plus identifiables et
  couvrir la zone ambiguë avec un polygone ignore.
- À un aiguillage, suivre une seule branche identifiable. Ne pas réunir
  deux voies en Y. Si le choix est ambigu, utiliser ignore et route_ambiguous.
- Cocher virage/aiguillage si pertinent. Si aucune portion n’est exploitable,
  utiliser exclude_from_training et expliquer la raison dans la description.
- Cocher reviewed **après relecture**, puis enregistrer avec Ctrl+S / Cmd+S.
  Le cochage seul ne suffit pas : la sauvegarde automatique est désactivée.

Les images sont vierges d’annotations. Les drapeaux ne sont pas des classes à dessiner.
Les aperçus permettent de repérer les scènes ; ne pas les annoter.

## Renvoyer le travail

1. Terminer les 30 images, relire, cocher reviewed et enregistrer chacune.
2. Fermer Labelme, puis lancer PREPARER_RETOUR.command sur Mac ou
   PREPARER_RETOUR_WINDOWS.bat sur Windows.
3. Envoyer le fichier **RETOUR_{lot}_date.zip** créé dans ce dossier.
   Il contient les 30 JSON et le manifeste ; inutile de renvoyer les images.

Le contrôle vérifie la structure, les noms, les images et les drapeaux.
Il ne certifie pas la précision du tracé : une vérification visuelle reste nécessaire.
Ne pas renommer les images ou les JSON. Ne pas travailler dans le lot de l’autre personne.
'''


def lot_html(lot):
    return f'''<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{lot} — Annotation du couloir</title>
<style>body{{font:18px/1.6 system-ui,sans-serif;max-width:950px;margin:40px auto;padding:0 22px;color:#183547;background:#f4f8fa}}section{{background:white;padding:20px 26px;border:1px solid #d9e3e7;border-radius:12px;margin:18px 0}}h1{{font-size:34px;line-height:1.2}}a{{color:#006a75}}img{{width:100%}}code{{background:#edf3f6;padding:2px 5px}}li{{margin:10px 0}}</style>
<h1>{lot}</h1><p>30 images supplémentaires · couloir entre les faces intérieures · sources : votre Drive</p>
<section><h2>1. Ouvrir Labelme</h2><p>Extraire le ZIP. Dans ce dossier, double-cliquer sur <b>ANNOTER.command</b> (Mac) ou <b>ANNOTER_WINDOWS.bat</b> (Windows).</p>
<p>Le lanceur charge les bons drapeaux, dont <b>reviewed</b>, et dirige les sauvegardes vers <b>json</b>. Sur un autre ordinateur : Python 3.12–3.14 requis, puis installation automatique de Labelme 7.7.0 au premier lancement. Le code d’entraînement n’est pas nécessaire.</p></section>
<section><h2>2. Dessiner et enregistrer</h2><ol><li>Créer un polygone <b>couloir</b> entre les <b>bords intérieurs</b> des deux rails de la voie suivie.</li><li>Ajouter des points dans les courbes. À un aiguillage, choisir une seule branche identifiable ; ne pas réunir deux voies en Y.</li><li>Arrêter la partie certaine et dessiner <b>ignore</b> là où les limites ou la branche sont ambiguës.</li><li>Relire, cocher <b>reviewed</b>, puis <b>enregistrer</b>. Le cochage seul ne sauvegarde pas le fichier.</li></ol><p>Les images sont vierges ; ne pas annoter les planches d’aperçu. Si une image est entièrement inutilisable : cocher <b>exclude_from_training</b> et expliquer la raison dans sa description.</p></section>
<section><h2>3. Renvoyer les annotations</h2><p>Après les 30 images, fermer Labelme et double-cliquer sur <b>PREPARER_RETOUR.command</b> (Mac) ou <b>PREPARER_RETOUR_WINDOWS.bat</b> (Windows).</p><p>Envoyer le <b>RETOUR_{lot}_date.zip</b> créé : il contient uniquement les 30 JSON et le manifeste. Les images n’ont pas besoin d’être renvoyées.</p><p><a href="LIRE_AVANT_ANNOTATION.md">Consignes complètes</a></p></section>
<section><h2>Aperçu des 30 images</h2><img src="APERCU_01_15.jpg" alt="Images 1 à 15"><img src="APERCU_16_30.jpg" alt="Images 16 à 30"></section></html>'''


def main():
    if BATCH.exists() or SHARE.exists():
        raise FileExistsError('Une sortie existe déjà : aucun écrasement autorisé.')
    old=json.loads((ROOT/'.local/sauvegarde_ameliorations_20261014/PRESERVATION.json').read_text())
    # Record present bytes, not a historical assumption about later user edits.
    protected={str(p.relative_to(ROOT)):sha(p) for p in [*(ROOT/n for n in old),*PREVIOUS.rglob('*.json')]}
    previous=json.loads((PREVIOUS/'MANIFESTE.json').read_text())
    if any(json.loads((ROOT/r['json']).read_text()).get('shapes') or
           json.loads((ROOT/r['json']).read_text()).get('flags',{}).get('reviewed') is True
           for r in previous['cases'] if r['role']=='renfort_a_annoter'):
        raise ValueError('Les premières candidates ont été annotées : préserver/rapprocher ce travail avant de redistribuer.')
    original=json.loads((ROOT/'data/annotation_corridor_v1_150_20261006/preparation.json').read_text())
    long_source=next(s for s in original['sources'] if s['sequence']=='video_03383')
    source_meta={long_source['path']:long_source}
    for s in json.loads((ROOT/'data/raw/drive_20261014/PROVENANCE.json').read_text()):
        source_meta[str(ROOT/'data/raw/drive_20261014'/s['file'])]=s
    run=ROOT/json.loads((ROOT/'configs/diagnostic_target.json').read_text())['run']
    forbidden={}
    def add_image(path, label):
        with Image.open(path) as im:rgb=im.convert('RGB');digest=hashlib.sha256(rgb.tobytes()).hexdigest()
        forbidden[digest]=label
    for lot in (ROOT/'data/annotation_corridor_v1_150_20261006').glob('lot_*'):
        for f in (lot/'images').glob('*.png'):add_image(f,str(f.relative_to(ROOT)))
    for split in ['train','val']:
        manifest=run/'dataset'/(split+'.csv')
        for row in csv.DictReader(manifest.open()):
            f=Path(row['image']);f=f if f.is_absolute() else manifest.parent/f
            add_image(f,str(f))
    for r in previous['cases']:forbidden[r['decoded_rgb_sha256']]=r['image']
    BATCH.mkdir(parents=True);SHARE.mkdir(parents=True)
    rows=[];counts={name:0 for name in NAMES};verified_sources=set()
    for lot,source,sequence,seconds in selections():
        path=ROOT/source;meta=source_meta[str(path)]
        if source not in verified_sources:
            if sha(path)!=meta['sha256']:raise ValueError('Source modifiée : '+source)
            verified_sources.add(source)
        folder=BATCH/lot;(folder/'images').mkdir(parents=True,exist_ok=True);(folder/'json').mkdir(exist_ok=True)
        cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_MSEC,seconds*1000)
        ok,bgr=cap.read();pos=cap.get(cv2.CAP_PROP_POS_MSEC)/1000;frame=int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1;cap.release()
        if not ok:raise ValueError(f'Décodage impossible : {source} {seconds}')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2];digest=hashlib.sha256(rgb.tobytes()).hexdigest()
        if digest in forbidden:raise ValueError(f'Doublon : {source} {seconds} / {forbidden[digest]}')
        forbidden[digest]=f'{lot} {seconds}'
        counts[lot]+=1;number=counts[lot];stem=f"R{NAMES.index(lot)+1}_{number:03d}_{path.stem}_{seconds:07.2f}s"
        image=folder/'images'/(stem+'.png');Image.fromarray(rgb).save(image)
        annotation=folder/'json'/(stem+'.json')
        write_json(annotation,dict(version='7.7.0',flags=dict(reviewed=False,aiguillage=False,virage=False,
            route_ambiguous=False,exclude_from_training=False),shapes=[],imagePath='../images/'+image.name,
            imageData=None,imageHeight=h,imageWidth=w))
        rows.append(dict(id=len(rows)+1,lot_id=lot,number=number,stem=stem,role='renfort_a_annoter',
            sequence=sequence,source=source,source_url=meta['source_url'],source_sha256=meta['sha256'],
            source_seconds_requested=seconds,decoded_seconds=pos,decoded_frame_index=frame,
            image=str(image.relative_to(ROOT)),json=str(annotation.relative_to(ROOT)),
            image_sha256=sha(image),decoded_rgb_sha256=digest,width=w,height=h,reviewed=False))
        print(lot,number,'/30',flush=True)
    assert counts==dict.fromkeys(NAMES,30)
    for lot in NAMES:
        folder=BATCH/lot;assigned=sorted((r for r in rows if r['lot_id']==lot),key=lambda r:r['number'])
        portable=[dict(r,image='images/'+r['stem']+'.png',json='json/'+r['stem']+'.json') for r in assigned]
        write_json(folder/'manifest.json',dict(batch_id=BATCH.name,lot_id=lot,expected=30,task='corridor',images=portable))
        (folder/'labelme.yaml').write_text('auto_save: false\nflags:\n  reviewed: false\n  aiguillage: false\n  virage: false\n  route_ambiguous: false\n  exclude_from_training: false\nlabels: [couloir, ignore]\nsort_labels: false\nvalidate_label: exact\ncanvas:\n  crosshair:\n    polygon: true\n')
        (folder/'labels.txt').write_text('couloir\nignore\n')
        shutil.copy2(ROOT/'examples/team_annotation/start_annotation.py',folder/'start_annotation.py')
        (folder/'preparer_retour.py').write_text(RETURN_SCRIPT)
        for name,script in [('ANNOTER','start_annotation.py'),('PREPARER_RETOUR','preparer_retour.py')]:
            command=folder/(name+'.command');command.write_text(launcher(script));command.chmod(0o755)
            (folder/(name+'_WINDOWS.bat')).write_text(windows(script))
        (folder/'LIRE_AVANT_ANNOTATION.md').write_text(instructions(lot))
        (folder/'COMMENCER_ICI.html').write_text(lot_html(lot))
        contact(assigned,folder)
        with (folder/'sources.csv').open('w',newline='') as f:
            fields=['number','stem','sequence','source','source_url','decoded_seconds','decoded_frame_index']
            wr=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');wr.writeheader();wr.writerows(assigned)
        archive=SHARE/(lot+'.zip')
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for f in sorted(folder.rglob('*')):
                if f.is_file():z.write(f,lot+'/'+f.relative_to(folder).as_posix())
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert len([n for n in z.namelist() if '/images/' in n and n.endswith('.png')])==30
            assert len([n for n in z.namelist() if '/json/' in n and n.endswith('.json')])==30
    controls=[dict(r,id=61+i,assignment='existing_control_unchanged') for i,r in enumerate(r for r in previous['cases'] if r['role']=='controle_nouvelle_capture')]
    write_json(BATCH/'MANIFESTE.json',dict(batch_id=BATCH.name,task='corridor',assigned_images=60,
        existing_controls_referenced=20,cases=rows+controls,training_performed=False,
        note='Only 60 reinforcement images assigned here; the existing 20 control images remain in their original folder. Whole capture groups remain separate.'))
    write_json(BATCH/'PRESERVATION.json',protected)
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    verification=dict(assigned_images=60,images_per_lot=counts,unique_decoded_images=60,
        overlap_between_lots=0,exact_duplicates_against_previous_lots_and_candidates=0,
        control_images_assigned_for_training=0,all_new_reviewed_flags=False,
        protected_files_unchanged=len(protected),sources_verified=len(verified_sources),
        visual_review_complete=False,zip_integrity_checked=True,
        archives=[dict(file=str((SHARE/(n+'.zip')).relative_to(ROOT)),sha256=sha(SHARE/(n+'.zip')),
                       bytes=(SHARE/(n+'.zip')).stat().st_size) for n in NAMES])
    write_json(BATCH/'VERIFICATION.json',verification)
    print(json.dumps(verification,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
