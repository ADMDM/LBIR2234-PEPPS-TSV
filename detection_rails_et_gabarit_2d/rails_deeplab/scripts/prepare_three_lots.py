"""Prepare 150 unannotated images in three lots, with capture provenance."""
import csv,json,re,shutil,subprocess,sys,zipfile
from pathlib import Path
import cv2,imageio_ffmpeg,numpy as np
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from railtsv.common import new_dir,dump,sha256
from scripts.prepare_team_annotations import make_lot_files,write_csv

PHOTO_NAMES=['20220803_113810.jpg','20220803_114907.jpg','20220803_131412.jpg',
 '20220803_132838.jpg','20220803_133548.jpg','20220803_133938.jpg',
 '20220803_134121.jpg','20220803_140601.jpg','20220803_144426.jpg',
 '20220803_153410.jpg','20220803_154029.jpg','20220803_154519.jpg']

def indices(count,total,fps,excluded=()):
    if count<1 or total<count or fps<=0:raise ValueError('Sélection vidéo impossible')
    available=np.ones(total,bool);frames=np.arange(total)
    for i in excluded:available&=abs(frames-i)>round(.45*fps)
    selected=[]
    for target in np.linspace(fps,total-fps-1,count):
        candidates=np.flatnonzero(available)
        if not len(candidates):raise ValueError('Pas assez d’images distinctes')
        i=int(candidates[np.argmin(abs(candidates-target))]);selected.append(i)
        available&=abs(frames-i)>max(1,round(total/count*.25))
    return sorted(selected)

def extract(source,sequence,selected,pool,category):
    cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS)
    w,h=int(cap.get(3)),int(cap.get(4));cap.release()
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();rows=[]
    select='+'.join(f'eq(n\\,{i})' for i in selected)
    with (pool/f'{sequence}.log').open('wb') as log:
        reader=subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','info','-nostdin','-i',str(source),
            '-map','0:v:0','-an','-vf','select='+select+',showinfo','-fps_mode','passthrough',
            '-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=log)
        try:
            for i in selected:
                raw=reader.stdout.read(w*h*3)
                if len(raw)!=w*h*3:raise ValueError('Image sélectionnée incomplète')
                image=np.frombuffer(raw,np.uint8).reshape(h,w,3)
                stem=f'{sequence}_frame_{i:09d}';path=pool/f'{stem}.png';Image.fromarray(image).save(path)
                roi=image[h//3:];brightness=float(roi.mean());contrast=float(roi.std())
                rows.append(dict(stem=stem,sequence=sequence,source_frame_index=i,seconds=i/fps,
                    image_sha256=sha256(path),category=category,source_path=str(source),
                    brightness=brightness,contrast=contrast,image=f'images/{stem}.png'))
            if reader.stdout.read(1) or reader.wait(timeout=60):raise ValueError('Décodage incomplet')
        finally:
            reader.stdout.close()
            if reader.poll() is None:reader.terminate();reader.wait(timeout=10)
    times=re.findall(r'\bpts_time:([\d.eE+-]+)',(pool/f'{sequence}.log').read_text())
    if len(times)!=len(rows):raise ValueError('Horodatages sélectionnés non vérifiés')
    for row,t in zip(rows,times):row['seconds']=float(t)
    return rows

def assign(rows):
    if len(rows)!=150 or len({r['stem'] for r in rows})!=150 or len({r['image_sha256'] for r in rows})!=150:
        raise ValueError('150 images originales distinctes requises')
    counts={'A':0,'B':0,'C':0};assigned=[]
    # Interleave all sources; capture groups remain unchanged.
    for row in rows:
        lot=min(counts,key=lambda k:counts[k]);counts[lot]+=1
        assigned.append(dict(row,assignee='lot_'+lot))
    assert set(counts.values())=={50}
    return assigned

def prepare(output):
    raw=ROOT/'data/raw/drive_20261005';out=new_dir(output);pool=out/'preparation_images';pool.mkdir()
    sessions=[ROOT/'data/annotation_videos_v1',
              *sorted((ROOT/'data/annotation_team_v2_50').glob('lot_*'))]
    protected=[p for session in sessions for sub in ['images','json']
               for p in (session/sub).rglob('*') if p.is_file()]
    before={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in protected}
    excluded=[]
    for p in protected:
        m=re.fullmatch(r'video_03383_frame_(\d+)',p.stem)
        if m:excluded.append(int(m.group(1)))
    source=raw/'video_03383.AVI';cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    # Source contains 5308 actual decoded images; use a verified sequential decode.
    from railtsv.video import probe_timing
    timing=probe_timing(source,fps)
    tsv=extract(source,'video_03383',indices(113,timing['decoded_frames'],fps,excluded),pool,'tsv_video')
    nordland=raw/'nordland_ete_02/nordland_ete_02.mp4';cap=cv2.VideoCapture(str(nordland))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));nfps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    candidates=extract(nordland,'nordland_ete',indices(110,total,nfps),pool,'additional_image')
    visible=[r for r in candidates if r['brightness']>=35 and r['contrast']>=18]
    if len(visible)<25:raise ValueError('Pas assez de vues Nordland suffisamment éclairées pour l’annotation')
    nord=[visible[int(i)] for i in np.linspace(0,len(visible)-1,25).round().astype(int)]
    provenance=json.loads((raw/'photos_provenance.json').read_text());photo_map={r['name']:r for r in provenance}
    photos=[]
    for name in PHOTO_NAMES:
        origin=raw/'photos_20220803'/name;stem='photo_'+origin.stem;path=pool/f'{stem}.png'
        with Image.open(origin) as im:ImageOps.exif_transpose(im).convert('RGB').save(path)
        photos.append(dict(stem=stem,sequence='photos_20220803',source_frame_index=None,seconds=None,
            image_sha256=sha256(path),category='additional_image',source_path=str(origin),
            source_url=photo_map[name]['url'],original_sha256=photo_map[name]['sha256'],
            transform='EXIF orientation normalized, no resize',image=f'images/{stem}.png'))
    rows=assign(tsv+nord+photos)
    for lot_id in ['A','B','C']:
        lot=out/f'lot_{lot_id}';(lot/'images').mkdir(parents=True)
        subset=[r for r in rows if r['assignee']==f'lot_{lot_id}']
        for row in subset:shutil.copy2(pool/f'{row["stem"]}.png',lot/'images'/f'{row["stem"]}.png')
        make_lot_files(lot,lot_id,subset,out.name)
        with zipfile.ZipFile(out/f'PARTAGER_LOT_{lot_id}_50_IMAGES.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for path in sorted(lot.rglob('*')):
                if path.is_file():z.write(path,Path(lot.name)/path.relative_to(lot))
            z.writestr(f'{lot.name}/json/','')
    write_csv(out/'repartition.csv',rows,['stem','assignee','sequence','category','source_frame_index','seconds','image_sha256'])
    dump(out/'preparation.json',dict(batch_id=out.name,sequence='video_03383',
        train_sequences=['video_03383','nordland_ete','photos_20220803'],sources=json.loads((raw/'video_provenance.json').read_text()),
        images=rows,new_images=150,tsv_video_images=113,additional_images=37,photos=12,nordland_frames=25,
        ratio_tsv_pct=113/150*100,heldout_capture='video_03676',
        excluded_source_indices=sorted(set(excluded)),old_files_verified_unchanged=len(protected),
        grouping='Assignment lots are mixed; capture groups preserved for train/validation separation',
        no_overwrite=True,annotation_status='Unannotated; all human validation remains to do',
        excluded_photos=[r['name'] for r in provenance if r['name'] not in PHOTO_NAMES],
        discarded_nordland_candidates=len(candidates)-25))
    assert all(p.stat().st_size==s and p.stat().st_mtime_ns==t for p,(s,t) in before.items())
    print('Trois lots de 50 prêts : 113 images TSV, 12 photos frontales, 25 vues Nordland.',flush=True)
    print(out,flush=True)
    return out

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();prepare(args.out)
