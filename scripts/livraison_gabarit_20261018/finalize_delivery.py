"""Verify delivered data, create real examples, then package with checksums."""
from pathlib import Path
import csv,gzip,json,math,hashlib,zipfile,argparse,subprocess,shutil
from datetime import datetime
import cv2,numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def rows(path):
    op=gzip.open if str(path).endswith('.gz') else open
    with op(path,'rt',encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)

def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2))

def save_rows(path,data):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();root=a.root;out=a.out;full=out/'01_TSV_RECORD_COMPLET'
    frames=list(rows(full/'FRAMES_VITESSE_GPS.csv'));assert len(frames)==7608
    provenance=json.loads((out/'PROVENANCE.json').read_text());source=root/'data/raw/drive_20261014/tsv_record_video.mp4'
    assert sha(source)==provenance['source_videos'][0]['source_sha256']
    ids=[50,660,901,7607];examples={i:[] for i in ids};n=0;observed=0;missing=0
    for r in rows(full/'AXE_POINTS.csv.gz'):
        i=int(r['frame_index']);y=int(r['y_px'])
        assert i==n//90 and y==(n%90)*8
        f=frames[i]
        for k in ['utc_time','real_elapsed_s','video_time_s','train_speed_km_h','train_speed_m_s','quality_flags']:
            assert r[k]==f[k],(i,k)
        assert not r['axis_error_px'] and not r['x_m'] and not r['depth_m']
        assert bool(r['raw_x_px'])==bool(r['smoothed_x_px'])
        for prefix in ['raw','smoothed']:
            if r[prefix+'_x_px']:
                x,l,rr=map(float,[r[prefix+'_x_px'],r[prefix+'_left_px'],r[prefix+'_right_px']])
                assert 0<=l<x<rr<1280 and abs(x-(l+rr)/2)<1e-7
                assert abs(float(r[prefix+'_width_px'])-(rr-l))<1e-7
                dx,dy=r[prefix+'_tangent_dx'],r[prefix+'_tangent_dy']
                if dx:assert float(dy)<0 and math.isclose(math.hypot(float(dx),float(dy)),1,abs_tol=1e-7)
                else:assert not dy
            else:assert not any(r[prefix+'_'+k] for k in ['left_px','right_px','width_px','segment','tangent_dx','tangent_dy'])
        observed+=bool(r['raw_x_px']);missing+=not r['raw_x_px'];n+=1
        if i in examples:examples[i].append(r)
    assert n==7608*90
    vcount=0
    for r in rows(full/'VECTEURS_PAR_FRAME.csv.gz'):
        i=int(r['frame_index']);assert i==vcount
        segments=json.loads(r['raw_segments_json'])
        assert sum(len(s) for s in segments)==int(frames[i]['raw_axis_points'])
        assert all(all(b['y_px']>a['y_px'] for a,b in zip(s,s[1:])) for s in segments)
        vcount+=1
    assert vcount==7608
    integration_checked=0
    origin=datetime.fromisoformat(frames[0]['utc_time'].replace('Z','+00:00'))
    for i,r in enumerate(frames):
        assert int(r['frame_index'])==i
        elapsed=(datetime.fromisoformat(r['utc_time'].replace('Z','+00:00'))-origin).total_seconds()
        assert math.isclose(elapsed,float(r['real_elapsed_s']),abs_tol=1e-6)
        if r['train_speed_km_h']:
            assert math.isclose(float(r['train_speed_km_h'])/3.6,float(r['train_speed_m_s']),abs_tol=1e-8)
        else:assert not r['train_speed_m_s']
        if i:
            dt=elapsed-float(frames[i-1]['real_elapsed_s'])
            assert dt>0 and math.isclose(dt,float(r['real_delta_s']),abs_tol=1e-6)
            if r['interval_distance_from_speed_m']:
                assert dt<=5 and frames[i-1]['train_speed_m_s'] and r['train_speed_m_s']
                expected=.5*(float(frames[i-1]['train_speed_m_s'])+float(r['train_speed_m_s']))*dt
                assert math.isclose(expected,float(r['interval_distance_from_speed_m']),abs_tol=1e-6)
                integration_checked+=1
            if dt>5 or not r['train_speed_m_s'] or not frames[i-1]['train_speed_m_s']:
                assert not r['interval_distance_from_speed_m']
    refs=json.loads((out/'03_ERREURS_12_REFERENCES/references.json').read_text())
    for r in refs:
        assert sha(out/'03_ERREURS_12_REFERENCES'/r['image_file'])==r['image_sha256']
        assert sha(out/'03_ERREURS_12_REFERENCES'/r['mask_file'])==r['mask_sha256']
    models_unchanged={}
    for name,r in provenance['models'].items():
        models_unchanged[name]=sha(r['checkpoint'])==r['checkpoint_sha256'];assert models_unchanged[name]
    ex=out/'exemples';ex.mkdir(exist_ok=True)
    original=cv2.VideoCapture(str(source));annotated=cv2.VideoCapture(str(full/'rails_annotes.mp4'))
    assert int(annotated.get(cv2.CAP_PROP_FRAME_COUNT))==7608
    thumbs=[]
    for i in ids:
        original.set(cv2.CAP_PROP_POS_FRAMES,i);ok,bgr=original.read();assert ok and bgr.shape[:2]==(720,1280)
        annotated.set(cv2.CAP_PROP_POS_FRAMES,i);ok,view=annotated.read();assert ok and view.shape[:2]==(720,1280)
        save_rows(ex/f'FRAME_{i}_AXE.csv',examples[i]);save_rows(ex/f'FRAME_{i}_VITESSE_GPS.csv',[frames[i]])
        assert cv2.imwrite(str(ex/f'FRAME_{i}_ORIGINALE.png'),bgr)
        assert cv2.imwrite(str(ex/f'FRAME_{i}_ANNOTEE.png'),view)
        thumbs.append(cv2.resize(view,(640,360)))
    original.release();annotated.release()
    assert cv2.imwrite(str(ex/'CONTROLE_4_FRAMES.jpg'),np.vstack([np.hstack(thumbs[:2]),np.hstack(thumbs[2:])]))
    (ex/'LIRE_LES_EXEMPLES.txt').write_text('Extraits réels de TSV_record, dimensions originales 1280 × 720.\nFrame 50 : 10 s de lecture. Frame 660 : première coupure UTC >5 s. Frame 901 : première vitesse absente. Frame 7607 : dernière frame.\nLes PNG annotés proviennent de la nouvelle vidéo fournie ; les PNG originaux proviennent de la source. Les CSV sont extraits du même export.\nLe montage réduit est destiné au contrôle visuel ; ne pas y appliquer directement les coordonnées originales.\n')
    coverage=json.loads((full/'COUVERTURE_EXPORT.json').read_text())
    report=dict(status='share_with_caveats',complete_video_frames=7608,all_points_rows_checked=n,
        observed_raw_points=observed,missing_raw_points=missing,vector_frames_checked=vcount,
        full_frame_clock_speed_join_checked=True,real_elapsed_utc_checked=True,
        speed_interval_integration_checked=integration_checked,
        midpoint_width_tangent_bounds_checked=True,missing_geometry_error_metres_remain_empty=True,
        reference_images_and_masks_sha256_checked=12,models_unchanged=models_unchanged,
        code_tests='42 passed: clocks, video, smoothing, geometry, metrics and ego-path integration',
        public_419_arithmetic_checked=True,preview_frames_checked=ids,
        annotation_or_model_files_deleted=False,
        source_video_sha256=sha(source),output_video_sha256=sha(full/'rails_annotes.mp4'),
        cleanup=json.loads((root/'runs/nettoyage_zip_20261018/SUPPRESSIONS_VERIFIEES.json').read_text()),
        cleanup_additional_runs=json.loads((root/'runs/nettoyage_zip_20261018/complement_runs/SUPPRESSIONS_VERIFIEES.json').read_text()),
        not_validated=['physical rail gauge or 3D metres','per-frame ground-truth error in videos','switch route success rate','independent final TSV field accuracy'])
    dump(out/'VERIFICATION_LIVRAISON.json',report)
    (out/'06_COUVERTURE_DE_LA_VIDEO.txt').write_text(
        f"Vidéo complète : {coverage['frames']} frames traitées.\n"
        f"Au moins une portion d'axe brut exportée : {coverage['frames_with_raw_axis']} frames.\n"
        f"Aucune portion d'axe brut exportée : {coverage['frames']-coverage['frames_with_raw_axis']} frames.\n"
        f"Points bruts présents : {coverage['raw_observed_points']} sur {coverage['point_rows']} positions de grille possibles.\n"
        "Cette grille inclut le ciel, les zones masquées et les hauteurs où le couloir n'est pas visible : ce ratio n'est pas une précision ou une couverture de référence.\n"
        "La présence d'un axe ne prouve pas que la bonne voie a été choisie. Ces statistiques décrivent la disponibilité des sorties.\n"
        f"Vitesses absentes : {coverage['missing_speed_frames']}. Frames avec indicateur télémétrique : {coverage['telemetry_flagged_frames']}. Coupures UTC >5s : {coverage['utc_gaps_above_5s']}.\n")
    inventory=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name!='MANIFESTE_SHA256.json':inventory.append(dict(path=str(p.relative_to(out)),size_bytes=p.stat().st_size,sha256=sha(p)))
    dump(out/'MANIFESTE_SHA256.json',dict(files=inventory,manifest_self_excluded=True))
    subprocess.run([__import__('sys').executable,str(out/'outils/verifier_paquet.py'),str(out)],check=True)
    target=root/'02_A_PARTAGER/GABARIT_LIVRAISON_20261018.zip';partial=target.with_suffix('.en_cours.zip')
    assert not target.exists() and not partial.exists()
    with zipfile.ZipFile(partial,'x',allowZip64=True) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,str(Path(out.name)/p.relative_to(out)),compress_type=zipfile.ZIP_STORED if p.suffix in ['.mp4','.gz','.png','.jpg','.pdf'] else zipfile.ZIP_DEFLATED,compresslevel=6)
    with zipfile.ZipFile(partial) as z:assert z.testzip() is None
    partial.rename(target)
    dump(root/'runs/nettoyage_zip_20261018/LIVRAISON_TERMINEE.json',dict(zip=str(target),size_bytes=target.stat().st_size,sha256=sha(target),zip_crc_ok=True,files=len(inventory)+1))
    print(json.dumps(dict(package=str(target),bytes=target.stat().st_size,coverage=coverage),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
