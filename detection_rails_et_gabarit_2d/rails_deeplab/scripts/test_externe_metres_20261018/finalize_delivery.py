"""Verify and package V3, retain V2 in archives, update the project entry points."""
import sys,csv,json,hashlib,zipfile,shutil,os,subprocess,re,urllib.parse,py_compile
from pathlib import Path
R=Path(sys.argv[1]);S=Path(__file__).parent;N='GABARIT_V3_METRES_ESTIMES_20261018';V=R/'03_RESULTATS'/N;O=R/'03_RESULTATS/TESTS_ET_METRES_20261018'
A=R/'archives/livraison_avant_v3_20261018';A.mkdir(exist_ok=False)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
assert (O/'AUDIT_EXPORTS_ET_SOURCES.json').exists()
for folder in ['01_TSV_RECORD','02_TSV_03676_180S']:assert (V/folder/'RENDU_VERIFIE.json').exists()
reader='''"""Read a V3 frame and its explicitly conditional metric data."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--dossier',required=True,type=Path);p.add_argument('--frame',required=True,type=int);a=p.parse_args()
with (a.dossier/'FRAMES_VITESSE_VECTEUR.csv').open(newline='') as f:
 r=next((r for r in csv.DictReader(f) if int(r['frame_index'])==a.frame),None)
if r is None:raise SystemExit('Frame absente')
keys=['video_id','frame_index','video_time_s','real_elapsed_s','train_speed_m_s','speed_status','axis_status','observation_status','axis_mae_px','error_status','assumed_gauge_m','gauge_status','travel_segment_id','travel_segment_distance_m_estimate','travel_from_video_start_m_estimate','travel_interval_status']
out={k:r[k] for k in keys};out['axis_segments']=json.loads(r['axis_segments_json'])
out['notice']='Empty = unknown. Assumed gauge and forward distance do not supply calibrated camera depth.'
print(json.dumps(out,ensure_ascii=False,indent=2))
'''
(V/'outils/lire_frame_metres.py').write_text(reader)
subprocess.run([str(R/'.venv/bin/python'),str(V/'outils/lire_frame_metres.py'),'--dossier',str(V/'02_TSV_03676_180S'),'--frame','1375'],check=True,stdout=subprocess.DEVNULL)
shutil.copy2(O/'AUDIT_EXPORTS_ET_SOURCES.json',V/'CONTROLES_EXPORTS.json')
files=[p for p in sorted(V.rglob('*')) if p.is_file() and p.suffix not in ['.log','.pyc'] and p.name!='.DS_Store' and '__pycache__' not in p.parts]
assert not any('railbench' in str(p.relative_to(V)).lower() or 'scene_01' in str(p.relative_to(V)) for p in files)
manifest=[dict(path=str(p.relative_to(V)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
dump(V/'MANIFESTE_SHA256.json',manifest);files.append(V/'MANIFESTE_SHA256.json')
zip_path=R/'02_A_PARTAGER'/f'{N}.zip';assert not zip_path.exists()
with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
    for p in files:z.write(p,N+'/'+str(p.relative_to(V)),compress_type=zipfile.ZIP_STORED if p.suffix.lower() in ['.mp4','.png','.jpg','.zip'] else zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(zip_path) as z:
    assert len(z.infolist())==len(files)
    for p in files:
        h=hashlib.sha256()
        with z.open(N+'/'+str(p.relative_to(V))) as f:
            for b in iter(lambda:f.read(1<<20),b''):h.update(b)
        assert h.hexdigest()==sha(p),str(p)
zip_result=dict(path=str(zip_path.relative_to(R)),bytes=zip_path.stat().st_size,sha256=sha(zip_path),files=len(files),all_member_sha256_and_crc_verified=True,public_source_images_redistributed=False)
dump(O/'PAQUET_V3_VERIFIE.json',zip_result)
print('ZIP_VERIFIED',json.dumps(zip_result),flush=True)
# Archive only the superseded V2 delivery. Keep compatible hidden links.
archived=[]
for old in [R/'03_RESULTATS/GABARIT_2_VIDEOS_V2_20261018',R/'02_A_PARTAGER/GABARIT_2_VIDEOS_V2_20261018.zip']:
    assert not old.is_symlink();before=sha(old) if old.is_file() else None
    dest=A/old.name;old.rename(dest);old.symlink_to(os.path.relpath(dest,old.parent),target_is_directory=dest.is_dir())
    subprocess.run(['/usr/bin/chflags','-h','hidden',str(old)],check=True)
    assert old.exists()
    if before:assert sha(old)==before
    archived.append(dict(previous_path=str(old.relative_to(R)),archive=str(dest.relative_to(R)),sha256=before,compatibility_link=True))
# Back up editable presentation surfaces before changing their entry points.
targets=['00_COMMENCER_ICI.html','03_RESULTATS/RESULTATS.html','03_RESULTATS/README.md','05_COMPRENDRE/LE_PROJET_EXPLIQUE.html','02_A_PARTAGER/00_PAQUET_GABARIT_A_ENVOYER.txt','02_A_PARTAGER/LIRE_AVANT_PARTAGE.md','01_ACTIONS/02_PAQUET_POUR_GABARIT.command','01_ACTIONS/04_RESULTATS_ACTUELS.command']
for rel in targets:
    p=R/rel
    if p.exists():
        q=A/'documents_avant_v3'/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def replace(p,s):
    # Atomic replacement does not mutate a possible hardlinked historical file.
    t=p.with_name(p.name+'.v3tmp');t.write_text(s);os.replace(t,p)
home=(R/'00_COMMENCER_ICI.html').read_text()
home=home.replace('GABARIT_2_VIDEOS_V2_20261018.zip',N+'.zip')
home=home.replace('03_RESULTATS/FIABILISATION_20261018/INDEX.html','03_RESULTATS/TESTS_ET_METRES_20261018/LIRE_LES_RESULTATS.html')
home=home.replace('Résultats et nouveaux contrôles','Test externe et premiers mètres').replace('Comparaison par image, lissage, débit Mac et essais contrôlés.','96 images externes, deux nouveaux clips, erreurs et mètres sous hypothèses.')
home=home.replace('<main>','<main><section><p class="eyebrow">Version actuelle · 18 octobre 2026</p><h2>V3 : vidéos, erreurs et premiers mètres</h2><p>Commencer par le bilan ci-dessous, puis transmettre uniquement le ZIP V3. Les anciens paquets restent archivés. Les annotations et modèles sont conservés.</p><a class="button" href="03_RESULTATS/TESTS_ET_METRES_20261018/LIRE_LES_RESULTATS.html">Voir le bilan et les nouvelles vidéos</a><a class="button secondary" href="03_RESULTATS/GABARIT_V3_METRES_ESTIMES_20261018/LIRE_MOI.html">Comprendre le paquet gabarit</a></section>',1)
replace(R/'00_COMMENCER_ICI.html',home)
guide=R/'05_COMPRENDRE/LE_PROJET_EXPLIQUE.html';s=guide.read_text()
banner='<section><p class="eyebrow">Mise à jour V3 · 18 octobre 2026</p><h2>Les derniers tests et les premières valeurs en mètres</h2><p>Le guide ci-dessous décrit le développement antérieur. Les résultats actualisés et la livraison courante sont ici. Les mètres latéraux restent conditionnés par un écartement supposé ; la précision physique n’est pas validée.</p><a class="button" href="../03_RESULTATS/TESTS_ET_METRES_20261018/LIRE_LES_RESULTATS.html">Lire le nouveau bilan</a><a class="button secondary" href="../03_RESULTATS/GABARIT_V3_METRES_ESTIMES_20261018/LIRE_MOI.html">Paquet gabarit V3</a></section>'
s=s.replace('<main>','<main>'+banner,1);replace(guide,s)
index='''<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Résultats TSV actuels</title><style>body{font:18px/1.7 system-ui;color:#193640;max-width:900px;margin:50px auto;padding:20px}a{color:#006e77}li{margin:18px 0}</style><h1>Résultats actuels</h1><ol><li><a href="TESTS_ET_METRES_20261018/LIRE_LES_RESULTATS.html">Test externe terminé et premiers mètres — lire d’abord</a></li><li><a href="GABARIT_V3_METRES_ESTIMES_20261018/LIRE_MOI.html">Paquet actuel pour l’équipe gabarit — V3</a></li><li><a href="FIABILISATION_20261018/INDEX.html">Contrôles précédents : lissage, débit et trois graines</a></li><li><a href="COMPARAISON_MODELES_ET_PRESENTATION_20261018/PRESENTATION_TSV_LUNDI.pptx">Présentation existante — actualiser ses chiffres avec le nouveau bilan</a></li><li><a href="../archives/livraison_avant_v3_20261018/INDEX.html">Versions précédentes conservées</a></li></ol><p>Les nouvelles vidéos publiques de recherche restent locales. Le ZIP V3 partage les deux vidéos TSV et leurs données.</p></html>'''
replace(R/'03_RESULTATS/RESULTATS.html',index)
replace(R/'03_RESULTATS/README.md','Résultats actuels : ouvrir RESULTATS.html.\n1. TESTS_ET_METRES_20261018 : test externe, robustesse et mètres conditionnels.\n2. GABARIT_V3_METRES_ESTIMES_20261018 : livraison actuelle.\n3. FIABILISATION_20261018 : contrôles antérieurs conservés.\nLa V2 est archivée avec liens compatibles dans 90_ARCHIVES/livraison_avant_v3_20261018.\n')
replace(R/'02_A_PARTAGER/00_PAQUET_GABARIT_A_ENVOYER.txt',N+'.zip\nVersion actuelle : axes, vitesses, avancement et premières valeurs métriques sous hypothèses.\nOuvrir LIRE_MOI.html dans le ZIP décompressé. Les anciennes versions sont archivées.\n')
replace(R/'02_A_PARTAGER/LIRE_AVANT_PARTAGE.md','Paquet actuel : '+N+'.zip\nLe dossier contient LIRE_MOI.html et MESSAGE_A_ENVOYER.txt.\nPartager uniquement ce paquet pour le gabarit, pas tous les essais ni toutes les données publiques.\nLes annotations et modèles restent dans le projet. Les anciens ZIP sont conservés en archives.\n')
replace(R/'01_ACTIONS/02_PAQUET_POUR_GABARIT.command','#!/bin/sh\nset -eu\ncd "$(dirname "$0")/.."\nexec open -R "02_A_PARTAGER/'+N+'.zip"\n')
replace(R/'01_ACTIONS/04_RESULTATS_ACTUELS.command','#!/bin/sh\nset -eu\ncd "$(dirname "$0")/.."\nexec open "03_RESULTATS/TESTS_ET_METRES_20261018/LIRE_LES_RESULTATS.html"\n')
for name in ['02_PAQUET_POUR_GABARIT.command','04_RESULTATS_ACTUELS.command']:(R/'01_ACTIONS'/name).chmod(0o755)
(A/'INDEX.html').write_text('<!doctype html><meta charset="utf-8"><title>Versions avant V3</title><h1>Versions précédentes conservées</h1><p>Livraison actuelle : V3. La V2 reste consultable pour traçabilité.</p><ul><li><a href="GABARIT_2_VIDEOS_V2_20261018/LIRE_MOI.html">Guide V2</a></li><li><a href="GABARIT_2_VIDEOS_V2_20261018.zip">ZIP V2</a></li><li>Les documents avant mise à jour sont dans documents_avant_v3.</li></ul>')
dump(A/'JOURNAL.json',archived)
# Install reproducible scripts and required small source metadata, not raw images.
code=R/'scripts/test_externe_metres_20261018';code.mkdir(exist_ok=False)
for p in S.iterdir():
    if p.is_file() and (p.suffix in ['.py','.csv','.md','.html']):shutil.copy2(p,code/p.name)
shutil.copytree(S/'sources_officielles',code/'sources_officielles')
(code/'README.txt').write_text('''TEST EXTERNE ET METRES — 18 octobre 2026
Résultats : 03_RESULTATS/TESTS_ET_METRES_20261018/LIRE_LES_RESULTATS.html
Livraison : 03_RESULTATS/GABARIT_V3_METRES_ESTIMES_20261018

Ordre exécuté avec la .venv du projet (le dernier argument est le chemin du projet) :
1. remote_zip.py sources_officielles : petits fichiers officiels et index des ZIP distants ; pas de téléchargement intégral.
2. freeze_and_download.py PROJET : protocole avant inférence, 96 images et 2 x 50 frames.
3. evaluate_locked.py PROJET external ; evaluate_locked.py PROJET stress.
4. export_metric_start.py PROJET : V3, mêmes pixels que V2 ; mètres conditionnels, pas calibration.
5. new_video_examples.py PROJET : deux clips et poses publiques. Dépendance optionnelle pyarrow==23.0.1 ; installée à /tmp/tsv_parquet_runtime pour ce lancement, sans modification de la .venv.
6. render_metric_video.py PROJET : deux vidéos TSV avec les données V3.
7. audit_outputs.py PROJET ; build_report.py PROJET ; finalize_delivery.py PROJET.

Les scripts de création refusent les dossiers existants : ne pas relancer en écrasant les résultats figés. Pour une nouvelle expérience, copier les scripts, changer les noms des sorties et figer un nouveau protocole.
Tous les CSV, paramètres et sources de ce lancement sont conservés. CORRECTION_METADATA_SOURCE.json décrit les 11 en-têtes publics inversés ; les annotations originales n’ont pas été réécrites.
Les images publiques sont dans data/external/railbench_locked_20261018 et restent locales selon leur licence. Elles ne sont pas incluses dans le ZIP gabarit.
''')
for p in code.glob('*.py'):py_compile.compile(str(p),cfile=str(S/('__check_'+p.stem+'.pyc')),doraise=True)
# Current protected annotation/model hashes remain identical to the prior inventory.
protected=json.loads((R/'archives/organisation_20261018/FICHIERS_PROTEGES_AVANT.json').read_text());issues=[]
for item in protected:
    p=R/item['path']
    if not p.is_file() or sha(p)!=item['sha256']:issues.append(item['path'])
assert not issues,issues
dump(O/'ANNOTATIONS_MODELES_PRESERVES.json',dict(protected_files=len(protected),hash_mismatches=issues))
# Static local-link integrity; no claim of browser-rendered HTML QA.
targets=[R/'00_COMMENCER_ICI.html',R/'03_RESULTATS/RESULTATS.html',R/'05_COMPRENDRE/LE_PROJET_EXPLIQUE.html',V/'LIRE_MOI.html',O/'LIRE_LES_RESULTATS.html',A/'INDEX.html'];checked=0;broken=[]
for p in targets:
    for link in re.findall(r'(?:href|src|poster)=["\']([^"\']+)',p.read_text()):
        u=urllib.parse.urlparse(link)
        if u.scheme or not u.path:continue
        dest=(p.parent/urllib.parse.unquote(u.path)).resolve();checked+=1
        if not dest.exists():broken.append(dict(page=str(p.relative_to(R)),link=link))
assert not broken,broken
dump(O/'VERIFICATION_FINALE.json',dict(local_links_checked=checked,broken_links=broken,protected_files=len(protected),public_images_not_in_delivery=True,source_scripts_compiled=True,reader_example_passed=True,plots_and_sample_video_frames_visually_inspected=True,html_browser_rendering_not_performed=True,zip=zip_result))
print('DELIVERY_COMPLETE',json.dumps(dict(zip=zip_result,protected_files=len(protected),links_checked=checked)),flush=True)
