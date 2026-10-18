"""Freeze external evaluation before inference; fetch only selected public images."""
import csv,json,hashlib,zipfile,sys,shutil,time,datetime,concurrent.futures
from pathlib import Path
from remote_zip import RemoteZipFile
R=Path(sys.argv[1]);S=Path(__file__).parent;OUT=R/'runs/test_externe_metres_20261018';DATA=R/'data/external/railbench_locked_20261018'
OUT.mkdir(parents=True,exist_ok=False);DATA.mkdir(parents=True,exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
rows=[r for r in csv.DictReader((S/'railbench_object_overview.csv').open()) if r['split']=='val' and r['original_dataset'] in ['OSDaR26_GAF','S4R','GAF']]
assert len(rows)==96
checkpoint=R/'runs/experience_egopath_20261016_141426/modele/best.pt'
protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='frozen_before_any_external_prediction',checkpoint=str(checkpoint.relative_to(R)),checkpoint_sha256=sha(checkpoint),production_model_changed=False,external_selection='ALL 96 official validation images from OSDaR26_GAF, S4R and GAF; exclusion of internet-video sources to reduce RailSem19 source overlap risk. No selection by model performance.',sources_url='https://www.mrt.kit.edu/railbench/track_detection',source_dates='Modern institutional recordings, including source filenames from 2021 and 2023; distinct from TSV Nigeria 2017, TSV photos 2022 and Nordland 2013. Exact duplicate check also performed later.',original_benchmark_split='val; this is a frozen external test for our model, NOT the hidden official RAIL-BENCH test or a TSV final field test',image_ids=[int(r['ID']) for r in rows],groups=sorted({r['original_dataset']+'/'+r['original_scene'] for r in rows}),preprocessing='Original full image letterboxed to checkpoint 640x480; inverse projection to original pixels; no ROI tuning',profile=dict(min_score=.55,step=8,smoothing=False),axis_reference='At rows with exactly two distinct unoccluded annotated rails, left/right flags ordered, no ignore crossing, apparent width >=2% image width: use their midpoint. Outer rail head edges differ from TSV inner-edge supervision. Rows with multiple possible pairs are not claimed to have known ego-axis.',metrics=['MAE per image in original px and percent width; coverage versus reference rows; mean by source scene; error distribution on comparable rows','No calibrated metre error on external images without geometry; no claimed switch-route accuracy'],no_training=True,no_test_based_tuning=True,robustness_test='The 12 TSV development images also tested with prescribed 10% crop, VGA letterbox, blur and exposure changes; transformed copies remain development, not new independent captures.',metres='TSV per-frame forward travel from speed/time, gap-aware. Per-row lateral metre proxy under declared gauge scenarios 1.067 and 1.435 m. Reference error converted using annotated width, not self-normalized predicted width. No 3D depth or measured gauge claim.',video_clips=dict(scenes=['scene_01_train','scene_17_train'],selection='First 50 timestamped images of each scene; no selection after viewing; new institutional captures, 10Hz nominal',scope='Qualitative temporal demonstration plus supplied camera pose reference; not annotated axis accuracy'))
(OUT/'PROTOCOLE_FIGE.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
with (OUT/'IMAGES_TEST_FIGEES.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for name in ['railbench_object_overview.csv','railbench_odometry_overview.csv','railbench_sources_overview.csv','railbench_readme.md','railbench_track.html']:shutil.copy2(S/name,OUT/name)
with zipfile.ZipFile(S/'sources_officielles/annotations.zip') as z:
 annotations=json.loads(z.read('annotations/rails/annotations_val.json'));ids=set(protocol['image_ids']);annotations['images']=[i for i in annotations['images'] if i['id'] in ids];annotations['annotations']=[a for a in annotations['annotations'] if a['image_id'] in ids];(DATA/'annotations_96.json').write_text(json.dumps(annotations))
for name in ['calibration.zip','gt_poses.zip']:shutil.copy2(S/'sources_officielles'/name,DATA/name)
base='https://www.mrt.kit.edu/railbench_data/'
jobs=[('images',r['image_path']) for r in rows]
index=json.loads((S/'sources_officielles/SCENES_ZIP_INDEX.json').read_text())
for scene in protocol['video_clips']['scenes']:
 selected=sorted([i for i in index if '/'+scene+'/' in i['name'] and i['name'].endswith('.png')],key=lambda x:int(Path(x['name']).name.split('_')[0]))[:50]
 jobs.extend(('scenes',i['name']) for i in selected)
urls={'images':base+'object_and_rail/images.zip','scenes':base+'odometry/scenes.zip'}
# Two network readers, each maintains its own seekable ZIP handle; no shared seeks.
def worker(part):
 remotes={k:RemoteZipFile(u) for k,u in urls.items()};archives={k:zipfile.ZipFile(v) for k,v in remotes.items()};done=[]
 for k,name in part:
  target=DATA/name;target.parent.mkdir(parents=True,exist_ok=True)
  for attempt in range(3):
   try:b=archives[k].read(name);break
   except Exception:
    if attempt==2:raise
    time.sleep(1)
  target.write_bytes(b);done.append(dict(archive=urls[k],member=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
  if len(done)%10==0:print('downloaded',len(done),'/',len(part),flush=True)
 return done
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 results=list(pool.map(worker,[jobs[::2],jobs[1::2]]))
manifest=sorted(sum(results,[]),key=lambda x:x['member']);(DATA/'DOWNLOAD_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
(OUT/'TELECHARGEMENT_TERMINE.json').write_text(json.dumps(dict(files=len(manifest),bytes=sum(x['bytes'] for x in manifest),no_full_archive_download=True),indent=2))
print('DOWNLOAD_COMPLETE',len(manifest),flush=True)
