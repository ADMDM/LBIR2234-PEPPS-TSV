"""Inspect a completed video export and publish local raw/smoothed diagnostics."""
import argparse,csv,json,math,html,sys,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
import cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from railtsv.common import sha256
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',type=Path,required=True)
args=parser.parse_args();OUT=args.folder.resolve()
validation=json.loads((OUT.parent/'diagnostic_validation/RESUME_VALIDATION.json').read_text())
annotations=json.loads((OUT.parent/'annotations.json').read_text())
report=json.loads((OUT/'video.json').read_text())
rows=[]
with (OUT/'mesures_lissees.csv').open() as f:
 for r in csv.DictReader(f):
  rows.append({k:r[k] for k in ['frame_index','seconds','source_frame_index','source_seconds','repeated_source_frame','raw_section_status','raw_center_x_px','raw_width_px','smoothed_center_x_px','smoothed_width_px','smoothing_status']})
assert len(rows)==report['frames']
num=lambda key:np.array([float(r[key]) if r[key] else np.nan for r in rows])
t=num('seconds');source=num('source_frame_index');a=num('raw_center_x_px');b=num('smoothed_center_x_px');wa=num('raw_width_px');wb=num('smoothed_width_px')
valid=np.isfinite(a)&np.isfinite(b);pairs=valid[1:]&valid[:-1]&(np.diff(source)==1)&(np.diff(num('source_seconds'))<=.15)
da=np.diff(a)[pairs];db=np.diff(b)[pairs]
summary=dict(frames=report['frames'],fps=report['fps'],duration_seconds=report['duration_seconds'],source=report['source'],
 dimensions=report['dimensions'],source_observations=report['source_decoded_frames'],repeated_frames=report['telemetry']['repeated_frames'],
 section_candidate_frames=report['telemetry']['candidate_frames'],section_rejected_frames=report['telemetry']['rejected_frames'],
 pairs_compared=int(pairs.sum()),raw_first_difference_rms_px=float(np.sqrt(np.mean(da*da))),smoothed_first_difference_rms_px=float(np.sqrt(np.mean(db*db))),
 mean_absolute_display_shift_px=float(np.abs(b[valid]-a[valid]).mean()),
 warning='First-difference variation and display shift describe stabilisation, not accuracy or physical uncertainty',
 validation_report='../diagnostic_validation/VOIR_LA_VALIDATION.html',metric_calibration=None)
# Audit all point midpoints and repeat exclusions without loading large files into memory.
raw_points=0;smooth_points=0;samples=[]
with (OUT/'axe_toutes_images.csv').open() as f:
 for r in csv.DictReader(f):
  assert math.isclose(float(r['center_x_px']),(float(r['left_boundary_x_px'])+float(r['right_boundary_x_px']))/2,abs_tol=1e-9)
  assert r['width_m']=='' and r['independent_accuracy_pct']=='';raw_points+=1
with (OUT/'axe_brut_et_lisse_toutes_images.csv').open() as f:
 for r in csv.DictReader(f):
  assert math.isclose(float(r['center_x_smoothed_px']),(float(r['left_smoothed_px'])+float(r['right_smoothed_px']))/2,abs_tol=1e-9)
  assert float(r['left_smoothed_px'])<float(r['right_smoothed_px'])
  assert r['width_m']=='' and r['independent_accuracy_pct']=='';smooth_points+=1
  if len(samples)<3:samples.append(r)
with (OUT/'mesures_par_seconde.csv').open() as f:seconds=list(csv.DictReader(f))
assert len(seconds)==math.ceil(report['duration_seconds']-1e-9)
summary.update(raw_axis_points=raw_points,raw_and_smoothed_points=smooth_points,summary_seconds=len(seconds),example_point_rows=samples)
figdir=OUT/'figures_lissage';figdir.mkdir()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#dfe5e8','svg.fonttype':'none'})
entries=[]
def fig(number,title,caption):
 f,ax=plt.subplots(figsize=(11,5.4));f.suptitle(title,x=.08,ha='left',fontsize=15,fontweight='bold');f.text(.08,.925,f'Vidéo 03676 entière — modèle sur {annotations["training_annotations"]} images — pixels origine',fontsize=10,color='#5c6870');f.subplots_adjust(left=.10,right=.96,bottom=.15,top=.82)
 def finish():
  for ext in ['png','svg']:f.savefig(figdir/f'{number:02d}.{ext}',dpi=170)
  plt.close(f);entries.append(dict(id=f'{number:02d}',title=title,caption=caption))
 return f,ax,finish
f,ax,done=fig(1,'Axe à la ligne de mesure : brut et lissé','Les interruptions restent absentes. La courbe décrit mouvement, perspective et détection, pas une incertitude physique.')
ax.plot(t,a,color='#b88334',linewidth=.8,alpha=.7,label='Brut');ax.plot(t,b,color='#166f93',linewidth=1,label='Lissé');ax.set(xlabel='Temps vidéo (s)',ylabel='Centre horizontal à la ligne mesurée (px)');ax.legend();done()
f,ax,done=fig(2,'Largeur apparente : brut et lissé','Distance horizontale entre les limites du couloir ; pas un écartement physique en mètres.')
ax.plot(t,wa,color='#b88334',linewidth=.8,alpha=.7,label='Brut');ax.plot(t,wb,color='#166f93',linewidth=1,label='Lissé');ax.set(xlabel='Temps vidéo (s)',ylabel='Largeur apparente à la ligne mesurée (px)');ax.legend();done()
f,ax,done=fig(3,'Décalage introduit par le filtre temporel','La stabilisation peut retarder la réponse dans un virage. Ce décalage ne compare pas le résultat à la voie réelle.')
ax.hist(b[valid]-a[valid],bins=50,color='#166f93');ax.axvline(0,color='#404a53');ax.set(xlabel='Centre lissé − centre brut (px)',ylabel='Images source avec les deux valeurs');done()
f,ax,done=fig(4,'Disponibilité de la section mesurée, seconde par seconde','Candidats du contrôle géométrique à une hauteur fixe ; ce n’est ni la précision, ni la couverture de l’axe contre les 12 annotations.')
ax.plot([int(r['second']) for r in seconds],[float(r['coverage_pct']) if r['coverage_pct'] else np.nan for r in seconds],color='#166f93');ax.set(xlabel='Temps vidéo (s)',ylabel='Images avec section candidate (%)',ylim=(0,100));done()
# Decode key portions, including the corrected switch and the last image.
cap=cv2.VideoCapture(str(OUT/'rails_annotes.mp4'));assert cap.isOpened();checks=[];thumbs=[]
for sec in [0,60,90,120,150,report['duration_seconds']-1/report['fps']]:
 k=min(report['frames']-1,round(sec*report['fps']));cap.set(cv2.CAP_PROP_POS_FRAMES,k);ok,bgr=cap.read();assert ok
 rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);p=OUT/f'apercu_{k:06d}.jpg';Image.fromarray(rgb).save(p,quality=93);checks.append(dict(frame=k,seconds=k/report['fps'],preview=p.name));thumbs.append(Image.fromarray(rgb).resize((640,360)))
cap.release();sheet=Image.new('RGB',(1280,1164),'white');font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',20)
for j,(im,c) in enumerate(zip(thumbs,checks)):
 x=(j%2)*640;y=(j//2)*388;sheet.paste(im,(x,y+28));ImageDraw.Draw(sheet).text((x+8,y+2),f'{c["seconds"]:.2f} s',font=font,fill='black')
sheet.save(OUT/'CONTROLE_VIDEO.jpg')
summary['decoded_preview_frames']=checks;summary['source_sha256_verified']=sha256(report['source'])==report['source_sha256'];assert summary['source_sha256_verified']
(OUT/'LISSAGE_RESULTATS.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'VERIFICATION_EXPORT.json').write_text(json.dumps(dict(frame_records_match_video=True,axis_midpoints_verified=raw_points,smoothed_midpoints_verified=smooth_points,unknown_meters_empty=True,source_preserved=True,decoded_key_frames=True),indent=2)+'\n')
# Sharing bundle: coordinate data and contract, without the large video.
archive=OUT/'DONNEES_POUR_EQUIPE_GABARIT.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
 for name in ['mesures_par_seconde.csv','axe_par_seconde.csv','axe_brut_et_lisse_par_seconde.csv','video.json','LISSAGE_RESULTATS.json','VERIFICATION_EXPORT.json']:z.write(OUT/name,name)
 z.write(ROOT/'docs/LISSAGE_ET_DONNEES_GABARIT.md','LIRE_AVANT_UTILISATION.md')
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
parts=['<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Vidéo complète et axe lissé TSV</title><style>body{font:17px/1.55 system-ui;color:#18323f;background:#f2f6f7;margin:0}main{max-width:1200px;margin:auto;padding:28px}section{background:white;padding:22px;border-radius:12px;margin:18px 0;border:1px solid #dbe4e8}video,img{width:100%;height:auto}a{color:#006d8e}.note{border-left:5px solid #bd7527}</style><main><h1>Seconde vidéo complète : couloir, axe lissé et CSV</h1>',
 f'<p>{report["duration_seconds"]:.2f} s • {report["frames"]} images • {report["dimensions"][0]}×{report["dimensions"][1]} • sans son</p>',
 '<section><video controls muted playsinline preload="metadata" src="rails_annotes.mp4"></video><p><a href="rails_annotes.mp4">Ouvrir la vidéo</a></p></section>',
 '<section class="note"><p>Le fond coloré est la segmentation brute. Les limites et l’axe vert sont lissés pour l’affichage. La ligne de mesure du panneau utilise les limites lissées quand un point observé est exploitable à cette hauteur.</p><p>Le filtre utilise les deux limites ensemble et ne force pas une largeur constante. Les interruptions restent des régions sans axe étayé. Les confusions avec d’autres voies peuvent persister.</p><p>Les mètres et la précision réelle par image restent indisponibles. Le score interne du modèle est non calibré.</p></section>',
 '<section><h2>Données pour l’équipe gabarit</h2><p><a href="DONNEES_POUR_EQUIPE_GABARIT.zip">ZIP des données et mode d’emploi</a> · <a href="axe_brut_et_lisse_par_seconde.csv">Axe brut et lissé par seconde</a> · <a href="mesures_par_seconde.csv">Synthèse brute par seconde</a> · <a href="axe_brut_et_lisse_toutes_images.csv">Tous les points bruts et lissés</a> · <a href="mesures.csv">Mesures brutes par image</a></p><p>Chaque seconde comporte plusieurs points d’axe. Les coordonnées restent dans le repère de l’image originale. Les régions non détectées ne sont pas reconstruites.</p></section>',
 f'<section><h2>Validation du modèle actuel</h2><p>Sur les 12 références : Dice moyen {100*validation["macro_image_dice"]:.2f} %, IoU moyenne {100*validation["macro_image_iou"]:.2f} %, erreur moyenne du milieu {validation["axis_mae_px"]:.2f} px, couverture de l’axe {validation["coverage_pct"]:.2f} %. Ces références ont servi au choix du modèle ; ces scores ne concernent pas toutes les images de la vidéo.</p><a href="../diagnostic_validation/VOIR_LA_VALIDATION.html">Voir les 12 références et les figures de validation</a></section>',
 '<section><h2>Effet du filtre</h2>',f'<p>Variation entre observations consécutives à la ligne mesurée : RMS brut {summary["raw_first_difference_rms_px"]:.2f} px, RMS lissé {summary["smoothed_first_difference_rms_px"]:.2f} px. Décalage absolu moyen entre affichage lissé et centre brut : {summary["mean_absolute_display_shift_px"]:.2f} px.</p><p>Ces nombres décrivent la stabilisation ; ils ne prouvent pas une précision accrue par rapport à la voie réelle.</p></section>']
for e in entries:parts.append(f'<section><h2>{html.escape(e["title"])}</h2><p>{html.escape(e["caption"])}</p><img src="figures_lissage/{e["id"]}.png" alt="{html.escape(e["title"])}"><p><a href="figures_lissage/{e["id"]}.svg">SVG</a></p></section>')
parts.append('</main></html>');(OUT/'VOIR_VIDEO_ET_MESURES.html').write_text('\n'.join(parts))
(OUT/'LIRE_VIDEO_ET_CSV.md').write_text('# Vidéo complète et lissage\n\n[Voir la vidéo et les données](VOIR_VIDEO_ET_MESURES.html).\n\n[Mode d’emploi des CSV](../../../docs/LISSAGE_ET_DONNEES_GABARIT.md).\n')
print(json.dumps(summary,indent=2));print('Résultat prêt :',OUT/'VOIR_VIDEO_ET_MESURES.html')
