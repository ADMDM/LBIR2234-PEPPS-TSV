"""Create a readable, auditable local report for corridor validation figures."""
import argparse,csv,json,html,zipfile,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
ROOT=Path(__file__).resolve().parents[1]
if not (ROOT/'railtsv').is_dir():raise RuntimeError('Exécuter ce script depuis le dépôt complet.')
sys.path.insert(0,str(ROOT))
from railtsv.common import device,sha256,read_manifest
from railtsv.annotation_config import active_validation
from railtsv.model import build_model,load_checkpoint
from railtsv.engine import predict_image
from railtsv.geometry import corridor_profile
from scripts.export_labelme import rasterize
from scripts.diagnostic_plots import image_metrics

def summarize(out,baseline=None,annotation_note=None):
 out=Path(out).resolve();m=json.loads((out/'mesures_et_sources.json').read_text())
 if m['scope']!='validation':raise ValueError('Un diagnostic de validation est requis')
 rows=list(csv.DictReader((out/'metriques_par_image.csv').open()));points=list(csv.DictReader((out/'erreurs_axe_sections.csv').open()))
 e=np.array([float(p['center_error_px']) for p in points]);session=active_validation(ROOT)
 report=dict(images=len(rows),scope=m['scope_note'],macro_image_dice=m['macro_image_dice'],macro_image_iou=m['macro_image_iou'],
 axis_mae_px=float(np.abs(e).mean()),axis_rmse_px=float(np.sqrt((e*e).mean())),axis_p95_absolute_px=float(np.percentile(np.abs(e),95)),axis_max_absolute_px=float(np.abs(e).max()),
 coverage_pct=100*m['axis_matched_points']/m['axis_reference_points'],matched_sections=m['axis_matched_points'],reference_sections=m['axis_reference_points'],
 units='original_image_pixels',dimensions=[1920,1080],checkpoint_sha256=m['checkpoint_sha256'],model_retrained=None,
 meters=None,physical_uncertainty=None,annotation_caveat=annotation_note or 'Références reviewed ; la qualité du tracé humain limite les scores.')
 if baseline:
  b=Path(baseline).resolve();old=json.loads((b/'RESUME_VALIDATION.json').read_text());report['previous_report']=str(b);report['model_retrained']=m['checkpoint_sha256']!=old['checkpoint_sha256']
  report['before']={k:old[k] for k in ['macro_image_dice','macro_image_iou','axis_mae_px','axis_p95_absolute_px','coverage_pct']}
 (out/'annotations_reference').mkdir(exist_ok=False);(out/'comparaisons').mkdir(exist_ok=False)
 audit=[]
 for row in rows:
  source=next(x for x in m['sources'] if Path(x['image']).stem==row['image'])
  p=Path(source.get('annotation') or session/'json'/(row['image']+'.json'))
  image=Path(source['image']);data=json.loads(p.read_text())
  with Image.open(image) as im:mask=rasterize(data,im.size,'corridor')
  assert np.array_equal(mask,np.array(Image.open(out/'masks'/(row['image']+'.png')))), 'Annotation changed since scoring'
  if baseline:
   old_mask=Path(baseline)/'masks'/(row['image']+'.png')
   assert np.array_equal(mask,np.array(Image.open(old_mask))), 'Baseline reference differs; model comparison would be invalid'
  shutil.copy2(p,out/'annotations_reference'/p.name)
  audit.append(dict(stem=row['image'],image_path=str(image),json_path=str(p),image_sha256=sha256(image),json_sha256=sha256(p)))
 trained=read_manifest(Path(m['checkpoint']).parent.parent/'dataset/train.csv');seqs={r['sequence'] for r in trained}
 assert 'video_03676' not in seqs
 assert not {sha256(r['image']) for r in trained}&{r['image_sha256'] for r in audit}
 cp=load_checkpoint(m['checkpoint']);dev=device('auto');torch.set_num_threads(4);model=build_model(2);model.load_state_dict(cp['state_dict']);model.to(dev).eval()
 font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',25) if sys.platform=='darwin' else ImageFont.load_default()
 for i,(row,s) in enumerate(zip(rows,audit),1):
  rgb=np.array(Image.open(s['image_path']).convert('RGB'));ref=np.array(Image.open(out/'masks'/(row['image']+'.png')))
  pred,prob=predict_image(model,rgb,dev,cp['height'],cp['width']);actual,_,_=image_metrics(pred,ref,prob,row['image'],'video_03676')
  assert np.isclose(actual['dice'],float(row['dice']),rtol=0,atol=1e-10)
  assert actual['matched_points']==int(row['matched_points'])
  profile=corridor_profile(pred,prob);board=Image.new('RGB',(1920,584),'white')
  for j,(label,mask,color) in enumerate([('Annotation',ref,(0,220,255)),('Prédiction',pred,(255,175,35))]):
   view=rgb.copy();sel=mask==1;view[sel]=(.83*view[sel]+.17*np.array(color)).astype(np.uint8)
   ignored=ref==255;view[ignored]=(.78*view[ignored]+.22*np.array((255,40,225))).astype(np.uint8)
   im=Image.fromarray(view);d=ImageDraw.Draw(im)
   if j==0:
    data=json.loads((out/'annotations_reference'/Path(s['json_path']).name).read_text())
    for shape in data['shapes']:
     if shape['label']=='couloir':d.line([tuple(p) for p in shape['points']]+[tuple(shape['points'][0])],fill=color,width=3)
   else:
    for segment in profile['segments']:
     for key in ['left_inner_x','right_inner_x']:d.line([(p[key],p['y']) for p in segment],fill=color,width=3)
     d.line([(p['center_x'],p['y']) for p in segment],fill=(60,255,90),width=3)
   board.paste(im.resize((960,540)),(j*960,44));ImageDraw.Draw(board).text((j*960+12,8),f'{label} — image {i:02d}',font=font,fill='black')
  board.save(out/'comparaisons'/f'{i:02d}.jpg',quality=94);print(f'Comparaison {i}/12',flush=True)
 assert all(sha256(s['json_path'])==s['json_sha256'] for s in audit)
 assert sha256(m['checkpoint'])==m['checkpoint_sha256']
 (out/'sources_annotations.json').write_text(json.dumps(audit,indent=2)+'\n')
 report['training_images']=m.get('training_images',len(trained))
 report['validation_used_for_model_selection']=m.get('validation_used_for_model_selection',False)
 report['same_reference_masks_as_baseline']=bool(baseline)
 report['network_size']=m['network_trained_size']
 (out/'RESUME_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
 def fmt(x):return f'{x:.2f}'.replace('.',',')
 metrics=[('Dice moyen',100*report['macro_image_dice'],'%'),('IoU moyenne',100*report['macro_image_iou'],'%'),('Erreur moyenne du milieu',report['axis_mae_px'],'px'),('95e percentile de l’erreur',report['axis_p95_absolute_px'],'px'),('Couverture de l’axe',report['coverage_pct'],'%')]
 train_note=f"<section><p>Ce modèle a appris sur <strong>{len(trained)} images</strong>, avec une entrée réseau {m['network_trained_size'][0]}×{m['network_trained_size'][1]}. "
 train_note+=('Les 12 références ci-dessous ont servi au choix du meilleur passage ; elles ne constituent pas un test final indépendant.' if report['validation_used_for_model_selection'] else 'Ces images ne faisaient pas partie de son entraînement.')+' Les erreurs sont relatives aux annotations humaines.</p></section>'
 parts=['<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nouvelle validation TSV</title><style>body{font:17px/1.55 system-ui;color:#18323f;background:#f2f6f7;margin:0}main{max-width:1200px;margin:auto;padding:28px}section,article{background:white;padding:22px;border-radius:12px;margin:18px 0;border:1px solid #dbe4e8}img{width:100%;height:auto}a{color:#006d8e}td,th{padding:10px;text-align:left;border-bottom:1px solid #dbe4e8}.note{border-left:5px solid #bd7527}</style><main><h1>Validation du couloir et de son axe médian</h1><p>12 références de la vidéo 03676 • coordonnées en pixels de l’image originale</p><section><table><tr><th>Mesure</th><th>Résultat</th></tr>']
 for label,value,unit in metrics:parts.append(f'<tr><td>{label}</td><td>{fmt(value)} {unit}</td></tr>')
 parts+=['</table><p>Erreurs en pixels des images originales 1920×1080. Les sections manquantes comptent dans la couverture. Ces scores comparent les prédictions aux polygones humains ; ils ne constituent pas une précision physique en mètres.</p></section>',f'<section class="note"><strong>Qualité des références</strong><p>{html.escape(report["annotation_caveat"])}</p><p>Les voies voisines et les aiguillages doivent être examinés dans les comparaisons ci-dessous. Les interruptions de l’axe restent visibles dans la couverture. Une seule autre capture, pas un test final multi-captures.</p></section>',
 train_note,
 '<section><a href="metriques_par_image.csv">CSV par image</a> · <a href="erreurs_axe_sections.csv">CSV des coordonnées et erreurs</a> · <a href="VALIDATION_FIGURES_ET_CSV.zip">Archive à partager</a> · <a href="LIRE_LES_FIGURES.md">Guide des figures</a></section><h2>Les 12 comparaisons</h2><p>Annotation bleue à gauche ; couloir prédit orange et axe vert à droite ; ignore en rose.</p>']
 for i,row in enumerate(rows,1):parts+=[f'<article><h3>Image {i:02d}</h3><p>Dice {fmt(100*float(row["dice"]))} % · erreur axe {fmt(float(row["axis_mae_px"]))} px · couverture {fmt(float(row["coverage_pct"]))} %</p><a href="comparaisons/{i:02d}.jpg"><img src="comparaisons/{i:02d}.jpg" alt="Comparaison image {i:02d}" loading="lazy"></a></article>']
 parts+=['<h2>Les figures</h2>']
 for entry in m['figures']:
  n=entry['id'];parts+=[f'<article><h3>{n} — {html.escape(entry["title"])}</h3><p>{html.escape(entry["scope"])}</p><p>{html.escape(entry["caption"])}</p><img src="figures/{n}.png" alt="{html.escape(entry["title"])}" loading="lazy"><p><a href="figures/{n}.svg">SVG</a></p></article>']
 parts+=['</main></html>'];(out/'VOIR_LA_VALIDATION.html').write_text('\n'.join(parts))
 lines=['# Validation du couloir et de son axe médian','',m['scope_note'],'','| Mesure | Résultat |','|---|---:|']+[f'| {label} | {fmt(value)} {unit} |' for label,value,unit in metrics]
 lines+=['','Les erreurs portent sur les sections appariées des images originales 1920×1080.','',report['annotation_caveat'],'',f'Références figées et sauvegardées. Modèle entraîné sur {len(trained)} images. Les essais précédents sont conservés.',
 f'Validation utilisée pour sélectionner le meilleur passage : {report["validation_used_for_model_selection"]}. Ce n’est pas un test final indépendant.',
 '', '[Voir les comparaisons](VOIR_LA_VALIDATION.html). Les CSV de ce dossier concernent les 12 images. Les CSV vidéo sont rangés dans le dossier video du nouvel entraînement.',
 '', 'Une seule capture de validation ; mètres et incertitude physique indisponibles.']
 (out/'LIRE_LE_RESULTAT.md').write_text('\n'.join(lines)+'\n')
 (out/'VERIFICATION_VALIDATION.json').write_text(json.dumps(dict(annotations_preserved=True,checkpoint_preserved=True,displayed_predictions_match_scores=True,validation_sequence_disjoint=True,validation_hashes_disjoint=True,images=len(rows)),indent=2)+'\n')
 archive=out/'VALIDATION_FIGURES_ET_CSV.zip'
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
  for p in sorted(out.rglob('*')):
   if p.is_file() and p!=archive and p.suffix in ['.png','.svg','.jpg','.csv','.json','.md','.html']:z.write(p,p.relative_to(out))
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 print('Rapport terminé :',out,flush=True)
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--baseline',type=Path);p.add_argument('--annotation-note')
 args=p.parse_args();summarize(args.out,args.baseline,args.annotation_note)
