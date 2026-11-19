"""Frozen external rail-pair test plus separately labelled development stress tests."""
import sys,json,csv,hashlib,time,collections
from pathlib import Path
import numpy as np,cv2,torch
from PIL import Image
R=Path(sys.argv[1]);sys.path.insert(0,str(R))
from railtsv.model import build_model,load_checkpoint
from railtsv.engine import predict_image
from railtsv.geometry import corridor_profile
from railtsv.data import letterbox
RUN=R/'runs/test_externe_metres_20261018';DATA=R/'data/external/railbench_locked_20261018';OUT=R/'03_RESULTATS/TESTS_ET_METRES_20261018';OUT.mkdir(exist_ok=True,parents=True)
torch.set_num_threads(4);dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
protocol=json.loads((RUN/'PROTOCOLE_FIGE.json').read_text());ck=R/protocol['checkpoint']
assert hashlib.sha256(ck.read_bytes()).hexdigest()==protocol['checkpoint_sha256']
c=load_checkpoint(ck);model=build_model(2);model.load_state_dict(c['state_dict']);model.eval().to(dev)
def write_csv(p,rows,fields=None):
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def prediction(im):
 mask,prob=predict_image(model,im,dev,c['height'],c['width']);profile=corridor_profile(mask,prob)
 return profile,mask,prob
def points(profile):return {p['y']:p for s in profile['segments'] for p in s}
def cross(poly,y):
 vals=[]
 for a,b in zip(poly,poly[1:]):
  if a[1]==b[1]:continue
  if min(a[1],b[1])<=y<max(a[1],b[1]):vals.append(a[0]+(b[0]-a[0])*(y-a[1])/(b[1]-a[1]))
 return vals
def external_reference(imeta,ann):
 h,w=imeta['height'],imeta['width'];ignore=np.zeros((h,w),np.uint8)
 for a in ann:
  if a['category_id']==2:cv2.fillPoly(ignore,[np.rint(a['polygon']).astype(np.int32)],1)
 refs={};complex_rows=0;allrows=0;rails=[a for a in ann if a['category_id']==1]
 for y in range(0,h,8):
  candidates=[]
  for a in rails:
   candidates.extend((x,a['rightRail'],a['occlusion']) for x in cross(a['polyline'],y) if 0<=x<w)
  candidates.sort()
  if len(candidates)>=2:allrows+=1
  if len(candidates)>2:complex_rows+=1
  if len(candidates)!=2:continue
  a,b=candidates
  if a[1]!=0 or b[1]!=1 or a[2]!=0 or b[2]!=0 or b[0]-a[0]<.02*w:continue
  if ignore[y,max(0,int(a[0])):min(w,int(b[0])+1)].any():continue
  refs[y]=dict(left=a[0],right=b[0],center=(a[0]+b[0])/2,width=b[0]-a[0])
 return refs,dict(rail_instances=len(rails),multi_rail_rows=complex_rows,rows_with_at_least_two_rails=allrows)
def score(pred,ref,w,allowed=None):
 ref={y:p for y,p in ref.items() if allowed is None or allowed(y,p)}
 errors=[pred[y]['center_x']-p['center'] for y,p in ref.items() if y in pred]
 return dict(reference_rows=len(ref),returned_rows=len(errors),coverage_pct=100*len(errors)/len(ref) if ref else None,mae_px=float(np.mean(np.abs(errors))) if errors else None,rmse_px=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,bias_px=float(np.mean(errors)) if errors else None,p95_abs_error_px=float(np.percentile(np.abs(errors),95)) if errors else None,mae_pct_width=float(100*np.mean(np.abs(errors))/w) if errors else None)
def overlay(im,profile,ref,path,title):
 im=im.copy();h,w=im.shape[:2];scale=min(1,1280/w);im=cv2.resize(im,(round(w*scale),round(h*scale)))
 for y,p in ref.items():cv2.circle(im,(round(p['center']*scale),round(y*scale)),1,(245,105,210),-1)
 for s in profile['segments']:
  for k,color in [('center_x',(55,240,90)),('left_inner_x',(255,195,65)),('right_inner_x',(255,195,65))]:
   cv2.polylines(im,[np.array([[round(p[k]*scale),round(p['y']*scale)] for p in s],np.int32)],False,color,2 if k=='center_x' else 1)
 cv2.rectangle(im,(0,0),(im.shape[1],67),(15,30,40),-1)
 cv2.putText(im,title,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.57,(240,245,245),1,cv2.LINE_AA)
 cv2.putText(im,'Vert: prediction | Rose: reference des paires uniques | Jaune: limites predites',(10,52),cv2.FONT_HERSHEY_SIMPLEX,.47,(240,245,245),1,cv2.LINE_AA)
 Image.fromarray(im).save(path,quality=90)

mode=sys.argv[2]
if mode=='external':
 assert (RUN/'TELECHARGEMENT_TERMINE.json').exists()
 dest=OUT/'01_TEST_EXTERNE_FIGE';dest.mkdir(exist_ok=False);(dest/'apercus').mkdir()
 anno=json.loads((DATA/'annotations_96.json').read_text());meta={a['id']:a for a in anno['images']};group=collections.defaultdict(list)
 for a in anno['annotations']:group[a['image_id']].append(a)
 rows=list(csv.DictReader((RUN/'IMAGES_TEST_FIGEES.csv').open()));metrics=[];sections=[];empty=[];start=time.monotonic()
 for i,row in enumerate(rows):
  iid=int(row['ID']);m=dict(meta[iid]);im=np.asarray(Image.open(DATA/row['image_path']).convert('RGB'))
  assert im.shape[:2]==(int(row['image_height']),int(row['image_width']))
  # The official images list swaps width/height for 11 unannotated images.
  # Actual pixels agree with independent data_overview.csv. Preserve source;
  # use actual dimensions, keep these images in inference but no invented GT.
  if im.shape[:2]!=(m['height'],m['width']):
   assert not group[iid], 'Annotated dimension mismatch requires manual review'
   m['height'],m['width']=im.shape[:2]
  ref,detail=external_reference(m,group[iid]);profile,mask,prob=prediction(im);p=points(profile);stat=score(p,ref,m['width'])
  rec=dict(image_id=iid,image=row['image_name'],source=row['original_dataset'],scene=row['original_scene'],width=m['width'],height=m['height'],predicted_points=len(p),**detail,**stat);metrics.append(rec)
  for y,q in ref.items():
   pred=p.get(y);sections.append(dict(image_id=iid,source=row['original_dataset'],scene=row['original_scene'],y_px=y,reference_left_px=q['left'],reference_right_px=q['right'],reference_axis_px=q['center'],predicted_axis_px=pred['center_x'] if pred else None,error_signed_px=pred['center_x']-q['center'] if pred else None,available=bool(pred)))
  overlay(im,profile,ref,dest/'apercus'/f'{iid}.jpg',f"RAIL-BENCH {iid} | {row['original_dataset']} | test externe fige | pas de calibration metrique")
  (dest/'apercus'/f'{iid}_profil.json').write_text(json.dumps(profile))
  del im,mask,prob
  if (i+1)%12==0:print('EXTERNAL',i+1,'/',len(rows),flush=True)
 write_csv(dest/'RESULTATS_PAR_IMAGE.csv',metrics);write_csv(dest/'ERREURS_PAR_SECTION.csv',sections)
 groups=[]
 for name in sorted({r['source']+'/'+r['scene'] for r in metrics}):
  a=[r for r in metrics if r['source']+'/'+r['scene']==name];v=[r for r in a if r['mae_pct_width'] is not None];den=sum(r['reference_rows'] for r in a)
  groups.append(dict(scene=name,images=len(a),images_with_reference=sum(r['reference_rows']>0 for r in a),images_with_measured_error=len(v),reference_rows=den,returned_rows=sum(r['returned_rows'] for r in a),coverage_pct=100*sum(r['returned_rows'] for r in a)/den if den else None,image_macro_mae_pct_width=float(np.mean([r['mae_pct_width'] for r in v])) if v else None))
 write_csv(dest/'RESULTATS_PAR_CAPTURE.csv',groups)
 usable=[r for r in metrics if r['mae_pct_width'] is not None];refcount=sum(r['reference_rows'] for r in metrics)
 result=dict(images_tested=len(metrics),source_groups=len(groups),images_with_unique_pair_reference=sum(r['reference_rows']>0 for r in metrics),images_with_measured_axis_error=len(usable),reference_sections=refcount,returned_sections=sum(r['returned_rows'] for r in metrics),coverage_pct=100*sum(r['returned_rows'] for r in metrics)/refcount if refcount else None,image_macro_mae_pct_width=float(np.mean([r['mae_pct_width'] for r in usable])) if usable else None,scene_macro_mae_pct_width=float(np.mean([r['image_macro_mae_pct_width'] for r in groups if r['image_macro_mae_pct_width'] is not None])) if usable else None,images_no_axis=sum(r['predicted_points']==0 for r in metrics),wall_seconds=time.monotonic()-start,scope='Frozen external evaluation of unique unoccluded rail pairs on original 96 institutional RAIL-BENCH validation images. Not hidden official challenge score; not independent TSV field certification. Outer-edge midpoint target; no branch correctness inferred from these errors.',missing_error='No eligible reference or no predicted axis is not zero error; inspect coverage.',checkpoint_sha256=protocol['checkpoint_sha256'],test_used_for_training_or_tuning=False)
 (dest/'RESULTAT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

elif mode=='stress':
 dest=OUT/'02_RECADRAGES_ET_ROBUSTESSE';dest.mkdir(exist_ok=False);(dest/'apercus').mkdir()
 pack=R/'03_RESULTATS/GABARIT_2_VIDEOS_V2_20261018/02_TSV_03676_180S';masks=sorted((pack/'REFERENCES_12').glob('*_mask.png'));metrics=[]
 for i,path in enumerate(masks):
  stem=path.stem.removesuffix('_mask');im=np.asarray(Image.open(pack/'IMAGES_COMMUNES'/f'{stem}.png').convert('RGB'));gt=np.asarray(Image.open(path));h,w=gt.shape
  ref={}
  for y in range(0,h,8):
   xs=np.flatnonzero(gt[y]==1)
   if len(xs)<2 or np.any(np.diff(xs)>1) or xs[0]==0 or xs[-1]==w-1 or gt[y,xs[0]-1]!=0 or gt[y,xs[-1]+1]!=0:continue
   ref[y]=dict(left=xs[0],right=xs[-1],center=(xs[0]+xs[-1])/2,width=xs[-1]-xs[0])
  profile,mask,prob=prediction(im);base=points(profile)
  for variant in ['original','vga_640x480','crop_10pct_each_edge','dark_55pct','blur_sigma4px']:
   allowed=None;show=im
   if variant=='original':vprofile=profile
   elif variant=='crop_10pct_each_edge':
    x0,x1,y0,y1=round(.1*w),round(.9*w),round(.1*h),round(.9*h);roi=im[y0:y1,x0:x1];_,pm,pp=prediction(roi)
    m=np.zeros((h,w),np.uint8);p=np.zeros((2,h,w),np.float32);p[0]=1;m[y0:y1,x0:x1]=pm;p[:,y0:y1,x0:x1]=pp;vprofile=corridor_profile(m,p)
    allowed=lambda y,r: y0<=y<y1 and r['left']>=x0 and r['right']<x1
   else:
    if variant=='vga_640x480':
     small,_,(top,left,nh,nw)=letterbox(im,480,640);_,pm,pp=prediction(small);p=np.array([cv2.resize(q[top:top+nh,left:left+nw],(w,h),interpolation=cv2.INTER_LINEAR) for q in pp]);m=p.argmax(0).astype(np.uint8);vprofile=corridor_profile(m,p)
    else:
     show=np.clip(im.astype(float)*.55,0,255).astype(np.uint8) if variant=='dark_55pct' else cv2.GaussianBlur(im,(25,25),4)
     vprofile,_,_=prediction(show)
   pred=points(vprofile);st=score(pred,ref,w,allowed);bst=score(base,ref,w,allowed)
   metrics.append(dict(image=stem,variant=variant,width=w,height=h,**st,baseline_same_domain_mae_px=bst['mae_px'],baseline_same_domain_coverage_pct=bst['coverage_pct'],role='development_stress_not_independent_new_capture'))
   if i in [0,5,11]:overlay(show,vprofile,{y:r for y,r in ref.items() if allowed is None or allowed(y,r)},dest/'apercus'/f'{stem}_{variant}.jpg',f'DEVELOPPEMENT | {variant} | recadrer ne cree pas une nouvelle capture')
  print('STRESS',i+1,'/',len(masks),flush=True)
 write_csv(dest/'RESULTATS_PAR_IMAGE.csv',metrics)
 summary=[]
 for v in sorted({r['variant'] for r in metrics}):
  a=[r for r in metrics if r['variant']==v];ok=[r for r in a if r['mae_px'] is not None];den=sum(r['reference_rows'] for r in a)
  summary.append(dict(variant=v,images=len(a),measured_images=len(ok),image_macro_mae_px=float(np.mean([r['mae_px'] for r in ok])) if ok else None,baseline_same_domain_mae_px=float(np.mean([r['baseline_same_domain_mae_px'] for r in ok])) if ok else None,coverage_pct=100*sum(r['returned_rows'] for r in a)/den))
 (dest/'RESUME.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
else:raise ValueError(mode)
