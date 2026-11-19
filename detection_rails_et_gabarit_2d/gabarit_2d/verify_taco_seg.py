#!/usr/bin/env python3
"""CPU-only final checks and transparent contour overlay for prepared TACO."""
import argparse,collections,hashlib,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def digest(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);a=ap.parse_args();root=a.dataset
 rows=json.loads((root/'MANIFEST_ALL.json').read_text());summary=json.loads((root/'SUMMARY.json').read_text());source=json.loads((root/'COCO_ORIGINAL_ALL.json').read_text())
 anns=collections.defaultdict(list)
 for x in source['annotations']:anns[x['image_id']].append(x)
 ids=set();counts=collections.Counter();batch_splits=collections.defaultdict(set);errors=[];labelcount=0
 for r in rows:
  iid=r['image_id'];ids.add(iid);batch_splits[r['batch']].add(r['split']);counts[r['split']]+=1
  p=Path(r['prepared_image']);lp=Path(r['prepared_label']);lines=lp.read_text().splitlines();labelcount+=len(lines)
  if len(lines)!=len(anns[iid])or len(lines)!=r['annotation_count']:errors.append([iid,'annotation_count'])
  if digest(p)!=r['prepared_image_sha256']:errors.append([iid,'prepared_hash'])
  with Image.open(p)as im:
   if im.mode!='RGB'or im.size!=(r['prepared_width'],r['prepared_height'])or max(im.size)>1280 or im.getexif().get(274,1)!=1:errors.append([iid,'image_geometry'])
  for line in lines:
   v=line.split();xy=np.array(list(map(float,v[1:])))
   if v[0]!='0'or len(xy)<6 or len(xy)%2 or not np.isfinite(xy).all()or(xy<0).any()or(xy>1).any():errors.append([iid,'label_geometry'])
 if len(ids)!=len(rows):errors.append(['all','duplicate_image_ids'])
 if any(len(v)>1 for v in batch_splits.values()):errors.append(['all','batch_split_leak'])
 byid={r['image_id']:r for r in rows}
 dup=json.loads((root/'DUPLICATES_PHASH.json').read_text())
 for d in dup:
  if byid[d['image_id_a']]['split']!=byid[d['image_id_b']]['split']:errors.append([d,'duplicate_split_leak'])
 for split in['train','val','test']:
  members=[r for r in rows if r['split']==split];lst=(root/f'{split}.txt').read_text().splitlines();m=json.loads((root/f'manifest_{split}.json').read_text());c=json.loads((root/f'COCO_ORIGINAL_{split}.json').read_text())
  if lst!=[r['prepared_image']for r in members]or m!=members:errors.append([split,'manifest_list_disagree'])
  if {i['id']for i in c['images']}!={r['image_id']for r in members}:errors.append([split,'COCO_image_membership'])
  expected=[x for x in source['annotations']if x['image_id']in{r['image_id']for r in members}]
  if c['annotations']!=expected:errors.append([split,'COCO_annotations_changed'])
 audit=json.loads((root/'ALL_POLYGON_AUDIT.json').read_text())
 accepted_audit=[x for x in audit if x['image_accepted']]
 if any(x['mask_iou']<.95 or x['prepared_resolution_audit']['mask_iou']<.95 for x in accepted_audit):errors.append(['all','accepted_below_iou_threshold'])
 # Show the worst retained conversion cases with original green and YOLO magenta contours.
 worst=sorted(accepted_audit,key=lambda x:x['prepared_resolution_audit']['mask_iou']);chosen=[];seen=set()
 for x in worst:
  if x['image_id']not in seen:chosen.append(byid[x['image_id']]);seen.add(x['image_id'])
  if len(chosen)==12:break
 canvas=Image.new('RGB',(1440,1120),'#152129');draw=ImageDraw.Draw(canvas)
 for k,r in enumerate(chosen):
  im=Image.open(r['prepared_image']).convert('RGB');im.thumbnail((460,235));d=ImageDraw.Draw(im)
  for ann in anns[r['image_id']]:
   for seg in ann['segmentation']:
    q=np.asarray(seg).reshape(-1,2)*[im.width/r['source_width'],im.height/r['source_height']];pts=[tuple(p)for p in q];d.line(pts+[pts[0]],fill='#00ff88',width=2)
  for line in Path(r['prepared_label']).read_text().splitlines():
   v=line.split();q=np.asarray([float(x)for x in v[1:]]).reshape(-1,2)*[im.width,im.height];pts=[tuple(p)for p in q];d.line(pts+[pts[0]],fill='#ff52d9',width=1)
  x=(k%3)*480+(480-im.width)//2;y=(k//3)*280+26;canvas.paste(im,(x,y))
  low=min(t['prepared_resolution_audit']['mask_iou']for t in accepted_audit if t['image_id']==r['image_id'])
  draw.text(((k%3)*480+10,(k//3)*280+5),f"{r['split']} id{r['image_id']} minIoU1280={low:.4f}",fill='white')
  draw.text(((k%3)*480+10,(k//3)*280+264),'Vert:COCO / Rose:YOLO converti',fill='white')
 canvas.save(root/'PLANCHE_CONVERSION_12_PIRES.png')
 summary['verification_passed']=not errors
 summary['retained_native_min_iou']=min(t['mask_iou']for t in accepted_audit)
 summary['retained_prepared_min_iou']=min(t['prepared_resolution_audit']['mask_iou']for t in accepted_audit)
 summary['retained_multipart_annotation_count']=sum(t['parts']>1 for t in accepted_audit)
 summary['audit_metric_scope']='Raster agreement of annotations during conversion, not model performance. multipart_min_iou includes excluded images; retained_* fields concern accepted images.'
 (root/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 result={'passed':not errors,'images_verified':len(rows),'annotations_verified':labelcount,'split_counts':dict(counts),'whole_batches_one_split':all(len(v)==1 for v in batch_splits.values()),'duplicate_pairs_verified':len(dup),'errors':errors,'all_COCO_split_annotations_equal_original':not any('COCO'in str(e)for e in errors),'source_metadata_sha256':digest(root/'COCO_ORIGINAL_ALL.json'),'prepared_images_total_bytes':sum(Path(r['prepared_image']).stat().st_size for r in rows),'device':'CPU','metrics_scope':'Data conversion and split verification, not trained model performance','contour_overlay_selection':[r['image_id']for r in chosen]}
 (root/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
 if errors:raise SystemExit(1)

if __name__=='__main__':main()
