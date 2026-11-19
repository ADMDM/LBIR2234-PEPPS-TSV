#!/usr/bin/env python3
"""Acquire official TACO originals and prepare an audited one-class YOLO segmentation set.

CPU only. Keeps all source annotations, groups entire batches and near duplicates,
retains original COCO masks for evaluation, and never invents masks from boxes.
"""
from __future__ import annotations
import argparse, collections, concurrent.futures, csv, hashlib, io, json, math, os
from pathlib import Path
import random, shutil, sys, tempfile, time, urllib.request, platform, importlib.metadata, inspect

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
cv2.setNumThreads(1)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def jsonl_read(path):
    out = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try: out.append(json.loads(line))
            except json.JSONDecodeError: pass  # A truncated final record is retried.
    return out


def image_info(path, source):
    with Image.open(path) as im:
        im.load()
        size = im.size; orient = int(im.getexif().get(274, 1))
        if im.format not in ('JPEG', 'PNG'): raise ValueError(f'format {im.format}')
        if im.mode not in ('RGB', 'RGBA', 'L', 'CMYK'): raise ValueError(f'mode {im.mode}')
        return {'decoded_width': size[0], 'decoded_height': size[1],
                'exif_orientation': orient, 'format': im.format, 'mode': im.mode,
                'source_width': source['width'], 'source_height': source['height']}


def acquire_one(source, raw_root, previous, reserve_gib):
    iid = source['id']; dest = raw_root / source['file_name']
    base = {'image_id': iid, 'file_name': source['file_name'], 'batch': Path(source['file_name']).parts[0],
            'official_original_url': source.get('flickr_url'), 'official_thumbnail_url': source.get('flickr_640_url'),
            'license_source_value': source.get('license'), 'license_status': 'provided_in_image_metadata_not_independently_verified' if source.get('license') else 'unknown_in_official_image_metadata',
            'source_width': source['width'], 'source_height': source['height']}
    if dest.exists() and previous and previous.get('status') == 'ok':
        digest = sha(dest)
        if digest == previous.get('sha256'):
            try: image_info(dest, source); return {**previous, 'reused': True}
            except Exception: pass
        return {**base, 'status': 'excluded', 'error': 'existing_original_changed_or_unreadable; not overwritten'}
    if dest.exists():
        return {**base, 'status': 'excluded', 'error': 'untracked_existing_file; not overwritten'}
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for kind, url in [('original', source.get('flickr_url')), ('thumbnail_fallback', source.get('flickr_640_url'))]:
        if not url: continue
        for attempt in range(2):
            tmp = dest.with_name(dest.name + '.part')
            try:
                if shutil.disk_usage(raw_root).free < reserve_gib * 1024**3:
                    raise RuntimeError('disk_reserve_reached')
                req = urllib.request.Request(url, headers={'User-Agent': 'TACO-research-acquisition/1.0'})
                with urllib.request.urlopen(req, timeout=25) as r, open(tmp, 'wb') as out:
                    if not r.headers.get_content_type().startswith('image/'):
                        raise ValueError('non_image_response; possible access challenge')
                    size = 0
                    while True:
                        b = r.read(1 << 20)
                        if not b: break
                        size += len(b)
                        if size > 40 * 1024**2: raise ValueError('image_exceeds_40_MiB')
                        out.write(b)
                info = image_info(tmp, source)
                srcdims = (source['width'], source['height'])
                got = (info['decoded_width'], info['decoded_height'])
                if kind == 'original' and got != srcdims and got[::-1] != srcdims:
                    raise ValueError(f'original_dimensions_mismatch {got} vs {srcdims}')
                if kind == 'thumbnail_fallback':
                    ratios = [abs(got[0] / got[1] / (srcdims[0] / srcdims[1]) - 1),
                              abs(got[1] / got[0] / (srcdims[0] / srcdims[1]) - 1)]
                    if min(ratios) > .005: raise ValueError('thumbnail_aspect_mismatch')
                digest = sha(tmp); tmp.replace(dest)
                return {**base, **info, 'status': 'ok', 'download_kind': kind, 'downloaded_url': url,
                        'sha256': digest, 'bytes': dest.stat().st_size, 'path': str(dest),
                        'downloaded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        'failed_attempts': errors, 'reused': False}
            except Exception as e:
                errors.append({'kind': kind, 'attempt': attempt + 1, 'error': str(e)})
                tmp.unlink(missing_ok=True)
                if 'disk_reserve_reached' in str(e):
                    return {**base, 'status': 'excluded', 'error': str(e), 'failed_attempts': errors}
                if attempt == 0: time.sleep(.3)
    return {**base, 'status': 'excluded', 'error': 'all_official_downloads_failed', 'failed_attempts': errors}


def download(args, data, source_root):
    raw = source_root / 'images'; raw.mkdir(exist_ok=True)
    progress = source_root / 'download_progress.jsonl'
    prev = {r['image_id']: r for r in jsonl_read(progress)}
    rows = []
    start = time.time()
    with open(progress, 'a', buffering=1) as log, concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(acquire_one, im, raw, prev.get(im['id']), args.reserve_gib) for im in data['images']]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            r = fut.result(); rows.append(r); log.write(json.dumps(r, ensure_ascii=False) + '\n')
            if i % 25 == 0 or i == len(futures):
                print(json.dumps({'download_done': i, 'total': len(futures),
                                  'ok': sum(x['status'] == 'ok' for x in rows),
                                  'original_bytes': sum(x.get('bytes', 0) for x in rows),
                                  'elapsed_seconds': round(time.time()-start)}), flush=True)
    rows.sort(key=lambda r: r['image_id'])
    write_json(source_root / 'download_manifest.json', rows)
    return rows


def strip_jpeg_exif(blob):
    """Remove only EXIF APP1 segments, preserving JPEG compressed pixels exactly."""
    if blob[:2] != b'\xff\xd8': raise ValueError('not_JPEG')
    out = bytearray(blob[:2]); p = 2
    while p < len(blob):
        start = p
        if blob[p] != 255: raise ValueError('invalid_JPEG_marker')
        while blob[p] == 255: p += 1
        marker = blob[p]; p += 1
        if marker in (0xda, 0xd9): out.extend(blob[start:]); return bytes(out)
        if marker in (0x01, *range(0xd0, 0xd8)): out.extend(blob[start:p]); continue
        n = int.from_bytes(blob[p:p+2], 'big'); end = p + n
        if not (marker == 0xe1 and blob[p+2:p+8] == b'Exif\0\0'): out.extend(blob[start:end])
        p = end
    raise ValueError('JPEG_without_end')


def normalized_image(source, row, output):
    src = Path(row['path']); sw, sh = source['width'], source['height']
    with Image.open(src) as im:
        im.load(); ori = int(im.getexif().get(274, 1)); rawsize = im.size
        aspect = sw / sh
        raw_match = abs((rawsize[0] / rawsize[1]) / aspect - 1) <= .005
        dimensions_match = rawsize == (sw, sh)
        if ori in (2,3,4,5,6,7,8):
            fixed=ImageOps.exif_transpose(im).convert('RGB')
            orientation_action='exif_transpose_to_annotation_display_grid'
            if row['download_kind']=='original' and fixed.size!=(sw,sh):raise ValueError('EXIF_display_dimensions_disagree_with_annotations')
            if abs((fixed.width/fixed.height)/aspect-1)>.005:raise ValueError('EXIF_display_aspect_mismatch')
        else:
            if row['download_kind']=='original' and rawsize!=(sw,sh):raise ValueError('raw_dimensions_disagree_with_annotations')
            fixed=im.convert('RGB')
            orientation_action='raw_pixel_grid; EXIF_orientation_0_ignored' if ori==0 else 'raw_pixel_grid_orientation_1'
        pixel_sha=hashlib.sha256(fixed.tobytes()).hexdigest()
        if max(fixed.size)>1280:fixed.thumbnail((1280,1280),Image.Resampling.LANCZOS)
        # Newly encoded training copies intentionally omit EXIF orientation and other metadata.
        fixed.save(output,'JPEG',quality=95,subsampling=0)
        with Image.open(output) as out:
            if out.mode!='RGB' or out.getexif().get(274,1)!=1:raise ValueError('prepared_RGB_EXIF_check_failed')
            w,h=out.size
            if abs((w/h)/aspect-1)>.005 or max(w,h)>1280:raise ValueError('prepared_dimensions_invalid')
            grey=np.asarray(out.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)
            dct=cv2.dct(grey)[:8,:8].reshape(-1);bits=dct[1:]>np.median(dct[1:])
            ph=sum(int(b)<<i for i,b in enumerate(bits))
    return {'prepared_width':w,'prepared_height':h,'image_transform':'RGB_JPEG_quality95_maxside1280_no_upscale',
            'orientation_action':orientation_action,'scale_x':w/sw,'scale_y':h/sh,
            'pixel_sha256':pixel_sha,'phash63':f'{ph:016x}','prepared_image_sha256':sha(output),
            'prepared_image_bytes':output.stat().st_size}


class Union:
    def __init__(self, xs): self.p = {x:x for x in xs}
    def find(self,x):
        while self.p[x] != x: self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a != b: self.p[max(a,b)] = min(a,b)


def split_groups(rows):
    batches=sorted({r['batch'] for r in rows}); groups=Union(batches); duplicates=[]
    ints=[int(r['phash63'],16) for r in rows]
    for i,a in enumerate(rows):
        for j in range(i):
            b=rows[j]; exact=a['pixel_sha256']==b['pixel_sha256'] or a['sha256']==b['sha256']; distance=(ints[i]^ints[j]).bit_count()
            if exact or distance <= 4:
                groups.union(a['batch'],b['batch'])
                duplicates.append({'image_id_a':a['image_id'],'image_id_b':b['image_id'],
                                   'batch_a':a['batch'],'batch_b':b['batch'],
                                   'exact_decoded_pixels':exact,'phash_hamming':distance,
                                   'rule':'exact_pixels_or_63bit_phash_hamming_le_4'})
    components=collections.defaultdict(list)
    for b in batches: components[groups.find(b)].append(b)
    batch_counts=collections.Counter(r['batch'] for r in rows)
    comps=[{'batches':v,'n':sum(batch_counts[b] for b in v)} for v in components.values()]
    if len(comps)<3: raise RuntimeError('duplicate_linked_batches_leave_fewer_than_three_independent_groups')
    # Deterministic bounded search over whole connected components, never individual frames.
    rng=random.Random(271828); n=len(rows); target=np.array([.70,.20,.10]); best=None
    for trial in range(100000):
        assign=[rng.choices(range(3),weights=[.7,.2,.1])[0] for _ in comps]
        counts=np.zeros(3,dtype=int)
        for c,s in zip(comps,assign): counts[s]+=c['n']
        if np.any(counts==0):continue
        loss=float(np.sum((counts/n-target)**2))
        score=(round(loss,12), -int(counts[2]), -int(counts[1]), tuple(assign))
        if best is None or score<best[0]:best=(score,assign,counts)
    if best is None: raise RuntimeError('no_three_way_split')
    names=['train','val','test']; mapping={}
    for c,s in zip(comps,best[1]):
        c['split']=names[s]
        for b in c['batches']:mapping[b]=names[s]
    return mapping,duplicates,{'seed':271828,'target_fractions':dict(zip(names,target.tolist())),
        'achieved_images':dict(zip(names,best[2].tolist())), 'components':comps,
        'independence_limit':'Entire source batches and detected near duplicates are grouped; a batch is not proof of independent capture. pHash cannot detect every repeated scene.'}


def stable_normalized(points, dimensions):
    """Preserve the intended native int32 raster vertices after YOLO float32 loading."""
    dims=np.asarray(dimensions,dtype=np.float32)
    n=np.asarray(points/np.asarray(dimensions,dtype=np.float64),dtype=np.float32)
    desired=np.floor(points).astype(np.int64)
    changes=0
    for _ in range(5):
        actual=np.floor(n*dims).astype(np.int64)
        low=actual<desired;high=actual>desired
        if not low.any() and not high.any():break
        changes+=int(low.sum()+high.sum())
        n[low]=np.nextafter(n[low],np.float32(np.inf))
        n[high]=np.nextafter(n[high],np.float32(-np.inf))
    if (n<0).any() or (n>1).any() or not np.array_equal(np.floor(n*dims).astype(np.int64),desired):
        raise ValueError('float32_native_raster_stabilization_failed')
    return n,changes,float(np.max(np.abs(n.astype(np.float64)*dimensions-points)))


def raster_compare(original_parts,joined,sw,sh):
    # A bounded crop retains exactly the same original-resolution raster pixels.
    allpoints=np.concatenate(original_parts+[joined],axis=0)
    x0=max(0,int(math.floor(allpoints[:,0].min()))-1);y0=max(0,int(math.floor(allpoints[:,1].min()))-1)
    x1=min(sw,int(math.ceil(allpoints[:,0].max()))+2);y1=min(sh,int(math.ceil(allpoints[:,1].max()))+2)
    if x1<=x0 or y1<=y0:raise ValueError('empty_polygon_crop')
    dims=(y1-y0,x1-x0);a=np.zeros(dims,np.uint8);b=np.zeros_like(a)
    offset=np.array([x0,y0],dtype=np.int32)
    for p in original_parts:cv2.fillPoly(a,[p.astype(np.int32)-offset],1)
    cv2.fillPoly(b,[joined.astype(np.int32)-offset],1)
    area=int(a.sum());add=int(np.count_nonzero((a==0)&(b!=0)));lost=int(np.count_nonzero((a!=0)&(b==0)))
    inter=area-lost;union=area+add
    return {'original_union_pixels':area,'joined_polygon_pixels':int(b.sum()),'added_pixels':add,
            'removed_pixels':lost,'mask_iou':inter/union if union else 1.,
            'added_fraction_of_union':add/max(area,1),'removed_fraction_of_union':lost/max(area,1)}


def labels_one(image, anns, width, height, path):
    from ultralytics.data.converter import merge_multi_segment
    sw,sh=image['width'],image['height']; lines=[]; audits=[]
    for ann in anns:
        segs=ann.get('segmentation')
        if not isinstance(segs,list) or not segs:raise ValueError(f'annotation_{ann["id"]}_not_polygon')
        original=[];arrays=[];clamped=0
        for seg in segs:
            a=np.asarray(seg,dtype=np.float64).reshape(-1,2)
            if len(a)<3 or not np.isfinite(a).all():raise ValueError(f'annotation_{ann["id"]}_invalid_polygon')
            if (a < -1).any() or (a[:,0]>sw+1).any() or (a[:,1]>sh+1).any():raise ValueError(f'annotation_{ann["id"]}_severely_out_of_bounds')
            if len(np.unique(a,axis=0))<3:raise ValueError(f'annotation_{ann["id"]}_degenerate')
            original.append(a.copy()); fixed=np.clip(a,[0,0],[sw,sh]);clamped+=int(np.count_nonzero(a!=fixed));arrays.append(fixed)
        fixed_segs=[a.reshape(-1).tolist() for a in arrays]
        joined=np.concatenate(merge_multi_segment(fixed_segs),axis=0) if len(segs)>1 else arrays[0]
        norm,nudges,maxshift=stable_normalized(joined,np.array([sw,sh]))
        values=['0']+[f'{v:.12f}' for v in norm.reshape(-1)]
        lines.append(' '.join(values))
        # Actual float32 read path, after serialization, not just pre-write polygons.
        parsed=np.array([float(x) for x in values[1:]],dtype=np.float32).reshape(-1,2)*np.array([sw,sh],dtype=np.float32)
        result=raster_compare(original,parsed,sw,sh)
        resized_original=[p*np.array([width/sw,height/sh]) for p in original]
        prepared_parsed=np.array([float(x) for x in values[1:]],dtype=np.float32).reshape(-1,2)*np.array([width,height],dtype=np.float32)
        prepared_result=raster_compare(resized_original,prepared_parsed,width,height)
        audits.append({'annotation_id':ann['id'],'image_id':image['id'],'parts':len(segs),**result,
            'prepared_resolution_audit':prepared_result, 'prepared_dimensions':[width,height],
            'border_coordinates_clamped':clamped,'float32_ulp_nudges':nudges,'max_coordinate_shift_px':maxshift,
            'rasterization':'OpenCV union of separately filled original components vs serialized YOLO polygon loaded float32 and multiplied float32, int32 truncation, at original dimensions; not official pycocotools COCO rasterization'})
    if len(lines)!=len(anns):raise ValueError('lost_annotation')
    path.write_text('\n'.join(lines)+'\n')
    return audits


def contact_sheet(rows, data, ann_by, root):
    multiparts={a['image_id'] for a in data['annotations'] if len(a.get('segmentation',[]))>1}
    chosen=[]
    for split in ['train','val','test']:
        candidates=[r for r in rows if r['split']==split]
        chosen.extend(candidates[:2]);chosen.extend([r for r in candidates if r['image_id'] in multiparts][:2])
    seen=set();chosen=[r for r in chosen if not (r['image_id'] in seen or seen.add(r['image_id']))]
    for r in rows:
        if len(chosen)>=12:break
        if r['image_id'] not in seen:chosen.append(r);seen.add(r['image_id'])
    canvas=Image.new('RGB',(1440,1120),'#152129');draw=ImageDraw.Draw(canvas)
    for k,r in enumerate(chosen[:12]):
        im=Image.open(r['prepared_image']).convert('RGB');im.thumbnail((460,235))
        ix=(k%3)*480+(480-im.width)//2;iy=(k//3)*280+26
        overlay=Image.new('RGBA',im.size,(0,0,0,0));od=ImageDraw.Draw(overlay)
        for a in ann_by[r['image_id']]:
            for seg in a['segmentation']:
                pts=np.asarray(seg).reshape(-1,2)*[im.width/r['source_width'],im.height/r['source_height']]
                od.polygon([tuple(p) for p in pts],fill=(0,220,180,65),outline=(0,255,200,255),width=2)
        canvas.paste(Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB'),(ix,iy))
        title=f"{r['split']} | {r['batch']} | id {r['image_id']} | {r['annotation_count']} objets"
        draw.text(((k%3)*480+10,(k//3)*280+6),title,fill='white')
        draw.text(((k%3)*480+10,(k//3)*280+264),'Masques COCO originaux; contours manuels',fill='#9fb5bd')
    canvas.save(root/'PLANCHE_12_CAS.png')
    write_json(root/'PLANCHE_SELECTION.json',[{k:r[k] for k in ['image_id','batch','split','prepared_image']} for r in chosen[:12]])


def prepare(args,data,downloads,source_root,root):
    if (root/'SUMMARY.json').exists() and not args.rebuild:
        print('Already prepared. Use --rebuild only after review.',flush=True);return
    root.mkdir(parents=True,exist_ok=True)
    pool=root/'images';labels=root/'labels';pool.mkdir(exist_ok=True);labels.mkdir(exist_ok=True)
    ann_by=collections.defaultdict(list)
    for a in data['annotations']:ann_by[a['image_id']].append(a)
    image_by={x['id']:x for x in data['images']};accepted=[];exclusions=[];audit=[]
    for i,row in enumerate(downloads,1):
        if row['status']!='ok':exclusions.append(row);continue
        im=image_by[row['image_id']]
        row['license_status']='provided_in_image_metadata_not_independently_verified' if im.get('license') else 'unknown_in_official_image_metadata'
        out=pool/f"taco_{im['id']:05d}.jpg";label=labels/(out.stem+'.txt')
        try:
            if out.exists():out.unlink()  # only this script's derived output, originals are immutable
            inf=normalized_image(im,row,out)
            checks=labels_one(im,ann_by[im['id']],inf['prepared_width'],inf['prepared_height'],label)
            audit.extend(checks)
            failures=[c['annotation_id'] for c in checks if c['mask_iou']<.95 or c['prepared_resolution_audit']['mask_iou']<.95]
            if failures:raise ValueError(f'whole_image_excluded_source_or_prepared_mask_iou_below_0.95_annotations_{failures}')
            accepted.append({**row,**inf,'prepared_image':str(out),'prepared_label':str(label),
                             'annotation_count':len(ann_by[im['id']]),'source_category_ids':sorted({a['category_id'] for a in ann_by[im['id']]})})
        except Exception as e:
            out.unlink(missing_ok=True);label.unlink(missing_ok=True)
            exclusions.append({**row,'status':'excluded_preparation','error':str(e)})
        if i%100==0:print(json.dumps({'preparation_done':i,'accepted':len(accepted),'excluded':len(exclusions)}),flush=True)
    mapping,duplicates,splitinfo=split_groups(accepted)
    for row in accepted:row['split']=mapping[row['batch']]
    accepted.sort(key=lambda r:r['image_id'])
    for split in ['train','val','test']:
        selected=[r for r in accepted if r['split']==split]; ids={r['image_id'] for r in selected}
        (root/f'{split}.txt').write_text('\n'.join(r['prepared_image'] for r in selected)+'\n')
        write_json(root/f'manifest_{split}.json',selected)
        coco={**data,'images':[im for im in data['images'] if im['id'] in ids],
              'annotations':[a for a in data['annotations'] if a['image_id'] in ids]}
        write_json(root/f'COCO_ORIGINAL_{split}.json',coco)
    (root/'data.yaml').write_text(f"path: {root}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n  0: dechet\n")
    accepted_ids={r['image_id'] for r in accepted}
    for check in audit:check['image_accepted']=check['image_id'] in accepted_ids
    multipart=[a for a in audit if a['parts']>1]
    write_json(root/'ALL_POLYGON_AUDIT.json',audit)
    write_json(root/'MULTIPOLYGON_AUDIT.json',multipart);write_json(root/'DUPLICATES_PHASH.json',duplicates)
    write_json(root/'SPLIT_GROUPS.json',splitinfo);write_json(root/'EXCLUSIONS.json',exclusions)
    write_json(root/'MANIFEST_ALL.json',accepted)
    shutil.copy2(source_root/'annotations.json',root/'COCO_ORIGINAL_ALL.json')
    source_meta=json.loads((source_root/'SOURCE.json').read_text())
    from ultralytics.data.converter import merge_multi_segment
    provenance={'source':source_meta,'script_path':str(Path(__file__).resolve()),'script_sha256':sha(__file__),
                'python':sys.version,'architecture':platform.machine(),
                'versions':{p:importlib.metadata.version(p) for p in ['ultralytics','numpy','Pillow','opencv-python']},
                'merge_multi_segment_source_sha256':hashlib.sha256(inspect.getsource(merge_multi_segment).encode()).hexdigest(),
                'original_downloaded_bytes':sum(r.get('bytes',0) for r in downloads),
                'prepared_image_bytes':sum(r.get('prepared_image_bytes',0) for r in accepted),
                'license_source_values':dict(collections.Counter(str(im.get('license')) for im in data['images'])),
                'EXIF_review':'Image559 orientation3 was visually checked against its annotation overlay: EXIF-transposed display grid matches; raw grid does not. Standard EXIF orientation applied to all valid2..8; invalid0 ignored only with source dimensions matching.',
                'preserved_source_annotations_sha256':sha(root/'COCO_ORIGINAL_ALL.json')}
    write_json(root/'PROVENANCE.json',provenance)
    summary={'images_source':len(data['images']),'annotations_source':len(data['annotations']),
             'images_prepared':len(accepted),'annotations_prepared':sum(r['annotation_count'] for r in accepted),
             'excluded_images':len(exclusions),'download_kind_counts':dict(collections.Counter(r['download_kind'] for r in accepted)),
             'transform_counts':dict(collections.Counter(r['image_transform'] for r in accepted)),
             'split':splitinfo,'duplicate_pairs':len(duplicates),'multipart_annotations_audited':len(multipart), 'all_annotations_audited':len(audit), 'source_and_prepared_iou_acceptance_threshold':.95, 'prepared_image_encoding':'RGB JPEG quality95 subsampling0 maxside1280 no upscale',
             'multipart_min_iou':min((a['mask_iou'] for a in multipart),default=1),
             'multipart_total_added_pixels':sum(a['added_pixels'] for a in multipart),
             'multipart_total_removed_pixels':sum(a['removed_pixels'] for a in multipart),
             'source_metadata':source_meta,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             'training_started':False,'device_policy':'CPU only; no tensor model/GPU invoked',
             'source_license_note':'MIT covers repository software. Official current annotations have an empty licenses list and image license may be null; not assumed to be image permission.',
             'evaluation_truth':'COCO_ORIGINAL_* files preserve each original segmentation component and category, source dimensions and URLs. Evaluate in original image coordinates. YOLO bridges are an audited training representation, not new ground truth.',
             'scope_limit':'TACO is general litter, not railway hazards. It does not establish TSV performance or exhaustive hazard coverage.'}
    contact_sheet(accepted,data,ann_by,root)
    write_json(root/'SUMMARY.json',summary)
    note = f'''TACO — préparation de segmentation, classe unique « dechet »

Point d'entrée : data.yaml.
{len(accepted)} images, {sum(r['annotation_count'] for r in accepted)} annotations conservées.
Les {len(exclusions)} images exclues et leurs annotations entières sont détaillées dans EXCLUSIONS.json.

Les originaux officiels restent dans 01_SOURCES/TACO_20261108/images.
Les copies d'apprentissage sont RGB JPEG qualité95, grand côté≤1280, sans agrandissement.
L'orientation EXIF est appliquée avant réduction ; dimensions, empreintes et transformations figurent dans les manifestes.

Toutes les annotations d'une image conservée sont réunies dans la classe « dechet ».
La conversion conserve les vrais polygones. Les composants multiples sont reliés par la fonction officielle Ultralytics merge_multi_segment.
Une image entière est exclue dès qu'une annotation a un IoU de rasterisation inférieur à0,95, à la résolution d'origine ou préparée.
Ce seuil est un choix d'ingénierie, pas une mesure de performance du futur modèle.
ALL_POLYGON_AUDIT.json et MULTIPOLYGON_AUDIT.json documentent chaque différence de masque, correction de bord et stabilisation numérique.

COCO_ORIGINAL_ALL.json et COCO_ORIGINAL_train/val/test.json gardent les composants, catégories et coordonnées d'origine pour l'évaluation.
Les comparaisons de préparation utilisent OpenCV ; elles ne sont pas des métriques COCO officielles.

La séparation train/val/test garde chaque lot batch entier et regroupe les lots reliés par des doublons de pixels ou une empreinte pHash63 à distance≤4.
Un lot ne garantit pas une capture indépendante ; les scènes semblables non détectées par pHash restent une limite.
SPLIT_GROUPS.json et DUPLICATES_PHASH.json rendent la séparation inspectable.

PROVENANCE.json et les manifestes conservent URLs, SHA-256, version du dépôt et valeur de licence par image.
Les métadonnées source contiennent des licences null, CC non précisée et ODBL OpenLitterMap. Ne pas appliquer la licence MIT du logiciel à toutes les images.

TACO décrit des déchets généraux, pas tous les dangers ferroviaires. Ces données seules ne prouvent pas une amélioration sur TSV.
Aucun entraînement n'a été lancé par ce script.
'''
    (root/'LIRE_MOI.txt').write_text(note)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--mode',choices=['download','prepare','all'],default='all');ap.add_argument('--reserve-gib',type=float,default=8)
    ap.add_argument('--rebuild',action='store_true');args=ap.parse_args()
    source=args.root/'01_SOURCES/TACO_20261108';out=args.root/'02_DONNEES/TACO_SEG_20261108'
    data=json.loads((source/'annotations.json').read_text())
    downloads=download(args,data,source) if args.mode in ('download','all') else json.loads((source/'download_manifest.json').read_text())
    image_by={im['id']:im for im in data['images']}
    for row in downloads:row['license_status']='provided_in_image_metadata_not_independently_verified' if image_by[row['image_id']].get('license') else 'unknown_in_official_image_metadata'
    write_json(source/'download_manifest.json',downloads)
    if args.mode in ('prepare','all'):prepare(args,data,downloads,source,out)


if __name__=='__main__':main()
