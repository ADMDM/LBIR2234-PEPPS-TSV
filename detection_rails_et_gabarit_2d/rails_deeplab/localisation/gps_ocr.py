"""Local Apple Vision OCR, multiple image preparations, explicit disagreement/rejection."""
from pathlib import Path
import json,subprocess,sys,time
from collections import Counter
import numpy as np
from PIL import Image
from gps_inputs import extract_gps_candidate,reject_spatial_outliers


def ensure_engine():
    if sys.platform!='darwin':raise RuntimeError('Apple Vision requires macOS; CSV mode remains portable')
    root=Path(__file__).resolve().parent;runtime=root/'.runtime';runtime.mkdir(exist_ok=True)
    binary=runtime/'ocr_vision';source=root/'ocr_vision.swift'
    if not binary.exists() or binary.stat().st_mtime<source.stat().st_mtime:
        subprocess.run(['swiftc','-module-cache-path',str(runtime/'swift-cache'),str(source),'-o',str(binary)],check=True)
    return binary


def read_samples(samples,crop_dir,config,metadata,out):
    binary=ensure_engine();started=time.perf_counter();out=Path(out)
    variants=out/'PREPARATIONS_OCR';variants.mkdir(exist_ok=False)
    jobs=[]
    for sample in samples:
        original=Path(crop_dir)/sample['crop_file'];paths=[original]
        if config.get('consensus_variants'):
            im=Image.open(original).convert('RGB');roi=config['coordinate_crop']
            im=im.crop(tuple(round(v*size) for v,size in zip(roi,[im.width,im.height,im.width,im.height])))
            color=variants/(original.stem+'_couleur.png');im.save(color);paths.append(color)
            a=np.asarray(im).astype(float)
            mask=(a[:,:,0]>130)&(a[:,:,1]>110)&(a[:,:,2]<.65*a[:,:,0])&(a[:,:,2]<.65*a[:,:,1])
            blackwhite=Image.fromarray(np.where(mask,0,255).astype('uint8'))
            masked=variants/(original.stem+'_jaune.png');blackwhite.save(masked);paths.append(masked)
        jobs.append(paths)
    flattened=[path for group in jobs for path in group]
    result=subprocess.run([str(binary)]+[str(p) for p in flattened],text=True,capture_output=True,check=True,timeout=300)
    raw=[json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if len(raw)!=len(flattened):raise RuntimeError('Missing OCR engine response')
    (out/'OCR_BRUT.jsonl').write_text(result.stdout);(out/'OCR_MOTEUR.log').write_text(result.stderr)
    responses={}
    for path,response in zip(flattened,raw):
        if Path(response.get('path','')).resolve()!=path.resolve():raise RuntimeError('OCR frame association mismatch')
        responses[str(path)]=response
    audit=[]
    for sample,paths in zip(samples,jobs):
        parsed=[];all_lines=[]
        for path in paths:
            response=responses[str(path)];lines=response.get('lines',[])
            all_lines.append({'variant':path.name,'lines':lines,'error':response.get('error')})
            candidate=extract_gps_candidate(lines,config.get('min_confidence',.75),config.get('coordinate_format','auto')) if not response.get('error') else {'status':'rejected_ocr_engine_error'}
            candidate['variant']=path.name;parsed.append(candidate)
        valid=[r for r in parsed if not r['status'].startswith('rejected')]
        counts=Counter((round(r['latitude_deg'],9),round(r['longitude_deg'],9)) for r in valid)
        if not counts:
            selected={'status':'rejected_no_valid_variant'}
        else:
            best,count=counts.most_common(1)[0]
            if len(counts)>1 and count<2:
                selected={'status':'rejected_ocr_variants_disagree'}
            else:
                selected=dict(max((r for r in valid if (round(r['latitude_deg'],9),round(r['longitude_deg'],9))==best),key=lambda r:r['ocr_confidence']))
                selected.update(ocr_agreement_count=count,ocr_valid_variant_count=len(valid),ocr_variants_tested=len(paths))
        record=dict(sample,**selected,ocr_variants_json=json.dumps(parsed,ensure_ascii=False),ocr_lines_json=json.dumps(all_lines,ensure_ascii=False))
        audit.append(record)
    reject_spatial_outliers(audit,config.get('max_speed_m_s',100),config.get('jump_tolerance_m',20),config.get('isolated_excess_m',20))
    accepted=[]
    for row in audit:
        if row['status'].startswith('rejected'):continue
        accepted.append({'video_id':metadata['video_id'],'frame_index':row['frame_index'],'video_time_s':row['video_time_s'],
                         'source_pts':row['source_pts'],'time_base':row['time_base'],'latitude_deg':row['latitude_deg'],'longitude_deg':row['longitude_deg'],
                         'gps_source':'video_overlay_apple_vision_ocr','gps_status':'ocr_consistency_checked_not_surveyed','gps_accuracy_m':'',
                         'ocr_confidence':row['ocr_confidence'],'ocr_agreement_count':row['ocr_agreement_count'],
                         'ocr_coordinate_text':row['ocr_coordinate_text'],'overlay_clock_raw':row['overlay_clock_raw'],
                         'source_video_sha256':metadata['source_video_sha256']})
    return accepted,audit,{'engine':'Apple Vision VNRecognizeTextRequest','ocr_seconds':time.perf_counter()-started,
                         'samples':len(samples),'accepted':len(accepted),'rejected':len(audit)-len(accepted),
                         'confidence_is_not_a_calibrated_probability':True,'variant_errors_can_be_correlated':True,
                         'coordinates_are_video_overlay_values_not_independent_gnss_measurements':True}
