"""New Drive frames, provenance, blind annotation files and model previews."""
import csv,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from railtsv.common import device,sha256
from railtsv.engine import predict_image
from railtsv.inference_region import predict_region
from railtsv.model import build_model,load_checkpoint
from railtsv.geometry import corridor_profile,draw_axis,draw_corridor_limits


def prepare(out):
    out=Path(out).resolve();out.mkdir(exist_ok=False,parents=True)
    run=ROOT/json.loads((ROOT/'configs/diagnostic_target.json').read_text())['run']
    checkpoint=run/'modele/best.pt';ck=load_checkpoint(checkpoint)
    model=build_model(2);model.load_state_dict(ck['state_dict']);dev=device('auto');model.to(dev).eval();torch.set_num_threads(4)
    cases=[]
    for name in ['20220803_114407.mp4','20220803_133227.mp4','20220803_134155.mp4','20220803_134951.mp4']:
        path=ROOT/'data/raw/drive_20261014'/name
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps;cap.release()
        for fraction in [.08,.28,.48,.68,.88]:cases.append((path,duration*fraction,'renfort_a_annoter','photos_20220803',None))
    path=ROOT/'data/raw/drive_20261005/video_03383.AVI'
    for sec in [31.4,35.4,39.4,43.4,47.4,51.4,67.4,71.4,75.4,79.4]:cases.append((path,sec,'renfort_a_annoter','video_03383',None))
    # Previously inspected capture: development challenge, NOT an independent final test.
    path=ROOT/'data/raw/drive_20261014/tsv_record_video.mp4'
    for sec in [20,45,90,160,240,320,380,460,550,630,715,800,880,960,1050,1140,1220,1310,1390,1480]:
        cases.append((path,sec,'controle_nouvelle_capture','tsv_record_20261005',None))
    records=[];seen=set();preview_images=[]
    for number,(path,sec,role,sequence,roi) in enumerate(cases,1):
        folder=out/role;folder.mkdir(exist_ok=True);(folder/'images').mkdir(exist_ok=True);(folder/'json').mkdir(exist_ok=True)
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000);ok,bgr=cap.read();pos=cap.get(cv2.CAP_PROP_POS_MSEC)/1000;cap.release()
        if not ok:raise RuntimeError(f'Frame indisponible : {path} {sec}')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2]
        digest=hashlib.sha256(rgb.tobytes()).hexdigest()
        if digest in seen:raise ValueError('Images décodées identiques dans le nouveau lot')
        seen.add(digest);stem=f'{number:02d}_{path.stem}_{sec:08.2f}s'
        image=folder/'images'/(stem+'.png');Image.fromarray(rgb).save(image)
        annotation=dict(version='5.5.0',flags=dict(reviewed=False,aiguillage=False,virage=False,route_ambiguous=False),shapes=[],imagePath='../images/'+image.name,imageData=None,imageHeight=h,imageWidth=w)
        (folder/'json'/(stem+'.json')).write_text(json.dumps(annotation,indent=2))
        mask,probs=predict_image(model,rgb,dev,ck['height'],ck['width'])
        profile=corridor_profile(mask,probs)
        canvas=bgr.copy();selected=mask==1;canvas[selected]=(.8*canvas[selected]+.2*np.array([220,185,30])).astype(np.uint8)
        draw_corridor_limits(canvas,profile);draw_axis(canvas,profile)
        previews=out/'predictions_sans_verite_terrain';previews.mkdir(exist_ok=True)
        cv2.imwrite(str(previews/(stem+'.jpg')),canvas)
        record=dict(id=number,image=str(image.relative_to(ROOT)),json=str((folder/'json'/(stem+'.json')).relative_to(ROOT)),
            source=str(path.relative_to(ROOT)),source_seconds_requested=sec,decoded_seconds=pos,sequence=sequence,role=role,
            image_sha256=sha256(image),decoded_rgb_sha256=digest,width=w,height=h,axis_points=profile['points'],
            reviewed=False,has_reference=False,independent_test_score=None)
        if role=='controle_nouvelle_capture':
            # Black bars confirmed in the supplied video; inference only, coordinates restored.
            active=np.flatnonzero(np.mean(np.any(rgb>12,axis=2),axis=0)>.2)
            roi=[int(active.min()),0,int(active.max())+1,h]
            m,p=predict_region(model,rgb,dev,ck['height'],ck['width'],roi,predict_image)
            prof=corridor_profile(m,p);c=bgr.copy();s=m==1;c[s]=(.8*c[s]+.2*np.array([220,185,30])).astype(np.uint8)
            draw_corridor_limits(c,prof);draw_axis(c,prof)
            cv2.imwrite(str(previews/(stem+'_roi.jpg')),c)
            record.update(roi_xyxy_exclusive=roi,roi_axis_points=prof['points'])
        records.append(record)
        preview_images.append((f'{number:02d}  {path.stem} {sec:.1f}s',Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))))
        print(number,'/',len(cases),stem,profile['points'],'points',flush=True)
    config='auto_save: false\nflags: [reviewed, aiguillage, virage, route_ambiguous]\nlabels: [couloir, ignore]\nlabel_flags: {}\n'
    for role in ['renfort_a_annoter','controle_nouvelle_capture']:
        folder=out/role;(folder/'labelme.yaml').write_text(config)
    for start in range(0,len(preview_images),10):
        sheet=Image.new('RGB',(1280,5*235),'white');d=ImageDraw.Draw(sheet)
        for i,(title,im) in enumerate(preview_images[start:start+10]):
            x=i%2*640;y=i//2*235;sheet.paste(ImageOps.contain(im,(630,205)),(x,y+26));d.text((x+5,y+5),title,fill='black')
        sheet.save(out/f'planche_predictions_{start//10+1}.jpg',quality=91)
    (out/'MANIFESTE.json').write_text(json.dumps(dict(checkpoint_sha256=sha256(checkpoint),cases=records,
        train_candidates=30,development_control_candidates=20,all_reviewed=False,training_performed=False,
        control_scope='development_exposed',predictions_seen_before_freeze=True,
        note='No accuracy without human references. Same-day clips and TSV_record are development data. Export defaults to challenge_dev; final designation requires verified source registry.'),indent=2))
    with (out/'A_ANNOTER.csv').open('x',newline='') as f:
        fields=['id','role','sequence','source','source_seconds_requested','image','json','reviewed'];writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)


if __name__=='__main__':prepare(sys.argv[1])
