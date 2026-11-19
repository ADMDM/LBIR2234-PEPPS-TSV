"""Refresh only the map rectangle of a completed presentation video.

The existing predictions and source images are not recomputed. Native frame
times and all rail/object coordinates remain unchanged. Previous delivered
files are preserved under preuves/ before an atomic replacement. This helper
is useful for a presentation repair; a normal new run uses build_video.py.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import csv
import hashlib
import json
import shutil
import time
import zipfile
import av
from PIL import ImageChops
from build_video import csv_write
from handoff_bundle import sha256
from map_view import MapView


def refresh(config, video_id):
    config=Path(config).resolve();cfg=json.loads(config.read_text())
    job=next(j for j in cfg['videos'] if j['video_id']==video_id)
    def resolve(value):
        path=Path(value);return path if path.is_absolute() else (config.parent/path).resolve()
    destination=resolve(job['output']);proof=destination.parent.parent/'preuves'/video_id
    report_path=proof/'VERIFICATION_FINALE.json';report=json.loads(report_path.read_text())
    if report['status']!='passed':raise ValueError('A verified delivery is required')
    if job['map_mode']!='africa':raise ValueError('This repair targets an AfTS route')
    if job['source_sha256']!=report['source_sha256']:raise ValueError('Source identity mismatch')
    for name in ['VIDEO_ANNOTEE.mp4','RAILS.csv','LIRE_MOI.txt']:
        if sha256(destination/name)!=report['output_files_sha256'][name]:raise ValueError('Delivery changed: '+name)
    backup=proof/'AVANT_CORRECTION_REPERE_GPS';backup.mkdir(exist_ok=False)
    shutil.copy2(report_path,backup/report_path.name)
    with (destination/'RAILS.csv').open(newline='') as file:rows=list(csv.DictReader(file))
    width,height=report['source_dimensions'];tr=report['display_transform']
    portrait=height>width or tr['crop_width']<width
    x,y,w,h=(1080,156,790,570) if portrait else (1480,156,400,570)
    view=MapView('africa',resolve(job['gps_result']),resolve(cfg['africa_network']),w,h,proof)
    output=backup/'VIDEO.partial.mp4';changed=0;clock=[];started=time.perf_counter()
    previews={0,len(rows)//4,len(rows)//2,3*len(rows)//4,len(rows)-1}
    with av.open(str(destination/'VIDEO_ANNOTEE.mp4')) as source, av.open(str(output),'w',options={'movflags':'+faststart'}) as target:
        encoder=target.add_stream('libx264',rate=source.streams.video[0].average_rate)
        encoder.width=1920;encoder.height=1080;encoder.pix_fmt='yuv420p'
        encoder.time_base=encoder.codec_context.time_base=Fraction(1,1000000)
        encoder.codec_context.thread_count=4;encoder.options={'crf':'18','preset':'veryfast'}
        for index,frame in enumerate(source.decode(video=0)):
            if index>=len(rows):raise ValueError('Unexpected extra frame')
            row=rows[index];seconds=float(row['video_time_s'])
            if abs(float(frame.pts*frame.time_base)-seconds)>2e-6:raise ValueError('Input clock mismatch')
            canvas=frame.to_image();before=canvas.copy() if index in previews else None
            panel,status=view.image(seconds);canvas.paste(panel,(x,y))
            if before is not None:
                bbox=ImageChops.difference(before,canvas).getbbox()
                if bbox and not (bbox[0]>=x and bbox[1]>=y and bbox[2]<=x+w and bbox[3]<=y+h):
                    raise ValueError('A pixel outside the map rectangle was modified before encoding')
                canvas.save(proof/f'APERCU_{index:06d}.jpg',quality=91)
            changed+=status!=row['map_display_status'];row['map_display_status']=status
            encoded=av.VideoFrame.from_image(canvas);encoded.pts=round(seconds*1000000);encoded.time_base=Fraction(1,1000000)
            for packet in encoder.encode(encoded):target.mux(packet)
            clock.append(seconds)
            if (index+1)%1000==0:print(video_id,': carte',index+1,'/',len(rows),flush=True)
        for packet in encoder.encode():target.mux(packet)
    if len(clock)!=len(rows):raise ValueError('Incomplete input')
    count=0;max_error=0
    with av.open(str(output)) as verify:
        for index,frame in enumerate(verify.decode(video=0)):
            if index>=len(clock):raise ValueError('Unexpected encoded frame')
            error=abs(float(frame.pts*frame.time_base)-clock[index]);max_error=max(max_error,error)
            if error>2e-6:raise ValueError('Output clock changed')
            count+=1
    if count!=len(rows):raise ValueError('Incomplete encoded output')
    for name in ['VIDEO_ANNOTEE.mp4','RAILS.csv','LIRE_MOI.txt']:
        shutil.move(str(destination/name),str(backup/name))
    output.rename(destination/'VIDEO_ANNOTEE.mp4');csv_write(destination/'RAILS.csv',rows)
    note=(backup/'LIRE_MOI.txt').read_text().rstrip()
    note+='\n\nREPÈRE SUR LA CARTE\nDeux lectures GPS identiques conservent un repère fixe entre ces lectures,\nsans en déduire que la vitesse réelle du train est nulle. Les lacunes de plus\nde 5 secondes restent masquées. Les positions entre lectures sont estimées.\n'
    (destination/'LIRE_MOI.txt').write_text(note)
    report['map_refresh']={'frames':count,'display_statuses_updated':changed,'only_map_rectangle_changed_before_encoding':True,'encoding_crf':18,'native_times_verified':True,'max_timestamp_error_s':max_error,'runtime_s':time.perf_counter()-started,'rails_and_object_coordinates_unchanged':True}
    for name in report['output_files_sha256']:report['output_files_sha256'][name]=sha256(destination/name)
    archive=destination.with_suffix('.zip');temporary=backup/'RELEASE.partial.zip'
    with zipfile.ZipFile(temporary,'x') as zipped:
        for file in sorted(destination.iterdir()):
            zipped.write(file,file.name,compress_type=zipfile.ZIP_STORED if file.suffix=='.mp4' else zipfile.ZIP_DEFLATED,compresslevel=5)
    with zipfile.ZipFile(temporary) as zipped:
        for name,expected in report['output_files_sha256'].items():
            digest=hashlib.sha256()
            with zipped.open(name) as file:
                for data in iter(lambda:file.read(4*1024*1024),b''):digest.update(data)
            if digest.hexdigest()!=expected:raise ValueError('Archive mismatch: '+name)
    shutil.move(str(archive),str(backup/'LIVRAISON_AVANT.zip'));temporary.rename(archive)
    report.update(archive_sha256=sha256(archive),archive_bytes=archive.stat().st_size)
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'video_id':video_id,'frames':count,'statuses_updated':changed,'status':'passed'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--video-id',required=True)
    args=parser.parse_args();refresh(args.config,args.video_id)
