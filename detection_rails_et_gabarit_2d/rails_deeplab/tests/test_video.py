from pathlib import Path
import csv
from types import SimpleNamespace
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import pytest
import torch

from railtsv import video


class KnownRails(torch.nn.Module):
    def forward(self, x):
        n, _, h, w = x.shape
        logits = torch.zeros((n, 3, h, w), device=x.device)
        logits[:, 1, :, 16:24] = 20
        logits[:, 2, :, 72:80] = 20
        return logits


def fixture_video(tmp_path):
    source = tmp_path/'original.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (96, 64))
    assert writer.isOpened()
    for i in range(10):
        writer.write(np.full((64, 96, 3), 60+i*10, np.uint8))
    writer.release()
    return source


@pytest.mark.parametrize('limit,expected', [(0,10), (4,4)])
def test_video_every_frame_cadence_dimensions_colors_and_no_overwrite(tmp_path, monkeypatch, limit, expected):
    source = fixture_video(tmp_path)
    original = source.read_bytes()
    checkpoint = tmp_path/'model.pt'
    checkpoint.write_bytes(b'test provenance')
    monkeypatch.setattr(video, 'load_checkpoint', lambda _: dict(task='tsv',classes=['background','rail_left','rail_right'],
        state_dict={},height=64,width=96,synthetic_demo=False,training_only_demo=True))
    monkeypatch.setattr(video, 'build_model', lambda _: KnownRails())
    args = SimpleNamespace(input=str(source), checkpoint=str(checkpoint), out=str(tmp_path/'out'),
        max_frames=limit, cpu_threads=1, device='cpu')
    result = video.infer_video(args)
    assert result['frames']==expected and result['fps']==10 and result['dimensions']==[96,64]
    assert result['duration_seconds']==expected/10 and result['complete_source_video']==(limit==0)
    assert result['technical_test_only'] is True
    with (tmp_path/'out/mesures.csv').open() as handle:
        measures=list(csv.DictReader(handle))
    assert len(measures)==expected
    assert float(measures[-1]['seconds'])==(expected-1)/10
    assert float(measures[0]['apparent_width_px'])==49
    assert measures[0]['width_m']=='' and measures[0]['independent_accuracy_pct']==''
    capture=cv2.VideoCapture(str(tmp_path/'out/rails_annotes.mp4'))
    frames=[]
    while True:
        ok,frame=capture.read()
        if not ok:break
        frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    capture.release()
    assert len(frames)==expected
    assert frames[0][45,20,2]>frames[0][45,20,0]+60  # cyan, not swapped BGR
    assert frames[0][45,75,0]>frames[0][45,75,2]+60  # orange
    assert frames[-1][45,45].mean()>frames[0][45,45].mean()+15
    assert source.read_bytes()==original
    with pytest.raises(FileExistsError): video.infer_video(args)


def test_source_audio_is_retained(tmp_path, monkeypatch):
    source=fixture_video(tmp_path)
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    sound=tmp_path/'with_audio.mp4'
    subprocess.run([ffmpeg,'-v','error','-n','-i',str(source),'-f','lavfi','-i','sine=frequency=440:duration=1',
        '-map','0:v','-map','1:a','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-shortest',str(sound)],check=True)
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'test')
    monkeypatch.setattr(video,'load_checkpoint',lambda _:dict(task='tsv',classes=['background','rail_left','rail_right'],
        state_dict={},height=64,width=96,synthetic_demo=False))
    monkeypatch.setattr(video,'build_model',lambda _:KnownRails())
    video.infer_video(SimpleNamespace(input=str(sound),checkpoint=str(checkpoint),out=str(tmp_path/'audio_out'),
        max_frames=0,cpu_threads=1,device='cpu',keep_audio=True))
    inspected=subprocess.run([ffmpeg,'-hide_banner','-i',str(tmp_path/'audio_out/rails_annotes.mp4')],capture_output=True,text=True)
    assert 'Audio: aac' in inspected.stderr
    report=video.infer_video(SimpleNamespace(input=str(sound),checkpoint=str(checkpoint),out=str(tmp_path/'silent_out'),
        max_frames=0,cpu_threads=1,device='cpu'))
    inspected=subprocess.run([ffmpeg,'-hide_banner','-i',str(tmp_path/'silent_out/rails_annotes.mp4')],capture_output=True,text=True)
    assert 'Audio:' not in inspected.stderr and report['audio']=='Disabled'


def test_missing_capture_intervals_are_preserved_and_old_predictions_recovered(tmp_path, monkeypatch):
    original=fixture_video(tmp_path)
    source=tmp_path/'gaps.avi'
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg,'-v','error','-n','-i',str(original),'-vf',r'setpts=PTS+gte(N\,3)*0.2/TB',
        '-fps_mode','vfr','-c:v','libx264','-bf','0','-pix_fmt','yuv420p',str(source)],check=True)
    timing=video.probe_timing(source,10)
    assert timing['decoded_frames']==10 and timing['normalized_frames']==12
    assert video.timeline_indices(timing).tolist()==[0,1,2,2,2,3,4,5,6,7,8,9]
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'test')
    monkeypatch.setattr(video,'load_checkpoint',lambda _:dict(task='tsv',classes=['background','rail_left','rail_right'],
        state_dict={},height=64,width=96,synthetic_demo=False,training_only_demo=True))
    monkeypatch.setattr(video,'build_model',lambda _:KnownRails())
    args=SimpleNamespace(input=str(source),checkpoint=str(checkpoint),out=str(tmp_path/'normalized'),
        max_frames=0,cpu_threads=1,device='cpu')
    result=video.infer_video(args)
    assert result['frames']==12 and result['duration_seconds']==1.2 and result['source_decoded_frames']==10
    assert result['complete_source_video'] is True
    with (tmp_path/'normalized/mesures.csv').open() as handle:
        measures=list(csv.DictReader(handle))
    repeats=[r for r in measures if r['repeated_source_frame']=='True']
    assert len(repeats)==2 and all(r['apparent_width_px']=='' and r['status']=='repeated' for r in repeats)
    args.input=str(original);args.out=str(tmp_path/'old')
    video.infer_video(args)
    recovered=video.recover_video(source,tmp_path/'old/rails_annotes.mp4',tmp_path/'recovered')
    assert recovered['frames']==12 and recovered['repeated_frames']==2 and recovered['source_frames_used']==10
    assert recovered['predictions_recomputed'] is False and recovered['duration_seconds']==1.2
    assert (tmp_path/'old/rails_annotes.mp4').exists()
