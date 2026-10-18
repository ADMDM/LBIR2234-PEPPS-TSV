import json
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from railtsv.common import write_manifest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import export_reviewed_challenge as module

def setup_case(tmp_path,monkeypatch):
    root=tmp_path/'project';root.mkdir();(root/'configs').mkdir();(root/'run/dataset').mkdir(parents=True)
    (root/'configs/diagnostic_target.json').write_text(json.dumps({'run':'run'}))
    for i in range(4):Image.fromarray(np.full((16,16,3),i*50,np.uint8)).save(root/f'image{i}.png')
    for split,i in [('train',0),('val',1)]:
        Image.fromarray(np.zeros((16,16),np.uint8)).save(root/f'mask{i}.png')
        write_manifest(root/'run/dataset'/f'{split}.csv',[dict(image=str(root/f'image{i}.png'),mask=str(root/f'mask{i}.png'),sequence=f'capture{i}')])
    records=[]
    for i,role in [(2,'renfort_a_annoter'),(3,'controle_nouvelle_capture')]:
        annotation=dict(task='corridor',flags={'reviewed':True},imageWidth=16,imageHeight=16,shapes=[dict(label='couloir',shape_type='polygon',points=[[4,1],[10,1],[10,15],[4,15]])])
        (root/f'image{i}.json').write_text(json.dumps(annotation));records.append(dict(id=i,image=f'image{i}.png',json=f'image{i}.json',image_sha256=module.sha256(root/f'image{i}.png'),role=role,sequence=f'capture{i}',source='source.mp4'))
    batch=root/'batch';batch.mkdir();(batch/'MANIFESTE.json').write_text(json.dumps({'cases':records}))
    monkeypatch.setattr(module,'ROOT',root)
    return root,batch

def test_historical_controls_are_never_final_by_default(tmp_path,monkeypatch):
    root,batch=setup_case(tmp_path,monkeypatch);out=root/'export'
    module.export(batch,out)
    assert (out/'challenge_dev.csv').exists()
    assert not (out/'test.csv').exists()
    assert json.loads((out/'EXPORT.json').read_text())['control_split']=='challenge_dev'

def test_final_export_fails_closed_for_used_capture(tmp_path,monkeypatch):
    root,batch=setup_case(tmp_path,monkeypatch)
    registry=root/'registry.json';registry.write_text(json.dumps([dict(capture_id='capture3',role='development',identity_verified=True,capture_group='trip3',source_sha256='abc',predictions_seen_before_freeze=True)]))
    with pytest.raises(ValueError,match='non réservée'):module.export(batch,root/'bad_export',registry)
    assert not (root/'bad_export').exists()
