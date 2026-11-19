import json
import numpy as np
from PIL import Image, UnidentifiedImageError
import pytest
from railtsv.common import sha256, mask_read, read_manifest
from scripts.trial_corridor_lot import prepare_trial


def fixture(root):
    batch=root/'data/batch';lot=batch/'lot_A'
    (lot/'images').mkdir(parents=True);(lot/'json').mkdir()
    (root/'configs').mkdir();(root/'configs/annotation_batch.json').write_text(json.dumps({'batch':'data/batch','task':'corridor'}))
    (batch/'preparation.json').write_text(json.dumps({'task':'corridor'}))
    rows=[]
    for i in range(2):
        stem=f'frame{i}';image=lot/'images'/f'{stem}.png'
        Image.fromarray(np.full((32,64,3),i*50,np.uint8)).save(image)
        data={'flags':{'reviewed':True},'imageWidth':64,'imageHeight':32,'shapes':[
            {'label':'couloir','shape_type':'polygon','points':[[20,0],[40,0],[40,32],[20,32]]}]}
        (lot/'json'/f'{stem}.json').write_text(json.dumps(data))
        rows.append({'stem':stem,'sequence':'capture','assignee':'lot_A','image_sha256':sha256(image)})
    (lot/'manifest.json').write_text(json.dumps({'batch_id':'batch','lot_id':'A','images':rows}))
    return lot


def test_one_lot_trial_preserves_originals_and_never_creates_fake_validation(tmp_path):
    root=tmp_path/'project';fixture(root)
    originals={p:p.read_bytes() for p in root.rglob('*.json')}
    out,records=prepare_trial(root,tmp_path/'trial',expected_count=2)
    assert len(records)==2 and len(read_manifest(out/'dataset/train.csv'))==2
    assert not (out/'dataset/val.csv').exists()
    assert json.loads((out/'annotations.json').read_text())['training_only_demo'] is True
    assert all(p.read_bytes()==raw for p,raw in originals.items())


def test_unreviewed_or_missing_json_blocks_before_any_output(tmp_path):
    root=tmp_path/'project';lot=fixture(root)
    p=lot/'json/frame0.json';data=json.loads(p.read_text());data['flags']['reviewed']=False;p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='non validée'):prepare_trial(root,tmp_path/'blocked',expected_count=2)
    assert not (tmp_path/'blocked').exists()


def test_transient_read_is_retried_but_unknown_ids_are_rejected(tmp_path,monkeypatch):
    p=tmp_path/'mask.png';Image.fromarray(np.array([[0,1,255]],np.uint8)).save(p)
    real=Image.open;calls=[]
    def flaky(*args,**kwargs):
        calls.append(1)
        if len(calls)<3:raise UnidentifiedImageError('temporary read')
        return real(*args,**kwargs)
    monkeypatch.setattr(Image,'open',flaky)
    assert mask_read(p,[0,1,255]).tolist()==[[0,1,255]] and len(calls)==3
    with pytest.raises(ValueError,match='invalides'):mask_read(p,[0,255])
