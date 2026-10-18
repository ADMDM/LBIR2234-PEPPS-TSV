import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from railtsv.common import sha256, read_manifest
from railtsv.railsem import track_mapping, track_mask, paper_splits, prepare_track_dataset, tsv_initialization


def config():
    return {'labels': [{'name': 'road'}, {'name': 'rail-track'}, {'name': 'tram-track'},
                       {'name': 'rail-raised'}, {'name': 'void', 'evaluate': False}]}


def test_track_target_distinguishes_corridor_metal_and_ignore():
    raw = np.array([[0,1,2,3,4,255]], np.uint8)
    assert track_mask(raw, track_mapping(config())).tolist() == [[0,1,1,0,255,255]]
    with pytest.raises(ValueError, match='inconnus'):
        track_mask(np.array([[9]], np.uint8), track_mapping(config()))


def test_official_splits_reject_overlap_and_missing_images(tmp_path):
    for role, stem in [('train','rs00001'),('val','rs00002'),('test','rs00003')]:
        (tmp_path/f'{role}.txt').write_text(stem+'\n')
    available = {'rs00001','rs00002','rs00003'}
    assert len(paper_splits(tmp_path, available)) == 3
    (tmp_path/'test.txt').write_text('rs00001\n')
    with pytest.raises(ValueError, match='dupliquées'): paper_splits(tmp_path, available)
    (tmp_path/'test.txt').write_text('rs00004\n')
    with pytest.raises(ValueError, match='absentes'): paper_splits(tmp_path, available)


def test_real_format_preparation_uses_supplied_split_without_original_writes(tmp_path):
    raw=tmp_path/'raw';splits=tmp_path/'splits';splits.mkdir()
    for folder in ['jpgs','uint8']:(raw/folder/'rs19_val').mkdir(parents=True)
    (raw/'rs19-config.json').write_text(json.dumps(config()))
    for i, role in enumerate(['train','val','test'], 1):
        stem=f'rs{i:05d}';(splits/f'{role}.txt').write_text(stem+'\n')
        Image.fromarray(np.full((8,12,3), i*30, np.uint8)).save(raw/'jpgs/rs19_val'/f'{stem}.jpg')
        mask=np.zeros((8,12),np.uint8);mask[:,3:8]=1;mask[:2]=255
        Image.fromarray(mask).save(raw/'uint8/rs19_val'/f'{stem}.png')
    source_digests={str(p):sha256(p) for p in raw.rglob('*') if p.is_file()}
    out=prepare_track_dataset(raw,splits,tmp_path/'prepared')
    for role in ['train','val','test']:
        rows=read_manifest(out/f'{role}.csv');assert len(rows)==1
        assert set(np.unique(np.asarray(Image.open(rows[0]['mask']))))=={0,1,255}
    assert all(sha256(p)==v for p,v in source_digests.items())
    assert json.loads((out/'preparation.json').read_text())['sequences_known'] is False
    with pytest.raises(FileExistsError): prepare_track_dataset(raw,splits,out)


def test_transfer_selection_never_uses_partial_or_synthetic_run(tmp_path):
    (tmp_path/'configs').mkdir();(tmp_path/'runs').mkdir()
    assert tsv_initialization(tmp_path)[0]==['--pretrained']
    pointer=tmp_path/'configs/railsem_pretraining.json'
    pointer.write_text(json.dumps({'status':'running'}))
    with pytest.raises(ValueError,match='pas encore terminé'):tsv_initialization(tmp_path)
    checkpoint=tmp_path/'runs/best.pt'
    data={'format':'railtsv-deeplab-v1','architecture':'DeepLabV3Plus','encoder':'resnet50',
          'task':'railsem_track','synthetic_demo':False,'training_only_demo':False,'steps':3}
    torch.save(data,checkpoint)
    pointer.write_text(json.dumps({'status':'completed','checkpoint':'runs/best.pt','checkpoint_sha256':sha256(checkpoint)}))
    assert tsv_initialization(tmp_path)[0]==['--init',str(checkpoint)]
    data['synthetic_demo']=True;torch.save(data,checkpoint)
    pointer.write_text(json.dumps({'status':'completed','checkpoint':'runs/best.pt','checkpoint_sha256':sha256(checkpoint)}))
    with pytest.raises(ValueError,match='démonstration'):tsv_initialization(tmp_path)
    assert tsv_initialization(tmp_path,'imagenet')[0]==['--pretrained']


def test_training_progress_and_new_task_work_without_test_leakage(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    from railtsv import engine
    from railtsv.common import write_manifest
    monkeypatch.setattr(engine,'build_model',lambda classes=2,pretrained=False:torch.nn.Conv2d(3,classes,1))
    for role, value in [('train',80),('val',160)]:
        image=tmp_path/f'{role}.png';mask=tmp_path/f'{role}_mask.png'
        Image.fromarray(np.full((32,64,3),value,np.uint8)).save(image)
        target=np.zeros((32,64),np.uint8);target[:,20:40]=1
        Image.fromarray(target).save(mask)
        write_manifest(tmp_path/f'{role}.csv',[{'image':str(image),'mask':str(mask),'sequence':role}])
    args=SimpleNamespace(train=str(tmp_path/'train.csv'),val=str(tmp_path/'val.csv'),out=str(tmp_path/'run'),
        training_only_demo=False,epochs=1,batch_size=1,lr=1e-3,pretrained=False,init=None,
        seed=42,cpu_threads=1,device='cpu',task='railsem_track',height=32,width=64,
        synthetic_demo=True,progress_every=1)
    engine.train(args)
    checkpoint=torch.load(tmp_path/'run/best.pt',weights_only=True)
    assert checkpoint['task']=='railsem_track' and checkpoint['classes']==['background','track_area']
    assert 'lot 1/1' in capsys.readouterr().out
