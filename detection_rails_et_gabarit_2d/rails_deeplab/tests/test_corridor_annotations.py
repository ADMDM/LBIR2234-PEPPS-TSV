import copy
import csv
import json
import numpy as np
import pytest
from PIL import Image
from railtsv.common import sha256,read_manifest
from scripts.prepare_corridor_annotations import convert_rails_annotation
from scripts.export_labelme import rasterize
from scripts.after_annotation import collect,snapshot


def test_conversion_preserves_sources_and_keeps_gaps_and_ignore():
    data=dict(flags={'reviewed':True},imageWidth=60,imageHeight=32,imagePath='x.png',shapes=[
        dict(label=label,shape_type='polygon',points=[[a,y0],[b,y0],[b,y1],[a,y1]])
        for label,a,b,y0,y1 in [('rail_left',8,10,0,10),('rail_left',8,10,14,31),
                              ('rail_right',40,42,0,10),('rail_right',40,42,14,31),
                              ('ignore',15,35,20,22)]])
    original=copy.deepcopy(data);converted=convert_rails_annotation(data,(60,32))
    assert data==original and converted['flags']['reviewed'] is False
    assert {s['label'] for s in converted['shapes']}=={'couloir','ignore'}
    converted['flags']['reviewed']=True
    mask=rasterize(converted,(60,32),task='corridor')
    assert np.all(mask[11:14]==0)
    assert np.all(mask[20:23,15:36]==255)
    assert mask[5,20]==1 and mask[5,5]==0


def fixture_corridor(root):
    batch=root/'data/corridor_batch';(root/'configs').mkdir(parents=True)
    (root/'configs/annotation_batch.json').write_text(json.dumps(dict(batch='data/corridor_batch',task='corridor')))
    shape=dict(label='couloir',shape_type='polygon',points=[[10,0],[45,0],[45,31],[10,31]])
    def make(session,stem,color,sequence,reviewed):
        (session/'images').mkdir(parents=True,exist_ok=True);(session/'json').mkdir(exist_ok=True)
        image=session/'images'/(stem+'.png');Image.new('RGB',(64,32),color).save(image)
        data=dict(task='corridor',flags={'reviewed':reviewed},imageWidth=64,imageHeight=32,shapes=[shape])
        (session/'json'/(stem+'.json')).write_text(json.dumps(data))
        return dict(stem=stem,sequence=sequence,image_sha256=sha256(image))
    for i,letter in enumerate('ABC'):
        lot=batch/f'lot_{letter}';row=make(lot,'train_'+letter,(30+i*20,40,70),'capture',True)
        row['assignee']='lot_'+letter
        (lot/'manifest.json').write_text(json.dumps(dict(task='corridor',lot_id=letter,batch_id=batch.name,images=[row])))
    for session,stem,color,seq,reviewed in [(batch/'validation','heldout','green','second',True),
                                         (batch/'supplemental','old','red','capture',False)]:
        row=make(session,stem,color,seq,reviewed)
        with (session/'sequences.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['stem','sequence'],extrasaction='ignore');w.writeheader();w.writerow(row)
    (batch/'preparation.json').write_text(json.dumps(dict(task='corridor',sequence='capture',train_sequences=['capture'],
        validation_session='data/corridor_batch/validation',supplemental_session='data/corridor_batch/supplemental')))
    return batch


def test_corridor_collect_snapshot_does_not_mix_legacy_or_require_optional(tmp_path):
    root=tmp_path/'project';batch=fixture_corridor(root)
    records,report=collect(root)
    assert report['ready'] and report['task']=='corridor'
    assert report['training_annotations']==3 and report['validation_annotations']==1
    assert report['old_training_annotations']==0 and report['optional_invalid_annotations']
    out=snapshot(records,report,tmp_path/'new')
    rows=read_manifest(out/'dataset/train.csv')+read_manifest(out/'dataset/val.csv')
    assert len(rows)==4
    assert all(set(np.unique(np.array(Image.open(r['mask']))))=={0,1} for r in rows)
    assert json.loads((batch/'supplemental/json/old.json').read_text())['flags']['reviewed'] is False


def test_corridor_validation_unreviewed_blocks_training(tmp_path):
    root=tmp_path/'project';batch=fixture_corridor(root)
    p=batch/'validation/json/heldout.json';d=json.loads(p.read_text());d['flags']['reviewed']=False;p.write_text(json.dumps(d))
    records,report=collect(root)
    assert not report['ready'] and report['validation_annotations']==0
    with pytest.raises(ValueError,match='incomplètes'):snapshot(records,report,tmp_path/'blocked')
    assert not (tmp_path/'blocked').exists()
