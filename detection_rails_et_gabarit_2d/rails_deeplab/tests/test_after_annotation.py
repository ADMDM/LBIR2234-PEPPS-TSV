import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from railtsv.common import sha256,read_manifest
from scripts.after_annotation import collect,snapshot


def fixture_project(tmp_path):
    root=tmp_path/'project';batch=root/'data/annotation_team_v2_50'
    shapes=[dict(label='rail_left',shape_type='polygon',points=[[5,0],[9,0],[9,32],[5,32]]),
            dict(label='rail_right',shape_type='polygon',points=[[40,0],[44,0],[44,32],[40,32]])]
    def annotation(image,json_path):
        json_path.write_text(json.dumps(dict(flags={'reviewed':True},imageWidth=64,imageHeight=32,
                                             shapes=shapes,imagePath=image.name)))
    for lot_id,number in [('A',1),('B',2)]:
        lot=batch/f'lot_{lot_id}';(lot/'images').mkdir(parents=True);(lot/'json').mkdir()
        stem=f'capture_frame_{number:09d}';image=lot/'images'/f'{stem}.png'
        Image.fromarray(np.full((32,64,3),50+number,np.uint8)).save(image)
        annotation(image,lot/'json'/f'{stem}.json')
        row=dict(stem=stem,sequence='capture',assignee=f'lot_{lot_id}',image_sha256=sha256(image))
        (lot/'manifest.json').write_text(json.dumps(dict(batch_id=batch.name,lot_id=lot_id,images=[row])))
    (batch/'preparation.json').write_text(json.dumps(dict(sequence='capture')))
    old=root/'data/annotation_videos_v1';(old/'images').mkdir(parents=True);(old/'json').mkdir()
    rows=[]
    for stem,sequence,color in [('old','capture',70),('heldout','other_capture',90)]:
        image=old/'images'/f'{stem}.png';Image.fromarray(np.full((32,64,3),color,np.uint8)).save(image)
        annotation(image,old/'json'/f'{stem}.json')
        rows.append(dict(stem=stem,sequence=sequence))
    with (old/'sequences.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['stem','sequence']);writer.writeheader();writer.writerows(rows)
    return root


def test_all_new_and_old_annotations_are_used_and_validation_is_separate(tmp_path):
    root=fixture_project(tmp_path)
    originals={p:p.read_bytes() for p in root.rglob('*.json')}
    records,report=collect(root)
    assert report['ready'] and report['training_annotations']==3 and report['validation_annotations']==1
    out=snapshot(records,report,tmp_path/'new_run')
    train=read_manifest(out/'dataset/train.csv');val=read_manifest(out/'dataset/val.csv')
    assert len(train)==3 and len(val)==1
    assert {r['sequence'] for r in train}=={'capture'} and {r['sequence'] for r in val}=={'other_capture'}
    for row in train+val:
        assert set(np.unique(np.array(Image.open(row['mask']))))=={0,1,2}
    assert all(p.read_bytes()==raw for p,raw in originals.items())
    with pytest.raises(FileExistsError):snapshot(records,report,out)


def test_missing_heldout_annotations_block_training_before_creating_output(tmp_path):
    root=fixture_project(tmp_path)
    (root/'data/annotation_videos_v1/json/heldout.json').unlink()
    records,report=collect(root)
    assert not report['ready'] and report['validation_annotations']==0
    with pytest.raises(ValueError,match='incomplètes'):snapshot(records,report,tmp_path/'blocked')
    assert not (tmp_path/'blocked').exists()


def test_identical_images_across_captures_are_rejected_even_with_other_names(tmp_path):
    root=fixture_project(tmp_path)
    old=root/'data/annotation_videos_v1/images'
    (old/'heldout.png').write_bytes((old/'old.png').read_bytes())
    with pytest.raises(ValueError,match='identiques'):collect(root)


def test_active_three_lots_preserve_mixed_capture_groups_and_old_annotations(tmp_path):
    root=fixture_project(tmp_path);batch=root/'data/annotation_team_v2_50'
    new=root/'data/mixed_three_lots';batch.rename(new)
    for path in new.glob('lot_*/manifest.json'):
        data=json.loads(path.read_text());data['batch_id']=new.name;path.write_text(json.dumps(data))
    lot=new/'lot_C';(lot/'images').mkdir(parents=True);(lot/'json').mkdir()
    image=lot/'images/additional_frame_000000003.png'
    Image.new('RGB',(64,32),'green').save(image)
    data=json.loads(next((new/'lot_B/json').glob('*.json')).read_text())
    (lot/'json'/(image.stem+'.json')).write_text(json.dumps(data))
    row=dict(stem=image.stem,sequence='nordland',assignee='lot_C',image_sha256=sha256(image))
    (lot/'manifest.json').write_text(json.dumps(dict(batch_id=new.name,lot_id='C',images=[row])))
    (new/'preparation.json').write_text(json.dumps(dict(sequence='capture',train_sequences=['capture','nordland'])))
    (root/'configs').mkdir();(root/'configs/annotation_batch.json').write_text(json.dumps(dict(batch='data/'+new.name)))
    records,report=collect(root)
    assert report['ready'] and report['training_annotations']==4 and report['validation_annotations']==1
    assert {r['sequence'] for r in records if r['role']=='train'}=={'capture','nordland'}
    assert set(report['lots'])=={'A','B','C'}


def test_partial_lots_only_use_reviewed_annotations_and_preserve_unreviewed_source(tmp_path):
    root=fixture_project(tmp_path)
    path=next((root/'data/annotation_team_v2_50/lot_B/json').glob('*.json'))
    data=json.loads(path.read_text());data['flags']['reviewed']=False
    path.write_text(json.dumps(data));raw=path.read_bytes()
    _,strict=collect(root)
    assert not strict['ready']
    records,report=collect(root,allow_partial_lots=True)
    assert report['ready'] and report['partial_lots_authorized'] and not report['lots_complete']
    assert report['training_annotations']==2 and report['validation_annotations']==1
    assert len(report['exclusions'])==1 and report['exclusions'][0]['lot']=='B'
    out=snapshot(records,report,tmp_path/'partial')
    assert len(read_manifest(out/'dataset/train.csv'))==2
    assert path.read_bytes()==raw and json.loads(path.read_text())['flags']['reviewed'] is False
    assert not any(row['original_json']==str(path) for row in json.loads((out/'sources.json').read_text()))


def test_partial_lots_still_require_complete_heldout_validation(tmp_path):
    root=fixture_project(tmp_path)
    (root/'data/annotation_videos_v1/json/heldout.json').unlink()
    records,report=collect(root,allow_partial_lots=True)
    assert not report['ready']
    with pytest.raises(ValueError,match='incomplètes'):snapshot(records,report,tmp_path/'blocked')
    assert not (tmp_path/'blocked').exists()


def test_partial_lots_still_reject_duplicate_image_leakage(tmp_path):
    root=fixture_project(tmp_path)
    old=root/'data/annotation_videos_v1/images'
    (old/'heldout.png').write_bytes((old/'old.png').read_bytes())
    with pytest.raises(ValueError,match='identiques'):collect(root,allow_partial_lots=True)
