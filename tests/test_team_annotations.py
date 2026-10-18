import json
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from railtsv.common import sha256
from scripts.prepare_team_annotations import plan_indices
from scripts.merge_team_annotations import inspect,merge
from scripts.receive_team_annotations import receive,returned_json
from railtsv.annotation_config import latest_returns


def test_new_indices_are_unique_exclude_old_and_spread_across_capture():
    timestamps=np.arange(5400)/30
    old=list(range(0,5400,450))
    indices=plan_indices(timestamps,50,old)
    assert len(indices)==50 and len(set(indices))==50 and not set(indices)&set(old)
    assert timestamps[indices[0]]<5 and timestamps[indices[-1]]>175
    assert np.diff(timestamps[indices]).min()>1
    assert all(abs(timestamps[i]-timestamps[j])>.89 for i in indices for j in old)
    assert not set(indices[::2])&set(indices[1::2])


def fixture_batch(tmp_path):
    batch=tmp_path/'team';batch.mkdir()
    for lot_id,stem in [('A','source_frame_000000001'),('B','source_frame_000000002')]:
        lot=batch/f'lot_{lot_id}';(lot/'images').mkdir(parents=True);(lot/'json').mkdir()
        image=lot/'images'/f'{stem}.png'
        Image.new('RGB',(100,60),'gray').save(image)
        row=dict(stem=stem,sequence='original_capture',assignee=f'lot_{lot_id}',source_frame_index=1,
                 seconds=1.,image_sha256=sha256(image),image='images/'+image.name)
        (lot/'manifest.json').write_text(json.dumps(dict(batch_id=batch.name,lot_id=lot_id,images=[row])))
        data=dict(imageWidth=100,imageHeight=60,imagePath='/foreign/computer/'+image.name,imageData='unused',
                  flags={'reviewed':True},shapes=[dict(label='rail_left',shape_type='polygon',points=[[10,0],[20,0],[20,60],[10,60]]),
                        dict(label='rail_right',shape_type='polygon',points=[[60,0],[70,0],[70,60],[60,60]])])
        (lot/'json'/f'{stem}.json').write_text(json.dumps(data))
    (batch/'preparation.json').write_text(json.dumps(dict(sequence='original_capture',source='/source.avi',source_sha256='sourcehash')))
    return batch


def test_merge_preserves_grouping_sources_and_returned_json_and_refuses_overwrite(tmp_path):
    batch=fixture_batch(tmp_path)
    originals={p:p.read_bytes() for p in batch.rglob('*.json')}
    out=tmp_path/'merged'
    merge(batch,out)
    assert len(list((out/'json').glob('*.json')))==2
    for path in (out/'json').glob('*.json'):
        data=json.loads(path.read_text())
        assert data['imagePath']=='../images/'+path.stem+'.png' and data['imageData'] is None
    assert 'original_capture' in (out/'sequences.csv').read_text()
    assert all(p.read_bytes()==raw for p,raw in originals.items())
    with pytest.raises(FileExistsError):merge(batch,out)


def test_foreign_lot_annotation_and_modified_canonical_image_are_rejected(tmp_path):
    batch=fixture_batch(tmp_path)
    foreign=batch/'lot_A/json/wrong.json';foreign.write_text('{}')
    with pytest.raises(ValueError,match='non attribués'):inspect(batch)
    foreign.unlink()
    Image.new('RGB',(100,60),'red').save(next((batch/'lot_A/images').glob('*.png')))
    with pytest.raises(ValueError,match='modifiée'):inspect(batch)


def test_unreviewed_work_cannot_be_silently_used_as_completed_training_data(tmp_path):
    batch=fixture_batch(tmp_path)
    path=next((batch/'lot_A/json').glob('*.json'))
    data=json.loads(path.read_text());data['flags']['reviewed']=False;path.write_text(json.dumps(data))
    _,report=inspect(batch)
    assert report['A']['reviewed_valid']==0 and report['A']['invalid']
    with pytest.raises(ValueError,match='complets'):merge(batch,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_received_zip_is_staged_without_arbitrary_path_extraction_and_checked(tmp_path):
    batch=fixture_batch(tmp_path)
    annotation=next((batch/'lot_B/json').glob('*.json'))
    archive=tmp_path/'return.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('../../'+annotation.name,annotation.read_bytes())
        z.writestr('../../unrelated.txt','do not extract')
    staged,report=receive(batch,archive,tmp_path/'received')
    assert (staged/'json'/annotation.name).read_bytes()==annotation.read_bytes()
    assert report['B']['reviewed_valid']==1
    assert not (tmp_path/'unrelated.txt').exists()
    with pytest.raises(FileExistsError):receive(batch,archive,tmp_path/'received')


def test_duplicate_json_in_return_zip_is_rejected(tmp_path):
    archive=tmp_path/'duplicate.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('json/assigned.json','{}');z.writestr('another/assigned.json','{}')
    with pytest.raises(ValueError,match='même nom'):returned_json(archive,{'assigned'})


def test_third_lot_is_received_and_other_batches_do_not_hide_valid_returns(tmp_path):
    batch=fixture_batch(tmp_path)
    source=batch/'lot_B';lot=batch/'lot_C';(lot/'images').mkdir(parents=True);(lot/'json').mkdir()
    original=json.loads((source/'manifest.json').read_text())['images'][0]
    stem='other_capture_frame_000000003';image=lot/'images'/(stem+'.png')
    Image.new('RGB',(100,60),'green').save(image)
    row=dict(original,stem=stem,assignee='lot_C',sequence='another_capture',image_sha256=sha256(image))
    (lot/'manifest.json').write_text(json.dumps(dict(batch_id=batch.name,lot_id='C',images=[row])))
    data=json.loads(next((source/'json').glob('*.json')).read_text())
    (lot/'json'/(stem+'.json')).write_text(json.dumps(data))
    root=tmp_path/'project';ret=root/'data/retours';ret.mkdir(parents=True)
    staged,report=receive(batch,lot/'json',ret/'lot_C_older_valid','C')
    assert set(report)=={'A','B','C'} and report['C']['reviewed_valid']==1
    # A newer folder from another batch must be filtered BEFORE choosing latest.
    wrong=ret/'lot_C_newer_wrong';(wrong/'json').mkdir(parents=True)
    (wrong/'reception.json').write_text(json.dumps(dict(batch_id='another_batch',lot_id='C')))
    assert latest_returns(root,batch)=={'C':staged/'json'}
    with pytest.raises(ValueError,match='attribué'):receive(batch,lot/'json',ret/'wrong_lot','B')
