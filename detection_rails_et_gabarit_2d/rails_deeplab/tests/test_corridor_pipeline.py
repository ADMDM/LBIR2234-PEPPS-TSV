import json
from pathlib import Path
import numpy as np
import pytest
from railtsv.common import TASKS
from railtsv.drafts import draft_annotation
from scripts.export_labelme import rasterize


def test_curved_corridor_drafts_keep_only_corridor_label_and_need_review():
    mask=np.zeros((256,400),np.uint8)
    for y in range(256):
        left=50+y//8;mask[y,left:left+170]=1
    prob=np.eye(2,dtype=np.float32)[mask].transpose(2,0,1)
    draft=draft_annotation(mask,prob,'curved.png','checkpoint',task='corridor')
    assert TASKS['corridor']==['background','couloir']
    assert {s['label'] for s in draft['shapes']}=={'couloir'}
    assert draft['flags']['reviewed'] is False
    with pytest.raises(ValueError,match='non validée'):rasterize(draft,(400,256),task='corridor')
    draft['flags']['reviewed']=True
    result=rasterize(draft,(400,256),task='corridor')
    assert ((result==1)&(mask==1)).sum()/((result==1)|(mask==1)).sum()>.98
    with pytest.raises(ValueError):rasterize(draft,(400,256),task='tsv')


def test_auto_video_model_selection_never_falls_back_to_rail_model_for_corridor(tmp_path):
    from scripts.video_with_info import model_candidates,result_page
    old=tmp_path/'runs/essai_10_images_old/modele';old.mkdir(parents=True);(old/'last.pt').touch()
    assert model_candidates(tmp_path,'corridor')==[]
    new=tmp_path/'runs/entrainement_couloir_new/modele';new.mkdir(parents=True);(new/'best.pt').touch()
    assert model_candidates(tmp_path,'corridor')==[new/'best.pt']
    assert new/'best.pt' not in model_candidates(tmp_path,'tsv')
    page=tmp_path/'result';page.mkdir();(page/'video.json').write_text(json.dumps(dict(task='corridor')))
    rendered=result_page(page).read_text()
    assert 'deux limites du couloir' in rendered and 'épaisseur ne sont pas prédites' in rendered
