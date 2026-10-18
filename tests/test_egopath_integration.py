import json
from pathlib import Path
import pytest
import torch
from railtsv.common import sha256
from railtsv.railsem import tsv_initialization

def test_egopath_is_explicit_completed_typed_and_hash_checked(tmp_path):
    (tmp_path/'configs').mkdir();(tmp_path/'runs').mkdir()
    # Default behavior must not silently pick a new experimental pretraining.
    assert tsv_initialization(tmp_path,'auto')[0]==['--pretrained']
    with pytest.raises(ValueError,match='ego-path'):tsv_initialization(tmp_path,'egopath')
    cp=tmp_path/'runs/ego.pt';pointer=tmp_path/'configs/railsem_egopath_pretraining.json'
    data=dict(format='railtsv-deeplab-v1',architecture='DeepLabV3Plus',encoder='resnet50',task='railsem_egopath',
              synthetic_demo=False,training_only_demo=False,steps=10,epoch=2)
    torch.save(data,cp)
    def state(status='completed'):
        pointer.write_text(json.dumps(dict(status=status,checkpoint='runs/ego.pt',checkpoint_sha256=sha256(cp),completed_epochs=2)))
    state('running')
    with pytest.raises(ValueError,match='pas encore terminé'):tsv_initialization(tmp_path,'egopath')
    state();args,info=tsv_initialization(tmp_path,'egopath')
    assert args==['--init',str(cp)] and info['source']=='railsem19_egopath'
    assert tsv_initialization(tmp_path,'auto')[0]==['--pretrained']
    data['task']='corridor';torch.save(data,cp)
    with pytest.raises(ValueError,match='modifié'):tsv_initialization(tmp_path,'egopath')
    state()
    with pytest.raises(ValueError,match='incompatible'):tsv_initialization(tmp_path,'egopath')
