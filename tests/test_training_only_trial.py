import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch
from railtsv import engine
from railtsv.common import write_manifest
from scripts.first_annotation_trial import prepare


def args_for(tmp_path):
    image = tmp_path / 'image.png'
    mask = tmp_path / 'mask.png'
    Image.fromarray(np.full((32, 64, 3), 128, np.uint8)).save(image)
    target = np.zeros((32, 64), np.uint8)
    target[:, 15:19] = 1
    target[:, 45:49] = 2
    target[:3] = 255
    Image.fromarray(target).save(mask)
    manifest = tmp_path / 'all.csv'
    write_manifest(manifest, [dict(image=str(image), mask=str(mask), sequence='one_video')])
    return SimpleNamespace(train=str(manifest), val=None, out=str(tmp_path / 'run'),
        training_only_demo=True, epochs=2, batch_size=1, lr=1e-3, pretrained=False,
        init=None, seed=42, cpu_threads=1, device='cpu', task='tsv', height=32,
        width=64, synthetic_demo=False)


def test_trial_train_and_infer_without_validation_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, 'build_model', lambda classes=2, pretrained=False: torch.nn.Conv2d(3, classes, 1))
    args = args_for(tmp_path)
    engine.train(args)
    out = tmp_path / 'run'
    checkpoint = torch.load(out / 'last.pt', weights_only=True)
    assert checkpoint['steps'] == 2 and checkpoint['training_only_demo'] is True
    assert checkpoint['best_val_dice'] is None and not (out / 'best.pt').exists()
    history = json.loads((out / 'history.json').read_text())
    assert len(history) == 2 and all('metrics' not in x and 'val_loss' not in x for x in history)
    assert json.loads((out / 'run.json').read_text())['val_manifest_sha256'] is None
    inference = SimpleNamespace(checkpoint=str(out/'last.pt'), input=str(tmp_path/'image.png'),
        out=str(tmp_path/'prediction'), device='cpu', cpu_threads=1, height=None, width=None)
    engine.infer(inference)
    result = json.loads((tmp_path/'prediction/image/prediction.json').read_text())
    assert result['technical_test_only'] is True
    assert result['classes'] == ['background', 'rail_left', 'rail_right']
    assert (tmp_path/'prediction/image/overlay.png').exists()


@pytest.mark.parametrize('trial,val', [(False, None), (True, 'val.csv')])
def test_validation_mode_must_be_explicit(trial, val):
    with pytest.raises(ValueError, match='--val'):
        engine.train(SimpleNamespace(training_only_demo=trial, val=val))


def test_normal_training_still_rejects_same_video(tmp_path):
    args = args_for(tmp_path)
    args.training_only_demo = False
    args.val = args.train
    with pytest.raises(ValueError, match='séquence'):
        engine.train(args)
    assert not (tmp_path/'run').exists()


def test_trial_requires_manual_review_before_any_output(tmp_path):
    session = tmp_path/'session'
    (session/'json').mkdir(parents=True)
    annotation = session/'json/frame.json'
    original = json.dumps({'flags': {'reviewed': False}})
    annotation.write_text(original)
    with pytest.raises(ValueError, match='reviewed'):
        prepare(session, 1)
    assert annotation.read_text() == original
