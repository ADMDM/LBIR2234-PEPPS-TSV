import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch
from railtsv.common import mask_read, check_disjoint, new_dir, write_manifest
from railtsv.data import letterbox, RailDataset, augmentation
from railtsv.metrics import Metrics
from railtsv.model import dice_ce
from railtsv.prepare import label_mapping, remap, prepare_railsem, split_dataset, extract_frames


def config():
    names = ["road", "sidewalk", "construction", "tram-track", "fence", "pole", "traffic-light", "traffic-sign",
             "vegetation", "terrain", "sky", "human", "rail-track", "car", "truck", "trackbed", "on-rails", "rail-raised", "rail-embedded"]
    return {"labels": [{"name": name, "evaluate": True} for name in names]}


def test_physical_rails_not_track_area():
    ids, rails, ignored = label_mapping(config())
    assert rails == {17, 18}
    x = np.array([[0, 3, 12, 15, 16, 17, 18, 255]], np.uint8)
    assert remap(x, ids, rails, ignored).tolist() == [[0, 0, 0, 0, 0, 1, 1, 255]]
    with pytest.raises(ValueError, match="absents"):
        remap(np.array([[42]], np.uint8), ids, rails, ignored)


def test_ignore_only_loss_and_no_gradient_from_ignore():
    logits = torch.randn(1, 2, 4, 4, requires_grad=True)
    mask = torch.full((1, 4, 4), 255, dtype=torch.long)
    loss = dice_ce(logits, mask)
    assert loss.item() == 0
    loss.backward()
    assert logits.grad.abs().sum() == 0
    logits.grad.zero_()
    mask[:, 2:] = 1
    dice_ce(logits, mask).backward()
    assert logits.grad[:, :, :2].abs().sum() == 0
    assert logits.grad[:, :, 2:].abs().sum() > 0


def test_metrics_count_false_positives_and_ignore():
    m = Metrics(["background", "rail"])
    m.update(np.array([[0, 1, 1, 0]]), np.array([[0, 1, 0, 255]]))
    result = m.result()
    assert result["per_class"]["rail"]["iou"] == .5
    assert result["per_class"]["rail"]["dice"] == pytest.approx(2/3)
    assert result["valid_pixels"] == 3
    empty = Metrics(["background", "rail"])
    empty.update(np.array([0]), np.array([0]))
    assert empty.result()["foreground_dice"] is None


def test_palette_mask_and_rgb_rejection(tmp_path):
    mask = Image.fromarray(np.array([[0, 1, 255]], np.uint8)).convert("P")
    mask.save(tmp_path / "mask.png")
    assert mask_read(tmp_path / "mask.png").tolist() == [[0, 1, 255]]
    mask.convert("RGB").save(tmp_path / "rgb.png")
    with pytest.raises(ValueError, match="RGB"):
        mask_read(tmp_path / "rgb.png")


def test_letterbox_preserves_ratio_and_ignore_padding():
    image = np.zeros((100, 200, 3), np.uint8)
    mask = np.ones((100, 200), np.uint8)
    x, y, roi = letterbox(image, 96, 96, mask)
    assert roi == (24, 0, 48, 96)
    assert x.shape == (96, 96, 3)
    assert (y[:24] == 255).all() and (y[24:72] == 1).all()
    with pytest.raises(ValueError):
        letterbox(image, 95, 96)


def test_augmentation_does_not_invent_labels():
    image = np.zeros((64, 96, 3), np.uint8)
    mask = np.zeros((64, 96), np.uint8)
    mask[:, 20:23], mask[:, 70:73], mask[:8] = 1, 2, 255
    transform = augmentation(42)
    for _ in range(20):
        result = transform(image=image, mask=mask)
        assert set(np.unique(result["mask"])) <= {0, 1, 2, 255}


def test_prepare_end_to_end_and_no_overwrite(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir(); masks.mkdir()
    Image.fromarray(np.zeros((2, 4, 3), np.uint8)).save(images / "rs00000.jpg")
    Image.fromarray(np.array([[17, 18, 12, 255], [3, 15, 0, 16]], np.uint8)).save(masks / "rs00000.png")
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config()))
    args = SimpleNamespace(images=images, masks=masks, config=cfg, sequences=None, out=tmp_path / "prepared")
    prepare_railsem(args)
    assert mask_read(args.out / "masks/rs00000.png").tolist() == [[1, 1, 0, 255], [0, 0, 0, 0]]
    with pytest.raises(FileExistsError):
        prepare_railsem(args)
    ds = RailDataset(args.out / "all.csv", height=64, width=96)
    x, y = ds[0]
    assert x.shape == (3, 64, 96) and y.dtype == torch.int64


def test_split_rejects_unknown_sequences(tmp_path):
    im, mask = tmp_path / "im.png", tmp_path / "mask.png"
    Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(im)
    Image.fromarray(np.zeros((2, 2), np.uint8)).save(mask)
    manifest = tmp_path / "all.csv"
    write_manifest(manifest, [dict(image=str(im), mask=str(mask), sequence="UNKNOWN")])
    args = SimpleNamespace(manifest=manifest, allow_image_split=False)
    with pytest.raises(ValueError, match="inconnues"):
        split_dataset(args)


def test_content_leakage_and_group_leakage(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"same-content"); b.write_bytes(a.read_bytes())
    train = [dict(image=str(a), mask="ma", sequence="A")]
    val = [dict(image=str(b), mask="mb", sequence="B")]
    with pytest.raises(ValueError, match="SHA256"):
        check_disjoint(train, val)
    val[0]["sequence"] = "A"
    with pytest.raises(ValueError, match="séquence"):
        check_disjoint(train, val)


def test_extract_sampling_and_preservation(tmp_path):
    import cv2
    video = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 32))
    assert writer.isOpened()
    for i in range(20):
        writer.write(np.full((32, 64, 3), i*10, np.uint8))
    writer.release()
    args = SimpleNamespace(video=video, out=tmp_path / "frames", every=.5, max_frames=3)
    extract_frames(args)
    report = json.loads((args.out / "extraction.json").read_text())
    assert [r["frame_index"] for r in report["frames"]] == [0, 5, 10]
    with pytest.raises(FileExistsError):
        extract_frames(args)
