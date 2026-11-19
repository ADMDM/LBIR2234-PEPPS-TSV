from pathlib import Path
import shutil
from railtsv.common import write_manifest, read_manifest


def test_manifest_survives_project_move(tmp_path):
    root = tmp_path / 'before'
    (root / 'data').mkdir(parents=True)
    (root / 'splits').mkdir()
    image, mask = root / 'data/image.png', root / 'data/mask.png'
    image.write_bytes(b'image')
    mask.write_bytes(b'mask')
    csv = root / 'splits/train.csv'
    write_manifest(csv, [{'image': str(image), 'mask': str(mask), 'sequence': 'sequence1'}])
    assert str(tmp_path) not in csv.read_text()
    destination = tmp_path / 'after'
    shutil.move(root, destination)
    rows = read_manifest(destination / 'splits/train.csv')
    assert Path(rows[0]['image']) == destination / 'data/image.png'
