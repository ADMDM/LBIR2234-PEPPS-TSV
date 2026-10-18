"""RailSem19 track-area pretraining, paper splits and explicit TSV transfer."""
import json
import os
from pathlib import Path
import re
import numpy as np
from PIL import Image, ImageDraw
from .common import dump, mask_read, new_dir, sha256, write_manifest

PAPER_NOTE = ('Official paper image split; source-video independence is not established. '
              'Generic track-area pretraining is not TSV ego-track evaluation.')


def track_mapping(config):
    labels = config['labels']
    ids = {item['name']: int(item.get('id', i)) for i, item in enumerate(labels)}
    if len(ids) != len(labels) or len(set(ids.values())) != len(ids):
        raise ValueError('Labels RailSem19 ambigus')
    if any(i < 0 or i >= 255 for i in ids.values()):
        raise ValueError('IDs RailSem19 invalides')
    if not {'rail-track', 'tram-track'}.issubset(ids):
        raise ValueError('rail-track et tram-track requis pour le préapprentissage')
    positive = {ids['rail-track'], ids['tram-track']}
    ignored = {ids[item['name']] for item in labels
               if item.get('evaluate') is False or item['name'] in {'void', 'ignore', 'unlabeled'}} | {255}
    if positive & ignored:
        raise ValueError('Classe cible marquée ignore dans la configuration')
    return ids, positive, ignored


def track_mask(raw, mapping):
    ids, positive, ignored = mapping
    unknown = set(np.unique(raw)) - set(ids.values()) - {255}
    if unknown:
        raise ValueError(f'IDs source inconnus : {sorted(unknown)}')
    result = np.zeros(raw.shape, np.uint8)
    result[np.isin(raw, list(positive))] = 1
    result[np.isin(raw, list(ignored))] = 255
    return result


def paper_splits(folder, available, expected=None):
    splits = {}
    seen = set()
    for role in ('train', 'val', 'test'):
        ids = (Path(folder) / f'{role}.txt').read_text().splitlines()
        if not ids or any(not re.fullmatch(r'rs\d{5}', stem) for stem in ids):
            raise ValueError(f'Liste {role} vide ou nom invalide')
        if len(ids) != len(set(ids)) or set(ids) & seen:
            raise ValueError('Images dupliquées entre/dans les listes officielles')
        if not set(ids).issubset(available):
            raise ValueError(f'Images absentes pour {role}')
        if expected and len(ids) != expected[role]:
            raise ValueError(f'Nombre inattendu dans {role}')
        splits[role] = ids
        seen.update(ids)
    return splits


def prepare_track_dataset(raw, splits_folder, output, expected=None):
    raw, splits_folder = Path(raw).resolve(), Path(splits_folder).resolve()
    config_path = raw / 'rs19-config.json'
    config = json.loads(config_path.read_text())
    mapping = track_mapping(config)
    image_dir, mask_dir = raw / 'jpgs/rs19_val', raw / 'uint8/rs19_val'
    images = {p.stem: p for p in image_dir.glob('*.jpg')}
    masks = {p.stem: p for p in mask_dir.glob('*.png')}
    if images.keys() != masks.keys():
        raise ValueError('Paires image/masque RailSem19 incomplètes')
    splits = paper_splits(splits_folder, set(images), expected)
    # Validate all identifiers before creating any output.
    out = new_dir(output)
    (out / 'masks').mkdir()
    counts, inventory, digest_sets = {}, [], {}
    processed = 0
    total = sum(map(len, splits.values()))
    for role, stems in splits.items():
        rows = []
        totals = {'background': 0, 'track_area': 0, 'ignore': 0, 'images_without_track_area': 0}
        digest_sets[role] = set()
        for stem in stems:
            source_mask = mask_read(masks[stem])
            with Image.open(images[stem]) as image:
                if image.size != source_mask.shape[::-1]:
                    raise ValueError('Dimensions incompatibles : ' + stem)
            remapped = track_mask(source_mask, mapping)
            destination = out / 'masks' / f'{stem}.png'
            Image.fromarray(remapped).save(destination)
            for label, value in [('background', 0), ('track_area', 1), ('ignore', 255)]:
                totals[label] += int((remapped == value).sum())
            totals['images_without_track_area'] += int(not (remapped == 1).any())
            digest = sha256(images[stem])
            digest_sets[role].add(digest)
            inventory.append({'stem': stem, 'role': role, 'image_sha256': digest,
                              'source_mask_sha256': sha256(masks[stem]),
                              'prepared_mask_sha256': sha256(destination)})
            rows.append({'image': str(images[stem]), 'mask': str(destination),
                         'sequence': 'RAILSEM_PAPER_IMAGE_' + stem})
            processed += 1
            if processed == 1 or processed % 200 == 0 or processed == total:
                print(f'Préparation RailSem19 : {processed}/{total}', flush=True)
        write_manifest(out / f'{role}.csv', rows)
        counts[role] = {'images': len(rows), **totals}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if digest_sets[a] & digest_sets[b]:
            raise ValueError('Images identiques entre ensembles officiels : ' + a + '/' + b)
    report = {'task': 'railsem_track', 'classes': ['background', 'track_area'],
              'positive_labels': ['rail-track', 'tram-track'],
              'positive_ids': sorted(mapping[1]), 'ignore_ids': sorted(mapping[2]),
              'mapping': {name: 255 if value in mapping[2] else int(value in mapping[1])
                          for name, value in mapping[0].items()},
              'counts': counts, 'unused_images': len(images)-total,
              'config_sha256': sha256(config_path), 'source': str(raw),
              'split_files_sha256': {role: sha256(splits_folder / f'{role}.txt') for role in splits},
              'manifest_sha256': {role: sha256(out / f'{role}.csv') for role in splits},
              'sequence_is_image_identifier': True, 'sequences_known': False,
              'split_note': PAPER_NOTE, 'inventory': inventory}
    dump(out / 'preparation.json', report)
    make_preview(out, images)
    return out


def make_preview(out, images):
    inventory = json.loads((out / 'preparation.json').read_text())['inventory']
    # Reproducible examples from across the training list; human-readable target check.
    stems = [row['stem'] for row in inventory if row['role'] == 'train']
    selected = [stems[int(i)] for i in np.linspace(0, len(stems)-1, min(8, len(stems)))]
    sheet = Image.new('RGB', (960, ((len(selected)+1)//2)*292), '#f3f5f7')
    for index, stem in enumerate(selected):
        image = np.asarray(Image.open(images[stem]).convert('RGB')).copy()
        mask = mask_read(out / 'masks' / f'{stem}.png')
        for label, color in [(1, [0, 220, 255]), (255, [255, 70, 180])]:
            pixels = mask == label
            image[pixels] = (.6*image[pixels] + .4*np.array(color)).astype(np.uint8)
        view = Image.fromarray(image).resize((480, 270))
        x, y = (index % 2)*480, (index // 2)*292
        sheet.paste(view, (x, y))
        ImageDraw.Draw(sheet).text((x+8, y+272), stem+' | cyan=zone de voie, rose=ignore', fill='black')
    sheet.save(out / 'APERCU_DES_MASQUES.jpg')


def write_pretraining_state(root, values):
    path = Path(root) / 'configs/railsem_pretraining.json'
    temp = path.with_suffix('.json.tmp')
    dump(temp, values)
    os.replace(temp, path)


def tsv_initialization(root, mode='auto'):
    """Accept only a real completed RailSem19 run, or explicit ImageNet baseline."""
    root = Path(root).resolve()
    if mode == 'imagenet':
        return ['--pretrained'], {'source': 'imagenet'}
    if mode not in {'auto', 'railsem', 'imagenet', 'egopath'}:
        raise ValueError('Mode d’initialisation inconnu')
    ego = mode == 'egopath'
    path = root / ('configs/railsem_egopath_pretraining.json' if ego else 'configs/railsem_pretraining.json')
    if not path.exists():
        if ego:
            raise ValueError('Préapprentissage ego-path terminé et enregistré requis')
        if mode == 'railsem':
            raise ValueError('Préapprentissage RailSem19 absent : lancer le bouton 14')
        return ['--pretrained'], {'source': 'imagenet', 'reason': 'RailSem19 non configuré'}
    state = json.loads(path.read_text())
    accepted = {'completed'} if ego else {'completed', 'available_after_interruption'}
    if state.get('status') not in accepted:
        if ego:
            raise ValueError('Préapprentissage ego-path pas encore terminé ; attendre la fin de l’expérience')
        raise ValueError('RailSem19 pas encore terminé. Attendre le bouton 14 ou choisir explicitement --initialization imagenet.')
    checkpoint = (root / state['checkpoint']).resolve()
    if not checkpoint.is_relative_to(root / 'runs') or not checkpoint.is_file():
        raise ValueError('Checkpoint RailSem19 local introuvable')
    if sha256(checkpoint) != state['checkpoint_sha256']:
        raise ValueError('Checkpoint RailSem19 modifié depuis son enregistrement')
    from .model import load_checkpoint
    data = load_checkpoint(checkpoint)
    if (data.get('task') != ('railsem_egopath' if ego else 'railsem_track') or data.get('synthetic_demo', True)
            or data.get('training_only_demo', True) or data.get('steps', 0) < 1
            or data.get('architecture') != 'DeepLabV3Plus' or data.get('encoder') != 'resnet50'):
        raise ValueError('Checkpoint incompatible ou simple démonstration')
    return ['--init', str(checkpoint)], {'source': 'railsem19_egopath' if ego else 'railsem19', 'checkpoint': str(checkpoint),
                                        'checkpoint_sha256': state['checkpoint_sha256'],
                                        'classifier': 'reinitialized', 'split_note': ('Official paper image split; source-video independence is not established. Public ego-path pretraining is not TSV evaluation.' if ego else PAPER_NOTE),
                                        'target_note': ('Public ego-path boundaries are not certified TSV inner rail faces' if ego else 'Generic track area'),
                                        'pretraining_status': state['status'],
                                        'completed_epochs': state.get('completed_epochs', data.get('epoch'))}
