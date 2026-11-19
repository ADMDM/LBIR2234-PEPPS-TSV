"""Join a verified GPS/rail run to an optional 2D run without changing predictions.

The common identity is video SHA-256 + video_id + decoded frame index + native
timestamp. Geometry is compared before objects can be attached. A successful
join is a consistency check, never an accuracy certificate.
"""
from pathlib import Path
from itertools import zip_longest
import csv
import hashlib
import json
import math
import shutil
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def run_identity(localisation):
    """Read full provenance; older completion markers contained only a status."""
    localisation = Path(localisation)
    completion = json.loads((localisation / 'RUN_COMPLETED.json').read_text())
    route = json.loads((localisation / 'RESULTATS_LOCALISATION.json').read_text())
    if completion.get('status') not in ['complete', 'complete_and_independently_verified']:
        raise ValueError('GPS run is incomplete')
    if completion['status'] == 'complete_and_independently_verified':
        evidence = json.loads((localisation / 'VERIFICATION_INDEPENDANTE.json').read_text())
        if evidence.get('status') != 'passed':
            raise ValueError('GPS independent verification did not pass')
        for name, digest in evidence.get('sha256', {}).items():
            if Path(name).name != name or sha256(localisation / 'GABARIT' / name) != digest:
                raise ValueError('GPS independently verified file changed')
    identity = {k: route[k] for k in ['video_id', 'source_video_sha256']}
    if len(identity['source_video_sha256']) != 64:
        raise ValueError('Missing full source SHA-256')
    for key, value in identity.items():
        if completion.get(key, value) != value:
            raise ValueError('GPS completion/provenance mismatch')
    return identity


def timeline(rows):
    if not rows:
        raise ValueError('Empty rail timeline')
    if len({r['video_id'] for r in rows}) != 1 or len({r['source_video_sha256'] for r in rows}) != 1:
        raise ValueError('Mixed video identity')
    result = {}
    previous = -math.inf
    for row in rows:
        i, t = int(row['frame_index']), float(row['video_time_s'])
        if i in result or not math.isfinite(t) or t <= previous:
            raise ValueError('Duplicate frame or invalid time')
        result[i] = row
        previous = t
    return result


def check_frame(row, index, time_key='time_video_s'):
    original = index.get(int(row['frame_index']))
    if original is None or original['video_id'] != row['video_id']:
        raise ValueError('Object/rail video or frame mismatch')
    time_value = float(row[time_key])
    if not math.isfinite(time_value) or abs(time_value - float(original['video_time_s'])) > 2e-6:
        raise ValueError('Object/rail native timestamp mismatch')
    for key, counterpart in [('image_width', 'image_width_px'), ('image_height', 'image_height_px')]:
        if key in row and int(row[key]) != int(original[counterpart]):
            raise ValueError('Object/rail image dimensions mismatch')
    return original


def validate_geometry(gabarit_rails, gps_points):
    """Compare every displayed point, including segment breaks and order."""
    count, largest = 0, 0.0
    with Path(gabarit_rails).open(newline='') as fa, Path(gps_points).open(newline='') as fb:
        valid = (r for r in csv.DictReader(fa) if r['display_valid'].lower() == 'true')
        for a, b in zip_longest(valid, csv.DictReader(fb)):
            if a is None or b is None:
                raise ValueError('Rail point count mismatch')
            for key in ['video_id', 'frame_index', 'segment_index']:
                if a[key] != b[key]:
                    raise ValueError('Rail segment identity mismatch')
            for ka, kb in [('time_video_s', 'video_time_s'), ('y_px', 'y_px'),
                           ('left_display_px', 'left_x_px'), ('right_display_px', 'right_x_px')]:
                difference = abs(float(a[ka]) - float(b[kb]))
                if not math.isfinite(difference) or difference > 1e-6:
                    raise ValueError(f'Rail coordinate mismatch: {ka}')
                largest = max(largest, difference)
            count += 1
    if not count:
        raise ValueError('No comparable rail geometry')
    return {'point_pairs_checked': count, 'largest_numeric_difference': largest}


GPS_COLUMNS = ['gps_latitude_deg', 'gps_longitude_deg', 'gps_observation_source',
               'gps_observation_status', 'gps_speed_provided_m_s', 'gps_speed_status',
               'route_cumulative_estimate_m', 'route_distance_status',
               'mapped_line', 'mapped_gauge_nominal_m', 'gauge_design_default_m',
               'physical_uncertainty_m']


def join_observations(source, destination, index):
    """Append camera-position context. Never label it as an object's GPS location."""
    count = 0
    with Path(source).open(newline='') as fi, Path(destination).open('w', newline='') as fo:
        reader = csv.DictReader(fi)
        if set(GPS_COLUMNS) & set(reader.fieldnames):
            raise ValueError('Object CSV already contains GPS enrichment')
        writer = csv.DictWriter(fo, fieldnames=reader.fieldnames + GPS_COLUMNS + ['gps_context_meaning'])
        writer.writeheader()
        for row in reader:
            rail = check_frame(row, index)
            writer.writerow(dict(row, **{k: rail.get(k, '') for k in GPS_COLUMNS},
                                 gps_context_meaning='camera_position_not_object_geolocation'))
            count += 1
    return count


def prepare_delivery(localisation, output, gabarit=None):
    """Create the minimal collaborator folder after checking all input joins."""
    localisation, output = Path(localisation), Path(output)
    completion = run_identity(localisation)
    source = localisation / 'GABARIT'
    rows = read_csv(source / 'RAILS_GAUCHE_DROIT_GPS.csv')
    index = timeline(rows)
    if completion['source_video_sha256'] != rows[0]['source_video_sha256'] or completion['video_id'] != rows[0]['video_id']:
        raise ValueError('GPS completion identity mismatch')
    checks = {'video_id': rows[0]['video_id'], 'source_video_sha256': rows[0]['source_video_sha256'],
              'frames': len(rows), 'scope': 'Data joins and preservation, not prediction accuracy',
              'gabarit_2d_attached': bool(gabarit), 'inputs_sha256': {}}
    if gabarit:
        gabarit = Path(gabarit)
        manifest = json.loads((gabarit / 'MANIFESTE.json').read_text())
        previous = json.loads((gabarit / 'VERIFICATION.json').read_text())
        if previous['status'] != 'passed' or manifest['status'] != 'complete':
            raise ValueError('2D run has not passed its verifier')
        if manifest['source_sha256'] != checks['source_video_sha256'] or manifest['video_id'] != checks['video_id']:
            raise ValueError('2D source video identity mismatch')
        two_d_frames = read_csv(gabarit / 'FRAMES.csv')
        if len(two_d_frames) != len(rows) or len({r['frame_index'] for r in two_d_frames}) != len(rows):
            raise ValueError('2D frame coverage mismatch')
        for row in two_d_frames:
            check_frame(row, index)
        checks.update(validate_geometry(gabarit / 'RAILS.csv', source / 'POINTS_RAILS_METRES.csv'))
    output.mkdir(parents=True, exist_ok=False)
    for name in ['RAILS_GAUCHE_DROIT_GPS.csv', 'POINTS_RAILS_METRES.csv', 'LIRE_MOI.txt']:
        shutil.copy2(source / name, output / name)
        checks['inputs_sha256'][name] = sha256(source / name)
    for name in ['TRAJET_CARTOGRAPHIQUE.geojson', 'GPS_OBSERVATIONS.csv']:
        shutil.copy2(localisation / name, output / name)
        checks['inputs_sha256'][name] = sha256(localisation / name)
    if gabarit:
        checks['object_observations'] = join_observations(gabarit / 'OBJETS.csv', output / 'OBJETS_GPS.csv', index)
        checks['quarantined_observations'] = join_observations(gabarit / 'A_VERIFIER.csv', output / 'A_VERIFIER_GPS.csv', index)
        for name in ['POINTS_OBJETS.csv', 'FRAMES.csv']:
            shutil.copy2(gabarit / name, output / name)
            checks['inputs_sha256'][name] = sha256(gabarit / name)
        # TXT documents preserve their entire original contents and exact filenames.
        # GPS is joined by video_id/frame_index via the frame CSV; no duplicate data.
        (output / 'TXT').mkdir()
        for frame in two_d_frames:
            relative = Path(frame['txt_file'])
            if relative.parent != Path('TXT') or relative.name != f"frame_{int(frame['frame_index']):06d}.txt":
                raise ValueError('Unexpected TXT path')
            src = gabarit / relative
            payload = json.loads(src.read_text())
            check_frame(payload, index)
            shutil.copy2(src, output / relative)
        checks['txt_frames_checked'] = len(two_d_frames)
        checks['inputs_sha256']['2D_manifest'] = sha256(gabarit / 'MANIFESTE.json')
    write_json(output / 'CONTROLES_JOINTURE.json', dict(checks, status='passed'))
    return checks


def package_delivery(folder, destination):
    """Create a portable ZIP and verify every file against SHA-256 after writing."""
    folder, destination = Path(folder), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    files = [p for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'MANIFESTE_FICHIERS.json']
    manifest = {str(p.relative_to(folder)): {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in files}
    write_json(folder / 'MANIFESTE_FICHIERS.json', manifest)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for p in files + [folder / 'MANIFESTE_FICHIERS.json']:
            z.write(p, str(Path(folder.name) / p.relative_to(folder)),
                    compress_type=zipfile.ZIP_STORED if p.suffix == '.mp4' else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(destination) as z:
        for name, record in manifest.items():
            digest = hashlib.sha256()
            with z.open(str(Path(folder.name) / name)) as f:
                for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != record['sha256']:
                raise ValueError('ZIP checksum mismatch: ' + name)
    return {'files_verified': len(manifest), 'zip_sha256': sha256(destination), 'bytes': destination.stat().st_size}
