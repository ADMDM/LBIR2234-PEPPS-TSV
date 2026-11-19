"""One command for GPS extraction/import, exact 2D joins, video and verified ZIP.

The configuration is data, not shell code. Paths are relative to the config.
Existing verified predictions can be reused; no silent training or model change.
"""
from pathlib import Path
import argparse
import json
import re
import shutil
import subprocess
import sys
from handoff_bundle import prepare_delivery, package_delivery, sha256, write_json, run_identity


def resolve(config_path, value):
    if value is None:
        return None
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(config_path).parent / p).resolve()


def run(config_path, check_only=False):
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    if cfg.get('schema_version') != 1:
        raise ValueError('Expected pipeline schema_version=1')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', cfg.get('video_id', '')) or cfg['video_id'] in ['.', '..']:
        raise ValueError('video_id must be a simple identifier, not a path')
    paths = {k: resolve(config_path, cfg.get(k)) for k in
             ['localisation_result', 'gabarit_result', 'comparison_video', 'output_dir', 'source_video']}
    if not paths['output_dir']:
        raise ValueError('output_dir is required')
    if paths['output_dir'].exists():
        raise FileExistsError('Choose a new output_dir; existing work is preserved')
    for key in ['localisation_result', 'gabarit_result', 'comparison_video', 'source_video']:
        if paths[key] is not None and not paths[key].exists():
            raise FileNotFoundError(f'{key}: {paths[key]}')
    if cfg.get('render_connected') and not paths['gabarit_result']:
        raise ValueError('render_connected requires a verified gabarit_result')
    if cfg.get('render_connected') and paths['comparison_video']:
        raise ValueError('Choose connected rendering or an existing comparison video')
    gps = cfg.get('gps')
    if bool(paths['localisation_result']) == bool(gps):
        raise ValueError('Choose an existing localisation_result OR a new gps configuration')
    if gps:
        if not paths['source_video'] or gps.get('mode') not in ['ocr', 'csv']:
            raise ValueError('New GPS extraction requires source_video and mode ocr/csv')
        for key in ['config', 'network', 'rails_csv'] + (['gps_csv'] if gps['mode'] == 'csv' else []):
            p = resolve(config_path, gps.get(key))
            if p is None or not p.is_file():
                raise FileNotFoundError(f'gps.{key}: {p}')
    if check_only:
        return {'status': 'configuration_paths_checked', 'note': 'No files created; detailed joins run with make run.'}
    out = paths['output_dir']
    out.mkdir(parents=True, exist_ok=False)
    try:
        loc = paths['localisation_result']
        if gps:
            loc = out / 'CALCUL_GPS'
            command = [sys.executable, str(Path(__file__).with_name('run_localisation.py')),
                       '--video', str(paths['source_video']), '--video-id', cfg['video_id'],
                       '--config', str(resolve(config_path, gps['config'])),
                       '--network', str(resolve(config_path, gps['network'])),
                       '--rails-csv', str(resolve(config_path, gps['rails_csv'])), '--out', str(loc),
                       '--design-gauge-m', str(gps.get('design_gauge_m', 1.676))]
            command += ['--ocr'] if gps['mode'] == 'ocr' else ['--gps-csv', str(resolve(config_path, gps['gps_csv']))]
            subprocess.run(command, check=True)
        identity = run_identity(loc)
        if identity['video_id'] != cfg['video_id']:
            raise ValueError('Configuration video identity differs from GPS results')
        if paths['source_video'] and sha256(paths['source_video']) != identity['source_video_sha256']:
            raise ValueError('Source video differs from GPS results')
        delivery = out / cfg['video_id']
        print('Vérification des temps et des contours, préparation des fichiers…', flush=True)
        checks = prepare_delivery(loc, delivery, paths['gabarit_result'])
        if cfg.get('render_connected'):
            from render_connected_video import render
            checks['video'] = render(loc, paths['gabarit_result'], delivery / 'VIDEO_RAILS_GABARIT_GPS.mp4')
        elif paths['comparison_video']:
            # A supplied visualization must identify its original source and match all PTS.
            if cfg.get('comparison_source_sha256') != identity['source_video_sha256']:
                raise ValueError('Comparison video source identity was not documented')
            import av
            from handoff_bundle import read_csv
            rows = read_csv(delivery / 'RAILS_GAUCHE_DROIT_GPS.csv')
            count = 0
            with av.open(str(paths['comparison_video'])) as video:
                for i, frame in enumerate(video.decode(video=0)):
                    if i >= len(rows) or abs(float(frame.pts*frame.time_base)-float(rows[i]['video_time_s'])) > 2e-6:
                        raise ValueError('Comparison video timestamp mismatch')
                    if (frame.width, frame.height) != (int(rows[i]['image_width_px']),int(rows[i]['image_height_px'])):
                        raise ValueError('Comparison video dimensions differ')
                    count += 1
            if count != len(rows):
                raise ValueError('Comparison video frame count differs')
            shutil.copy2(paths['comparison_video'], delivery / 'VIDEO_COMPARAISON.mp4')
            checks['video'] = {'status': 'passed', 'frames_checked': count,
                               'comparison_sha256': sha256(paths['comparison_video']),
                               'meaning': 'Existing rail visualization; no object result supplied'}
        title = 'Vidéo 3 min : rails, objets et GPS' if paths['gabarit_result'] else 'Rails et GPS ; aucun gabarit 2D associé'
        notice = f"""{title}

COMMENCER : RAILS_GAUCHE_DROIT_GPS.csv, une ligne par image, pixels d'origine.
POINTS_RAILS_METRES.csv détaille les paires de points et les conversions sous hypothèse.
Les mètres transversaux ne sont pas calibrés. Une cellule vide signifie inconnu.
Le temps est celui de la vidéo source, pas frame_index / fps.
Le GPS nouveau est préfixé gps_ ; les colonnes historiques sont conservées.
L'avancement cartographique est estimé, interpolé et séparé de la vitesse réelle.
Sans réseau cartographique couvrant la vidéo : parcours et écartement nominal vides.

Pour les objets, lorsqu'ils sont fournis : OBJETS_GPS.csv contient les observations
retenues ; A_VERIFIER_GPS.csv les hypothèses à relire. Le GPS est celui de la caméra,
pas de l'objet. POINTS_OBJETS.csv et TXT/ décrivent les mêmes silhouettes.
Joindre avec video_id + frame_index ; vérifier aussi le temps. Ne pas sommer des
observations successives comme des objets distincts. Aucun objet ne signifie pas
voie libre. La présence dans le couloir est un recouvrement 2D, pas un danger confirmé.

Dans la vidéo connectée, l'image originale est réduite et décalée pour laisser la
place à la carte. CORRESPONDANCE_VIDEO.csv donne la transformation d'affichage.
Les CSV/TXT restent dans l'image originale, sans cette transformation.
CONTROLES_JOINTURE.json certifie la cohérence des données, pas leur précision terrain.
"""
        (delivery / '00_LIRE_AVANT_UTILISATION.txt').write_text(notice, encoding='utf-8')
        write_json(delivery / 'CONTROLES_JOINTURE.json', dict(checks, status='passed'))
        print('Création du ZIP et vérification de tous ses fichiers…', flush=True)
        packaged = package_delivery(delivery, out / f"{cfg['video_id']}.zip")
        result = {'status': 'complete', 'video_id': cfg['video_id'], 'configuration_sha256': sha256(config_path),
                  'checks': checks, 'package': packaged, 'reused_localisation': bool(paths['localisation_result'])}
        write_json(out / 'TERMINE.json', result)
        return result
    except Exception as exc:
        write_json(out / 'ECHEC.json', {'status': 'failed', 'error': str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='Check paths without creating files')
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.check), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
