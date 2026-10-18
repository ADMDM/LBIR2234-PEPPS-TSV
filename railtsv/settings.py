"""Processing resolution is separate from an unestablished metric calibration."""
import json
from pathlib import Path


def processing_resolution(root, fallback=(384, 640)):
    path=Path(root)/'configs/processing_measurement.json'
    if not path.exists():return fallback
    config=json.loads(path.read_text())
    values=config['processing_resolution']
    height,width=values['height_px'],values['width_px']
    if any(type(v) is not int or v<32 or v%32 for v in [height,width]):
        raise ValueError('Résolution réseau : entiers >=32 et multiples de 32')
    if values.get('preserve_aspect_ratio') is not True:
        raise ValueError('Conserver le ratio de l’image est requis')
    if config.get('metric_calibration',{}).get('enabled'):
        raise ValueError('La conversion métrique doit être implémentée et validée avant activation ; ce réglage VGA ne calibre pas la vidéo')
    return height,width
