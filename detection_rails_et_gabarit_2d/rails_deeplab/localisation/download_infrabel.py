"""Download public Infrabel railway geometry, never a private GPS trace.

The hash binds this presentation to the downloaded 2026-11-19 snapshot.
If the provider updates the data, inspect the new version before adopting it.
"""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

URL='https://opendata.infrabel.be/api/explore/v2.1/catalog/datasets/geosporen/exports/geojson'
EXPECTED_SHA256='599831b682243b6786a0f9edab28c64abb3986d95c3e77bb170adc68acc8fc00'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    with urllib.request.urlopen(URL,timeout=60) as response:data=response.read()
    digest=hashlib.sha256(data).hexdigest()
    if digest!=EXPECTED_SHA256:raise ValueError('The public dataset changed. Review its provenance before updating the pinned checksum. New SHA-256: '+digest)
    if json.loads(data).get('type')!='FeatureCollection':raise ValueError('Expected GeoJSON FeatureCollection')
    a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_bytes(data)
    print(json.dumps(dict(path=str(a.out),sha256=digest,license='CC0',attribution='Infrabel',url=URL,private_gps_sent=False),indent=2))
