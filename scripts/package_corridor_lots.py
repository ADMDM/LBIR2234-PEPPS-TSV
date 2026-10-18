"""Create portable, disjoint annotation ZIPs, dereferencing canonical images."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from railtsv.common import dump,new_dir,sha256
from railtsv.annotation_config import active_batch


def package(batch,output):
    batch=Path(batch).resolve()
    preparation=json.loads((batch/'preparation.json').read_text())
    if preparation.get('task')!='corridor':raise ValueError('Lots couloir requis')
    out=new_dir(output);seen=set();reports=[]
    for letter in 'ABC':
        lot=batch/f'lot_{letter}';manifest=json.loads((lot/'manifest.json').read_text());rows=manifest['images']
        expected={row['stem']+'.png':row for row in rows}
        if len(rows)!=50 or len(expected)!=50 or seen & set(expected):raise ValueError('Lots attribués de 50 images distinctes requis')
        seen|=set(expected)
        if {p.name for p in (lot/'images').glob('*.png')}!=set(expected):raise ValueError('Images du lot incompatibles')
        for name,row in expected.items():
            if sha256(lot/'images'/name)!=row['image_sha256']:raise ValueError('Image canonique modifiée')
        files=[*(lot/'images').glob('*.png'),*(lot/'json').glob('*.json'),
            *(lot/name for name in ['labels.txt','labelme.yaml','manifest.json','sequences.csv','sources.csv',
                'start_annotation.py','ANNOTER.command','ANNOTER_WINDOWS.bat','LIRE_AVANT_ANNOTATION.md'])]
        archive=out/f'LOT_{letter}_50_IMAGES_COULOIR.zip'
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            z.writestr(lot.name+'/json/','')
            for p in sorted(files):z.write(p,lot.name+'/'+p.relative_to(lot).as_posix())
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:raise ValueError('Archive corrompue')
            if len([n for n in z.namelist() if '/images/' in n and n.endswith('.png')])!=50:
                raise ValueError('Archive incomplète')
        reports.append(dict(lot=letter,file=archive.name,sha256=sha256(archive),images=50,
            proposals=len(list((lot/'json').glob('*.json'))),size_bytes=archive.stat().st_size))
        print(f'Lot {letter} : 50 images, archive vérifiée.',flush=True)
    dump(out/'archives.json',dict(batch_id=batch.name,task='corridor',lots=reports,images=150,
        note='Images réelles incluses ; propositions à revoir. Aucun partage externe effectué.'))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,default=active_batch(ROOT));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(package(a.batch,a.out))
