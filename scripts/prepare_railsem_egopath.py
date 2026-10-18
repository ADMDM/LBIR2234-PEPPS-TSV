"""Prepare public ego-path annotations against already imported RailSem19 images."""
import argparse,csv,json,sys,hashlib
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from railtsv.egopath import egopath_mask
from railtsv.common import write_manifest,sha256


def prepare(annotation,images,splits,out):
    annotation,images,splits,out=map(Path,[annotation,images,splits,out])
    data=json.loads(annotation.read_text());out.mkdir(parents=True,exist_ok=False);(out/'masks').mkdir()
    groups={}
    for split in ['train','val','test']:
        for stem in (splits/(split+'.txt')).read_text().split():
            if stem in groups:raise ValueError('Identifiant partagé entre partitions')
            groups[stem]=split
    rows={k:[] for k in ['train','val','test','unassigned']};issues=[];missing=[]
    for number,(name,ann) in enumerate(sorted(data.items()),1):
        image=images/name
        if not image.is_file():missing.append(name);continue
        try:
            with Image.open(image) as im:mask=egopath_mask(ann,im.size)
            target=out/'masks'/(image.stem+'.png');Image.fromarray(mask).save(target)
            group=groups.get(image.stem,'unassigned')
            rows[group].append(dict(image=str(image.resolve()),mask=str(target.resolve()),sequence='railsem_image_'+image.stem))
        except (ValueError,KeyError,TypeError) as e:issues.append(dict(image=name,reason=str(e)))
        if number%1000==0:print(number,'annotations traitées',flush=True)
    for name,records in rows.items():write_manifest(out/(name+'.csv'),records)
    result=dict(source=str(annotation.resolve()),source_sha256=sha256(annotation),source_annotations=len(data),
        prepared={k:len(v) for k,v in rows.items()},missing_images=missing,invalid_annotations=issues,
        task='railsem_egopath',classes={'background':0,'ego_path':1,'ignore':255},
        split='Paper RailSem19 split intersected with available ego-path annotations; remaining images unassigned',
        video_sequence_independence_established=False,tsv_inner_edges_equivalent=False,
        masks='Between individually continuous source polylines; unsupported rows ignored; no manual TSV reviewed flag',
        training_performed=False,license='CC BY-NC-SA 4.0',
        source_url='https://github.com/irtrailenium/train-ego-path-detection')
    (out/'preparation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['invalid_annotations']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['annotation','images','splits','out']:p.add_argument('--'+key,required=True)
    a=p.parse_args();prepare(a.annotation,a.images,a.splits,a.out)
