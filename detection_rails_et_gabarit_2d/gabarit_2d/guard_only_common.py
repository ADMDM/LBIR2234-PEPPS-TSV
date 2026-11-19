"""Byte-preserving CSV reader and frozen rejection evidence; no ML imports."""
import csv,hashlib,json,math
from pathlib import Path

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))

class RawCSV:
    """Read parsed records and retain their exact original text, including CRLF.

    csv.reader drives the collecting iterator, so quoted multi-line values also
    remain one record. This is deliberately not implemented with split(',').
    """
    def __init__(self,handle):
        self.handle=handle;self.parts=[];self.reader=csv.reader(self);self.count=0
        self.parts=[];self.fields=next(self.reader);self.header=''.join(self.parts)
        require(len(self.fields)==len(set(self.fields)),'Duplicate CSV fields')
        self.pending=self._next()
    def __iter__(self):return self
    def __next__(self):
        value=next(self.handle);self.parts.append(value);return value
    def _next(self):
        self.parts=[]
        values=next(self.reader,None)
        if values is None:return None
        require(len(values)==len(self.fields),'CSV record field count differs')
        return dict(zip(self.fields,values)),''.join(self.parts)
    def pop(self):
        require(self.pending is not None,'CSV ended early')
        result=self.pending;self.pending=self._next();self.count+=1;return result
    def frame(self,index):
        values=[]
        while self.pending is not None and int(self.pending[0]['frame_index'])==index:values.append(self.pop())
        require(self.pending is None or int(self.pending[0]['frame_index'])>index,'Out-of-order CSV frame')
        return values

def rejection_evidence(paths,checkpoint):
    """Require an explicitly rejected initial recipe plus rejected selection(s).

    The frozen criteria are recalculated; rejection is never converted into an
    admissible small-object gain. Extra selection trials use the same five gates.
    """
    require(len(paths)>=2,'At least initial and selective rejection reports required')
    frozen={};reports=[];seen=set();kinds=set()
    for path in paths:
        path=Path(path).resolve(strict=True);require(str(path) not in seen,'Repeated rejection report');seen.add(str(path))
        mpath=path.parent/'EVALUATION_MANIFEST.json';m=read(mpath);e=read(path)
        require(e['status']==m['status']=='complete' and e['split']==m['split']=='val','Complete VAL rejection required')
        require(e['images']==m['image_count']==288 and m['test_images_or_annotations_opened'] is False and m['inputs_unchanged'] is True,'Invalid rejection scope')
        require(e['checkpoint_sha256']==checkpoint,'Rejection checkpoint differs from baseline')
        require(hashlib.sha256(json.dumps(m['configuration'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==m['configuration_sha256']==e['configuration_sha256'],'Rejection configuration changed')
        if path.name=='METRICS.json':
            kinds.add('initial');link=m['metrics'];base,new=e['baseline'],e['tiled']
            eligible=new['AP_small']>base['AP_small'] and new['AP50_95']>=base['AP50_95']-.01
            require(e['eligible_by_frozen_rule'] is eligible and not eligible,'Initial recipe is not rejected')
        else:
            require(path.name=='SELECTION.json','Unexpected rejection report name');kinds.add('selection');link=m['selection']
            require(e['selected_eligible'] is False and all(e[k] is None for k in ('selected_config','selected','selected_operating','selected_grid_index')),'Selection has an eligible winner')
            require(len(e['candidates'])==12,'Twelve selective configurations required')
            b,bo=e['baseline'],e['baseline_operating']
            for candidate in e['candidates']:
                a,o=candidate['metrics'],candidate['operating']
                checks=dict(precision=o['precision']>=bo['precision']-.03,small_recall=o['small_recall']>bo['small_recall'],F2=o['F2']>=bo['F2'],AP_small=a['AP_small']>b['AP_small'],AP50_95=a['AP50_95']>=b['AP50_95']-.01)
                require(candidate['eligibility_checks']==checks and candidate['eligible'] is all(checks.values()) and not all(checks.values()),'A selection configuration is not rejected')
        require(Path(link['path']).resolve()==path and link['sha256']==sha(path) and link['bytes']==path.stat().st_size,'Report not bound to completed manifest')
        for record in m['inputs']+m['output_files']:
            target=Path(record['path']).resolve(strict=True)
            require(sha(target)==record['sha256'] and target.stat().st_size==record['bytes'],'Changed rejection evidence: '+str(target));frozen[str(target)]=record['sha256']
        frozen[str(path)]=sha(path);frozen[str(mpath)]=sha(mpath)
        reports.append(dict(path=str(path),sha256=sha(path),manifest_sha256=sha(mpath),outcome='rejected',images=288))
    require(kinds=={'initial','selection'},'Initial and selective rejection evidence both required')
    return reports,frozen
