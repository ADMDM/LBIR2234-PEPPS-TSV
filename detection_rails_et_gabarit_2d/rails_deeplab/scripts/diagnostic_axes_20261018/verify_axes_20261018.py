"""Independent arithmetic and interchange checks; no inference rerun."""
from pathlib import Path
import csv,gzip,json,math,sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'railtsv').is_dir())
OUT=ROOT/'runs/fusion_axes_20261018'
sys.path.insert(0,str(Path(__file__).parent))
from axes_export_20261018 import vector_points,spatial_profile
from railtsv.adaptive_smoothing import AdaptiveProfileSmoother

def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

def main():
    # A curved polyline and a disconnected second segment; no cross-gap tangent.
    p={'segments':[[dict(y=y,left_inner_x=100+y*y/1000,right_inner_x=200+y*y/1000,
        center_x=150+y*y/1000,apparent_width_px=100) for y in [8,16,24,32,40,48,56]],
        [dict(y=y,left_inner_x=600,right_inner_x=700,center_x=650,apparent_width_px=100) for y in [88,96,104]]]}
    vp=vector_points(p);assert set(vp)=={8,16,24,32,40,48,56,88,96,104}
    assert vp[88]['tangent_dx']==0 and vp[88]['tangent_dy']==-1
    assert all(q['tangent_dy']<0 for q in vp.values())
    assert all(math.isclose(math.hypot(q['tangent_dx'],q['tangent_dy']),1) for q in vp.values())
    sp=spatial_profile(p)
    assert max(abs(a['center_x']-b['center_x']) for a,b in zip(p['segments'][0],sp['segments'][0]))<1e-8
    f=AdaptiveProfileSmoother(1920);f.update(p,0);f.update({'segments':[]},.033)
    reset=f.update(p,.066);assert all(q['smoothing_status']=='new_row' for s in reset['segments'] for q in s)
    duplicate_rejected=False
    try:f.update(p,.066)
    except ValueError:duplicate_rejected=True
    assert duplicate_rejected
    wide=read(OUT/'fusion_validation/AXES_MEMES_IMAGES.csv');assert len(wide)==12*135
    assert len({(r['image_id'],r['y_px']) for r in wide})==len(wide)
    long=read(OUT/'fusion_validation/VECTEURS_ET_ERREURS.csv');count=0
    for r in long:
        if r['x_px']:
            assert math.isclose(float(r['x_px']),(float(r['left_x_px'])+float(r['right_x_px']))/2,abs_tol=1e-8)
            if r['tangent_dx']:
                assert float(r['tangent_dy'])<0
                assert math.isclose(math.hypot(float(r['tangent_dx']),float(r['tangent_dy'])),1,abs_tol=1e-8)
        if r['error_signed_px']:
            assert math.isclose(float(r['error_signed_px']),float(r['x_px'])-float(r['reference_x_px']),abs_tol=1e-8);count+=1
    # Recalculate matched-section MAE without importing the analysis helper.
    sums={k:[] for k in ['ancien_tsv','ego_tsv','moyenne_50_50']}
    for r in wide:
        if not all(r[k+'_error_px']!='' for k in ['ancien_tsv','ego_tsv']):continue
        a,b=float(r['ancien_tsv_error_px']),float(r['ego_tsv_error_px'])
        sums['ancien_tsv'].append(abs(a));sums['ego_tsv'].append(abs(b));sums['moyenne_50_50'].append(abs((a+b)/2))
    for r in read(OUT/'fusion_validation/FUSION_SECTIONS_COMMUNES.csv'):
        x=sums[r['method']];assert len(x)==int(r['n']) and math.isclose(sum(x)/len(x),float(r['mae_px']),rel_tol=1e-10)
    results=[]
    for folder in sorted((OUT/'equipe_gabarit').iterdir()):
        if not folder.is_dir():continue
        meta=json.loads((folder/'CONTRAT.json').read_text());width,height=meta['original_dimensions'];n=0;last=None;repeated_frames=set();observed_frames=set()
        with gzip.open(folder/'AXES_SYNCHRONISES.csv.gz','rt',newline='') as f:
            for r in csv.DictReader(f):
                key=(int(r['frame_index']),int(r['y_px']))
                assert last is None or key>last;last=key;n+=1
                assert 0<=key[1]<height and key[1]%8==0
                assert int(r['width_px'])==width and int(r['height_px'])==height
                assert math.isclose(float(r['time_s']),key[0]/meta['source_fps'],abs_tol=1e-8)
                assert r['ancien_error_px']==r['ego_error_px']==r['x_m']==r['y_m']==''
                for k in ['ancien','ego','ancien_lisse','ego_lisse']:
                    if r[k+'_x_px']!='':
                        x,l,rr=map(float,[r[k+'_x_px'],r[k+'_left_px'],r[k+'_right_px']])
                        assert 0<=l<rr<width and math.isclose(x,(l+rr)/2,abs_tol=1e-8)
                        if r[k+'_tangent_dx']:
                            assert math.isclose(math.hypot(float(r[k+'_tangent_dx']),float(r[k+'_tangent_dy'])),1,abs_tol=1e-8)
                            assert float(r[k+'_tangent_dy'])<0
                    if r['repeated_source_frame']=='True':assert r[k+'_x_px']==''
                if r['repeated_source_frame']=='True':repeated_frames.add(key[0])
                else:observed_frames.add(key[0])
                if r['disagreement_px']!='':
                    a,b=float(r['ancien_x_px']),float(r['ego_x_px'])
                    assert math.isclose(float(r['disagreement_px']),abs(a-b),abs_tol=1e-8)
                    assert math.isclose(float(r['fusion_mean_raw_x_px']),(a+b)/2,abs_tol=1e-8)
        assert n==meta['frames']*len(range(0,height,8))==meta['point_rows']
        assert len(repeated_frames)==meta['repeated_frames']
        assert len(observed_frames|repeated_frames)==meta['frames']
        vectors=0
        with gzip.open(folder/'VECTEURS_PAR_FRAME.csv.gz','rt',newline='') as f:
            for r in csv.DictReader(f):
                a=json.loads(r['axis_raw_json']);b=json.loads(r['axis_filtered_json']);assert sum(map(len,a))==int(r['observed_points'])
                assert {p['y'] for s in a for p in s}=={p['y'] for s in b for p in s}
                if r['repeated_source_frame']=='True':assert a==b==[]
                vectors+=1
        assert vectors==2*meta['frames']
        results.append(dict(sequence=meta['sequence_id'],rows=n,vector_rows=vectors,repeated_frames=len(repeated_frames),verified=True))
        print('Verified',meta['sequence_id'],n,flush=True)
    report=dict(passed=True,checks=['fixed-row keys unique','frame clocks','raw/filter aligned observations','midpoint identity',
        'unit forward tangent','gap preservation','empty missing errors/metres','exclude repeats','independent MAE'],
        validation_error_records=count,sequences=results,physical_accuracy_validated=False)
    (OUT/'VERIFICATION.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
