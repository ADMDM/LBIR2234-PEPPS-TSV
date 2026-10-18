"""Read-only scientific/algebra audit; writes a new audit record and camera strata."""
import sys, csv, json, hashlib, math, collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
R=Path(sys.argv[1]); O=R/'03_RESULTATS/TESTS_ET_METRES_20261018'; V=R/'03_RESULTATS/GABARIT_V3_METRES_ESTIMES_20261018'; B=R/'03_RESULTATS/GABARIT_2_VIDEOS_V2_20261018'
def rows(p):
    with p.open(newline='') as f:yield from csv.DictReader(f)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
protocol=json.loads((R/'runs/test_externe_metres_20261018/PROTOCOLE_FIGE.json').read_text())
assert sha(R/protocol['checkpoint'])==protocol['checkpoint_sha256']
train_manifests=[R/'data/prepared/railsem_track_paper_v1/train.csv',R/'data/prepared/railsem_egopath_20261014/train.csv',R/'runs/experience_egopath_20261016_141426/dataset/train.csv']
paths=sorted({(p.parent/r['image']).resolve() for p in train_manifests for r in rows(p)})
with ThreadPoolExecutor(max_workers=4) as ex:train_hashes=dict(zip(paths,ex.map(sha,paths)))
newpaths=sorted((R/'data/external/railbench_locked_20261018/images/val').glob('*.png'))
with ThreadPoolExecutor(max_workers=4) as ex:new_hashes=dict(zip(newpaths,ex.map(sha,newpaths)))
matches=[str(p) for p,h in new_hashes.items() if h in set(train_hashes.values())]
assert not matches
meta={r['ID']:r for r in rows(R/'runs/test_externe_metres_20261018/IMAGES_TEST_FIGEES.csv')}; measures=list(rows(O/'01_TEST_EXTERNE_FIGE/RESULTATS_PAR_IMAGE.csv'));strata=[]
for label,cameras in [('Frontales (center + middle)',{'center','rgb_12mp_middle'}),('Laterales (left + right)',{'rgb_12mp_left','rgb_12mp_right'})]:
    a=[r for r in measures if meta[r['image_id']]['original_camera'] in cameras];den=sum(int(r['reference_rows']) for r in a);num=sum(int(r['returned_rows']) for r in a);err=[float(r['mae_pct_width']) for r in a if r['mae_pct_width']]
    strata.append(dict(camera_group=label,images=len(a),images_without_axis=sum(int(r['predicted_points'])==0 for r in a),images_with_reference=sum(int(r['reference_rows'])>0 for r in a),images_with_error=len(err),reference_rows=den,returned_rows=num,coverage_pct=100*num/den if den else None,mean_image_mae_pct_width=sum(err)/len(err) if err else None,interpretation='Post-test descriptive stratification by publisher camera ID; all frozen results preserved; not a new selection or model tuning'))
(O/'01_TEST_EXTERNE_FIGE/RESULTATS_PAR_ORIENTATION.json').write_text(json.dumps(strata,ensure_ascii=False,indent=2))
counts=[]
for folder in ['01_TSV_RECORD','02_TSV_03676_180S']:
    old=list(rows(B/folder/'FRAMES_VITESSE_VECTEUR.csv'));new=list(rows(V/folder/'FRAMES_VITESSE_VECTEUR.csv'));assert len(old)==len(new)
    for i,(a,b) in enumerate(zip(old,new)):
        assert int(b['frame_index'])==i
        assert all(a[k]==b[k] for k in a),'Original frame fields modified'
        if b['travel_interval_distance_m_estimate']:
            dt=float(b['travel_interval_duration_s']);assert 0<dt<=5
        if b['travel_interval_status'].startswith('telemetry') or b['travel_interval_status'].startswith('time_gap'):
            assert not b['travel_interval_distance_m_estimate']
        assert not b['ground_depth_z_m'] and not b['travel_distance_uncertainty_m']
    n=converted=refs=0
    aiter=rows(B/folder/'AXE_MEDIAN.csv');biter=rows(V/folder/'AXE_MEDIAN.csv')
    import itertools
    for a,b in itertools.zip_longest(aiter,biter):
        assert a is not None and b is not None
        assert all(a[k]==b[k] for k in a),'Original axis fields modified'
        n+=1
        for key in ['ground_x_m','ground_z_m','physical_uncertainty_m','measured_gauge_m']:assert not b[key]
        if b['scale_m_per_px_assumed']:
            g=float(b['assumed_gauge_m']);l=float(b['left_x_px']);rr=float(b['right_x_px']);x=float(b['x_px']);w=float(b['image_width_px']);s=float(b['scale_m_per_px_assumed']);converted+=1
            assert math.isclose(s*(rr-l),g,abs_tol=1e-10)
            assert math.isclose(float(b['axis_offset_from_image_midline_m_assumed']),(x-(w-1)/2)*s,abs_tol=1e-10)
            assert math.isclose(float(b['axis_offset_m_scenario_1435'])*1.067,float(b['axis_offset_m_scenario_1067'])*1.435,abs_tol=1e-10)
        if b['reference_error_equiv_m_assumed']:
            refs+=1
            assert b['error_signed_px'] and b['reference_width_px']
            assert math.isclose(float(b['reference_error_equiv_m_assumed']),float(b['error_signed_px'])*float(b['assumed_gauge_m'])/float(b['reference_width_px']),abs_tol=1e-10)
    counts.append(dict(folder=folder,frame_rows=len(new),axis_rows=n,metric_proxy_rows=converted,known_conditional_residual_rows=refs))
motion=json.loads((V/'AVANCEMENT_PAR_VIDEO.json').read_text());assert math.isclose(motion[1]['cumulative_distance_of_valid_parts_m'],1397.282481239693,abs_tol=1e-5)
result=dict(checkpoint_unchanged=True,training_images_checked=len(paths),external_images_checked=len(newpaths),identical_file_duplicates_with_training=matches,duplicate_method='SHA256 of full file bytes across the three actual training manifests; does not prove absence of re-encoding, transformed duplicates, generic ImageNet overlap or hidden publisher overlap.',training_manifests=[dict(path=str(p.relative_to(R)),sha256=sha(p)) for p in train_manifests],exports=counts,original_v2_coordinates_and_fields_unchanged=True,missing_errors_and_uncertainties_not_filled=True,metre_and_motion_algebra_passed=True,scientific_scope='Software/data checks, not physical accuracy certification',camera_stratification=strata)
(O/'AUDIT_EXPORTS_ET_SOURCES.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
