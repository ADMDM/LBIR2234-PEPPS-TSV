import pytest
from railtsv.validation_protocol import aggregate_axis_errors,final_split_guard,branch_metrics,profile_alerts

def test_sections_cannot_outvote_images_or_create_independent_experiments():
    r=[dict(image_id='a',capture_id='one',error_signed_px=1)]*100
    r += [dict(image_id='b',capture_id='one',error_signed_px=9)]
    x=aggregate_axis_errors(r)
    assert x['sections']==101 and x['images']==2 and x['captures']==1
    assert x['image_macro_mae_px']==5 and x['confidence_interval'] is None

def test_renamed_video_and_same_journey_cannot_become_final_test():
    train=dict(capture_id='used',role='development',capture_group='trip_a',source_sha256='hash_a')
    new=dict(capture_id='new',role='final_reserved',capture_group='trip_b',source_sha256='hash_b',identity_verified=True,predictions_seen_before_freeze=False)
    assert final_split_guard([train,new],['new'])['eligible']
    for update in [dict(source_sha256='hash_a'),dict(capture_group='trip_a'),dict(identity_verified=False),dict(predictions_seen_before_freeze=True)]:
        with pytest.raises(ValueError):final_split_guard([train,{**new,**update}],['new'])
    with pytest.raises(ValueError):final_split_guard([train],[])

def test_branch_abstention_does_not_artificially_raise_success():
    refs=[dict(event_id='switch',frame_index=i,branch_id='left',reviewed=True,route_known=True) for i in range(3)]
    preds=[dict(event_id='switch',frame_index=0,branch_id='left',status='accepted'),dict(event_id='switch',frame_index=1,branch_id='right',status='accepted')]
    m=branch_metrics(refs,preds)
    assert m['correct_among_accepted']==.5 and m['success_all_known']==pytest.approx(1/3)
    assert m['coverage']==pytest.approx(2/3) and m['abstained']==1 and m['events_with_wrong_branch']==1
    assert branch_metrics([{**r,'route_known':False} for r in refs],preds)['success_all_known'] is None

def test_fragmented_axis_is_not_declared_a_switch_or_filled():
    s=[[dict(y=20,center_x=50)],[dict(y=95,center_x=80)]]
    assert set(profile_alerts(s,100,100))=={'axis_fragmented','large_lateral_gap_between_segments'}
    assert profile_alerts([],100,100)==['axis_absent']


def test_stability_keeps_playback_columns_and_excludes_real_clock_gap(tmp_path):
    import csv,json,importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/benchmark_stability.py'
    spec=importlib.util.spec_from_file_location('stability_clock_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'source';source.mkdir()
    (source/'video.json').write_text(json.dumps(dict(dimensions=[1000,100],fps=5,checkpoint_sha256='test',clock=dict(filter_clock='physical_utc'))))
    data=[]
    for i,t in enumerate([0,2,4,20,22]):
        center=500+i
        points=[dict(y=y,left_inner_x=center-50,right_inner_x=center+50,center_x=center,apparent_width_px=100) for y in [16,24,32]]
        data.append(dict(frame_index=i,seconds=i/5,physical_seconds=t,source_seconds=i/5,repeated_source_frame='False',axis_points_json=json.dumps([points]),left_inner_x=center-50,right_inner_x=center+50,center_x_px=center,apparent_width_px=100,status='candidate',row_y=24))
    with (source/'mesures.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    out=tmp_path/'output';module.run(source,out,None)
    summary=json.loads((out/'RESULTATS_STABILITE.json').read_text())
    assert summary['matched_adjacent_pairs']==3 and summary['matched_adjacent_triples']==1
    result=list(csv.DictReader((out/'comparaison_par_image.csv').open()))
    assert float(result[1]['seconds'])==.2 and float(result[1]['filter_seconds'])==2
    assert all(r['assumed_gauge_m']=='' for r in result)
