"""Generic frame/rail handoff. Preserve source geometry and telemetry, add explicit GPS fields."""
from bisect import bisect_right
from pathlib import Path
import argparse,csv,hashlib,json,math
from rail_localisation import transverse_proxy


def build_handoff(rails_csv, route, out, design_gauge_m=1.676, max_gap_s=5):
    rails_csv,out=Path(rails_csv),Path(out)
    if not .5 <= design_gauge_m <= 2 or max_gap_s<=0: raise ValueError('Invalid design gauge or interpolation gap')
    observations=route.get('matches',[])
    actual={int(m['frame_index']):m for m in observations}
    if len(actual)!=len(observations):raise ValueError('Repeated observation frame')
    times=[float(m['video_time_s']) for m in observations]
    if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('GPS times not increasing')
    mapped=route.get('route_status')=='mapped_route_estimate_with_observation_checks'
    gauge=route.get('gauge_selected_nominal_m') if mapped else None
    if gauge is not None and not .5 <= float(gauge) <= 2:raise ValueError('Invalid mapped nominal gauge')
    ids={m.get('video_id') for m in observations if m.get('video_id')}
    hashes={m.get('source_video_sha256') for m in observations if m.get('source_video_sha256')}
    if route.get('video_id'):ids.add(route['video_id'])
    if route.get('source_video_sha256'):hashes.add(route['source_video_sha256'])
    if len(ids)!=1 or len(hashes)!=1:raise ValueError('A unique video identity and SHA-256 are required')
    expected_id=next(iter(ids));expected_hash=next(iter(hashes))
    new_columns=['gps_latitude_deg','gps_longitude_deg','gps_observation_source','gps_observation_status','gps_ocr_confidence',
                 'gps_speed_provided_m_s','gps_speed_status','gps_map_status','distance_gps_to_map_m','matched_edge_id',
                 'mapped_line','mapped_gauge_nominal_m','gauge_design_default_m','gauge_selection_status',
                 'route_cumulative_estimate_m','route_distance_status','route_interval_start_frame','route_interval_end_frame',
                 'route_progress_m_per_video_second','route_progress_status','map_dataset_doi','physical_uncertainty_m']
    point_columns=['video_id','frame_index','video_time_s','segment_index','point_index','y_px','left_x_px','right_x_px',
                   'width_px','image_width_px','mapped_gauge_nominal_m','gauge_design_default_m','metric_status',
                   'scale_m_per_px_nominal','left_offset_from_image_midline_m_nominal','right_offset_from_image_midline_m_nominal',
                   'scale_m_per_px_design','left_offset_from_image_midline_m_design','right_offset_from_image_midline_m_design',
                   'actual_error_m','physical_uncertainty_m']
    counts={'frames':0,'point_pairs':0,'gps_observations_on_exported_frames':0,'distance_interpolated_frames':0,
            'distance_withheld_frames':0,'nominal_metric_pairs':0,'design_metric_pairs':0,'withheld_narrow_pairs':0}
    out.mkdir(parents=True,exist_ok=False)
    frames_out=out/'RAILS_GAUCHE_DROIT_GPS.csv';points_out=out/'POINTS_RAILS_METRES.csv'
    with rails_csv.open(newline='',encoding='utf-8-sig') as inp,frames_out.open('w',newline='') as fout,points_out.open('w',newline='') as pout:
        reader=csv.DictReader(inp)
        required={'video_id','frame_index','video_time_s','source_video_sha256','image_width_px','left_segments_json','right_segments_json'}
        if not required.issubset(reader.fieldnames or []):raise ValueError('Rail CSV missing required columns')
        if set(new_columns)&set(reader.fieldnames):raise ValueError('Already enriched CSV: supply the original rail CSV')
        writer=csv.DictWriter(fout,fieldnames=reader.fieldnames+new_columns);writer.writeheader()
        pw=csv.DictWriter(pout,fieldnames=point_columns);pw.writeheader()
        previous_time=-math.inf;seen=set()
        for row in reader:
            if row['video_id']!=expected_id or row['source_video_sha256']!=expected_hash:raise ValueError('Rail/GPS video identity mismatch')
            index=int(row['frame_index']);t=float(row['video_time_s']);width=int(row['image_width_px'])
            if index in seen or t<=previous_time:raise ValueError('Invalid rail timeline')
            seen.add(index);previous_time=t
            observation=actual.get(index)
            if observation and abs(float(observation['video_time_s'])-t)>1e-6:raise ValueError('GPS/frame timestamp mismatch')
            additions=dict.fromkeys(new_columns,'')
            additions.update(gauge_design_default_m=design_gauge_m,mapped_gauge_nominal_m=gauge if gauge else '',
                             gauge_selection_status=route.get('gauge_selection_status','unresolved'),
                             route_distance_status='not_available_no_resolved_route',gps_observation_status='not_observed_on_this_frame',
                             map_dataset_doi=route.get('network',{}).get('dataset_doi',''),
                             mapped_line=' / '.join(sorted({e['line'] for e in route.get('route_edges',[]) if e.get('line')})))
            if observation:
                additions.update(gps_latitude_deg=observation['latitude_deg'],gps_longitude_deg=observation['longitude_deg'],
                                 gps_observation_source=observation.get('gps_source','provided_observation'),
                                 gps_observation_status=observation.get('gps_status','source_status_not_provided'),
                                 gps_ocr_confidence=observation.get('ocr_confidence',''),
                                 gps_speed_provided_m_s=observation.get('gps_speed_provided_m_s',''),
                                 gps_speed_status=observation.get('gps_speed_status','not_provided'),
                                 gps_map_status=observation.get('match_status','unresolved'),
                                 distance_gps_to_map_m=observation.get('distance_gps_to_map_m',''),
                                 matched_edge_id=observation.get('matched_edge_id',''))
                counts['gps_observations_on_exported_frames']+=1
            if mapped:
                if observation:
                    additions.update(route_cumulative_estimate_m=observation['route_cumulative_estimate_m'],
                                     route_distance_status='mapped_at_gps_observation')
                elif times[0] <= t <= times[-1]:
                    j=min(len(times)-2,max(0,bisect_right(times,t)-1));dt=times[j+1]-times[j]
                    additions.update(route_interval_start_frame=observations[j]['frame_index'],route_interval_end_frame=observations[j+1]['frame_index'])
                    if dt<=max_gap_s:
                        a=observations[j]['route_cumulative_estimate_m'];b=observations[j+1]['route_cumulative_estimate_m']
                        additions.update(route_cumulative_estimate_m=a+(t-times[j])*(b-a)/dt,
                                         route_distance_status='linear_between_observations_not_odometry')
                        counts['distance_interpolated_frames']+=1
                    else:additions['route_distance_status']='withheld_gap_too_long'
                else:additions['route_distance_status']='outside_gps_coverage_no_extrapolation'
                if times[0] <= t <= times[-1]:
                    j=min(len(times)-2,max(0,bisect_right(times,t)-1));dt=times[j+1]-times[j]
                    if dt<=max_gap_s:
                        additions.update(route_interval_start_frame=observations[j]['frame_index'],route_interval_end_frame=observations[j+1]['frame_index'],
                                         route_progress_m_per_video_second=(observations[j+1]['route_cumulative_estimate_m']-observations[j]['route_cumulative_estimate_m'])/dt,
                                         route_progress_status='interval_estimate_per_video_second_not_instantaneous_real_speed')
            if additions['route_cumulative_estimate_m']=='':counts['distance_withheld_frames']+=1
            writer.writerow(dict(row,**additions));counts['frames']+=1
            left,right=json.loads(row['left_segments_json']),json.loads(row['right_segments_json'])
            if len(left)!=len(right):raise ValueError('Unpaired rail segments')
            for si,(ls,rs) in enumerate(zip(left,right)):
                if len(ls)!=len(rs):raise ValueError('Unpaired rail points')
                for pi,((xl,yl),(xr,yr)) in enumerate(zip(ls,rs)):
                    if yl!=yr:raise ValueError('Rail points are not on the same row')
                    nominal=transverse_proxy(xl,xr,width,gauge) if gauge else None
                    design=transverse_proxy(xl,xr,width,design_gauge_m)
                    record=dict(video_id=expected_id,frame_index=index,video_time_s=t,segment_index=si,point_index=pi,y_px=yl,
                                left_x_px=xl,right_x_px=xr,width_px=xr-xl,image_width_px=width,mapped_gauge_nominal_m=gauge or '',
                                gauge_design_default_m=design_gauge_m,actual_error_m='',physical_uncertainty_m='',
                                metric_status='withheld_narrow_corridor' if design is None else 'conditional_design_only_nominal_unknown' if gauge is None else 'conditional_row_local_not_calibrated')
                    for suffix,result in [('nominal',nominal),('design',design)]:
                        for key in ['scale_m_per_px','left_offset_from_image_midline_m','right_offset_from_image_midline_m']:
                            record[key+'_'+suffix]=result[key] if result else ''
                    pw.writerow(record);counts['point_pairs']+=1
                    counts['nominal_metric_pairs']+=int(nominal is not None);counts['design_metric_pairs']+=int(design is not None)
                    counts['withheld_narrow_pairs']+=int(design is None)
    counts.update(source_rails_sha256=hashlib.sha256(rails_csv.read_bytes()).hexdigest(),source_video_sha256=expected_hash,
                  source_geometry_and_telemetry_preserved=True,minimum_width_fraction=.02,max_interpolation_gap_s=max_gap_s,
                  nominal_gauge_m=gauge,design_gauge_m=design_gauge_m,
                  metric_error_measured=False,route_distance_estimate_m=route.get('route_distance_estimate_m'))
    (out/'CONTROLES_EXPORT.json').write_text(json.dumps(counts,indent=2))
    notice=Path(__file__).resolve().parent/'NOTICE_GABARIT.txt'
    if notice.exists():
        (out/'LIRE_MOI.txt').write_text(notice.read_text()+f'\nVidéo : {expected_id}\nImages exportées : {counts["frames"]}\nÉcartement nominal : {gauge if gauge else "inconnu"} m\nDéfaut de dimensionnement : {design_gauge_m} m\n')
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rails-csv',type=Path,required=True);p.add_argument('--route-json',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--design-gauge-m',type=float,default=1.676)
    p.add_argument('--max-interpolation-gap-s',type=float,default=5)
    a=p.parse_args()
    print(json.dumps(build_handoff(a.rails_csv,json.loads(a.route_json.read_text()),a.out,a.design_gauge_m,a.max_interpolation_gap_s),indent=2))


if __name__=='__main__':main()
