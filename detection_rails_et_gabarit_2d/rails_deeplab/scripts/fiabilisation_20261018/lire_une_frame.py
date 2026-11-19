"""Read one team's axis vector and available speed/error for one exact frame."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--dossier',type=Path,required=True);p.add_argument('--frame',type=int,required=True);a=p.parse_args()
with (a.dossier/'FRAMES_VITESSE_VECTEUR.csv').open(newline='') as f:
    row=next((r for r in csv.DictReader(f) if int(r['frame_index'])==a.frame),None)
if row is None:raise SystemExit('Frame absente de ce dossier')
result={k:row[k] for k in ['video_id','frame_index','source_frame_index','video_time_s','utc_time','real_elapsed_s','train_speed_km_h','train_speed_m_s','speed_status','axis_status','observation_status','axis_mae_px','error_status']}
result['axis_segments']=json.loads(row['axis_segments_json'])
print(json.dumps(result,ensure_ascii=False,indent=2))
