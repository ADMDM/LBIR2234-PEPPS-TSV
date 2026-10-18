"""Fit measured planar references, never an assumed rail width."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from railtsv.metric_geometry import fit_plane_calibration
from railtsv.common import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--points',required=True);p.add_argument('--camera-description',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    camera=json.loads(Path(a.camera_description).read_text())
    required=['camera_id','source_capture','image_width_px','image_height_px','camera_matrix','distortion_coefficients','measured_point_source']
    if any(camera.get(k) is None for k in required) or camera.get('fixed_pose_confirmed') is not True:
        raise ValueError('Renseigner caméra, dimensions, correction de distorsion, origine des mesures et pose fixe')
    rows=list(csv.DictReader(open(a.points,newline='')))
    if any(r['role'] not in ['fit','check'] for r in rows):raise ValueError('role doit être fit ou check')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Identifiants de points distincts requis')
    def coordinates(role,fields):return [[float(r[k]) for k in fields] for r in rows if r['role']==role]
    result=fit_plane_calibration(coordinates('fit',['u_undistorted_px','v_undistorted_px']),coordinates('fit',['X_m','Z_m']),
        coordinates('check',['u_undistorted_px','v_undistorted_px']),coordinates('check',['X_m','Z_m']))
    result.update(camera=camera,points_sha256=sha256(a.points),applicable_only_to='Matching undistorted pixels, camera pose and measured local track plane')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2)
    print('Calage préparé, erreur de contrôle en m :',result['control_rmse_m'],'; validation terrain requise')


if __name__=='__main__':main()
