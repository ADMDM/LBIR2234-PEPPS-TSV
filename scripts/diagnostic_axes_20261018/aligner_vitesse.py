"""Join explicit speed telemetry to video-frame times without extrapolation.

Input telemetry CSV: time_s,speed (or named columns). time_s is measured in
seconds in the telemetry clock. video_time = telemetry_time + explicit offset.
No clock offset, speed unit or physical speed is inferred from image width.
"""
import argparse,bisect,csv,hashlib,json,math
from pathlib import Path

def read_samples(path,time_column,speed_column,unit,offset):
    samples=[]
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            t=float(r[time_column])+offset;v=float(r[speed_column])/(3.6 if unit=='km/h' else 1)
            if not math.isfinite(t) or not math.isfinite(v) or v<0:raise ValueError('Temps et vitesse finis, vitesse non négative requis')
            if samples and t<=samples[-1][0]:raise ValueError('Horodatages télémétriques strictement croissants requis')
            samples.append((t,v))
    if not samples:raise ValueError('Relevé vide')
    return samples

class SpeedTimeline:
    def __init__(self,samples,max_gap):
        if not math.isfinite(max_gap) or max_gap<=0:raise ValueError('max_gap doit être positif')
        self.samples=samples;self.times=[x[0] for x in samples];self.max_gap=max_gap
        if not samples or any(not math.isfinite(t) or not math.isfinite(v) or v<0 for t,v in samples):raise ValueError('Relevé invalide')
        if any(b<=a for a,b in zip(self.times,self.times[1:])):raise ValueError('Temps strictement croissants requis')
    def at(self,t):
        i=bisect.bisect_left(self.times,t)
        if i<len(self.times) and math.isclose(self.times[i],t,abs_tol=1e-9,rel_tol=0):return self.samples[i][1],'telemetry_sample'
        if i==0 or i==len(self.times):return None,'outside_telemetry'
        a,b=self.samples[i-1],self.samples[i]
        if b[0]-a[0]>self.max_gap:return None,'telemetry_gap'
        u=(t-a[0])/(b[0]-a[0]);return a[1]+u*(b[1]-a[1]),'interpolated'
    def integral(self,a,b):
        if b<=a:raise ValueError('Intervalle positif requis')
        boundaries=[a]+[t for t in self.times[bisect.bisect_right(self.times,a):bisect.bisect_left(self.times,b)]]+[b]
        total=0.
        for x,y in zip(boundaries,boundaries[1:]):
            vx,_=self.at(x);vy,_=self.at(y);vm,_=self.at((x+y)/2)
            if vx is None or vy is None or vm is None:return None
            total+=(vx+vy)/2*(y-x)
        return total

def align(frames,timeline,source,offset,time_column='time_s'):
    output=[];previous=None;distance=0.;continuous_from_start=True;segment=0;segment_distance=0.
    for r in frames:
        t=float(r[time_column]);speed,status=timeline.at(t)
        if previous is not None and t<=previous:raise ValueError('Frames dans l’ordre temporel strict requis')
        delta=timeline.integral(previous,t) if previous is not None else None
        if previous is None:
            continuous_from_start=abs(t)<1e-9 and speed is not None
        elif delta is None:
            continuous_from_start=False;segment+=1;segment_distance=0.
        else:distance+=delta;segment_distance+=delta
        output.append(dict(r,train_speed_m_s=speed,train_speed_km_h=speed*3.6 if speed is not None else None,
            speed_status=status,speed_source=source,telemetry_offset_s=offset,
            speed_uncertainty_m_s=None,speed_uncertainty_status='not_provided',
            distance_from_video_start_m=distance if continuous_from_start else None,
            interval_distance_m=delta,continuous_segment_id=segment,
            segment_distance_m=segment_distance if speed is not None else None,
            pose_status='speed_only_no_camera_rotation_or_position',
            note='GPS speed and derived travel do not calibrate image depth or rail gauge'))
        previous=t
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',type=Path,required=True,help='INDEX_FRAMES.csv')
    p.add_argument('--frames-time-column',required=True,help='Physical elapsed-seconds column; do not use playback seconds for time-lapse')
    p.add_argument('--telemetry',type=Path,required=True)
    p.add_argument('--time-column',default='time_s');p.add_argument('--speed-column',default='speed')
    p.add_argument('--speed-unit',choices=['m/s','km/h'],required=True)
    p.add_argument('--video-minus-telemetry-seconds',type=float,required=True,help='Explicit known time offset, including 0 when clocks are synchronized')
    p.add_argument('--max-gap-seconds',type=float,required=True,help='Explicit maximum interpolation interval between samples')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();offset=args.video_minus_telemetry_seconds
    if not math.isfinite(offset):raise ValueError('Décalage fini requis')
    samples=read_samples(args.telemetry,args.time_column,args.speed_column,args.speed_unit,offset)
    with args.frames.open(newline='') as f:frames=list(csv.DictReader(f))
    if not frames:raise ValueError('Index des frames vide')
    if len({r['sequence_id'] for r in frames})!=1:raise ValueError('Une seule séquence par fichier')
    rows=align(frames,SpeedTimeline(samples,args.max_gap_seconds),args.telemetry.name,offset,args.frames_time_column)
    with args.out.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=dict(frames=len(rows),frames_with_speed=sum(r['train_speed_m_s'] is not None for r in rows),
        input_sha256=hashlib.sha256(args.telemetry.read_bytes()).hexdigest(),
        frames_sha256=hashlib.sha256(args.frames.read_bytes()).hexdigest(),offset_s=offset,
        interpolation='linear, no extrapolation, gaps excluded',max_gap_s=args.max_gap_seconds,
        independent_speed_error=None,output=str(args.out))
    args.out.with_suffix('.provenance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
