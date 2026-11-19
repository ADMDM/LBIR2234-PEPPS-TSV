"""One entry point: video OCR or synchronized CSV, then offline rail matching."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from gps_inputs import normalize_csv
from gps_ocr import read_samples
from video_timeline import inspect_video
from rail_localisation import RailNetwork, analyze_trace, write_csv, DATASET_MD5


def load_timeline_csv(path, video_id):
    """Use a previously verified frame manifest (including a rail-frame CSV)."""
    frames = []
    with Path(path).open(newline="",encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["video_id"] != video_id:
                raise ValueError("Timeline video identity mismatch")
            frames.append({k:row[k] for k in ["video_id","frame_index","source_frame_index","video_time_s","source_pts_s",
                                              "image_width_px","image_height_px","source_video_sha256"] if k in row})
    if not frames or len({r["source_video_sha256"] for r in frames}) != 1:
        raise ValueError("Empty or mixed timeline source hashes")
    if any(float(b["video_time_s"]) <= float(a["video_time_s"]) for a,b in zip(frames,frames[1:])):
        raise ValueError("Nonmonotonic video timeline")
    if len({int(r["frame_index"]) for r in frames}) != len(frames):
        raise ValueError("Duplicate frame identifiers")
    return frames, {"video_id":video_id,"source_video_sha256":frames[0]["source_video_sha256"],
                    "timeline_rows":len(frames),"timeline_source":"provided_frame_manifest_not_new_video_decode",
                    "first_pts_s":float(frames[0]["video_time_s"]),"last_pts_s":float(frames[-1]["video_time_s"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video",type=Path)
    source.add_argument("--frames-csv",type=Path)
    gps = parser.add_mutually_exclusive_group(required=True)
    gps.add_argument("--ocr",action="store_true")
    gps.add_argument("--gps-csv",type=Path)
    parser.add_argument("--video-id",required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--network",type=Path,required=True)
    parser.add_argument("--rails-csv",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--design-gauge-m",type=float,default=1.676)
    parser.add_argument("--max-interpolation-gap-s",type=float,default=5.0)
    args = parser.parse_args()
    if args.ocr and not args.video:
        parser.error("OCR requires a video")
    for path in [args.video,args.frames_csv,args.gps_csv,args.config,args.network,args.rails_csv]:
        if path is not None and not path.is_file():
            parser.error(f"Input file not found: {path}")
    if hashlib.md5(args.network.read_bytes()).hexdigest() != DATASET_MD5:
        raise ValueError("Expected the pinned AfTS-Db v3 network")
    config = json.loads(args.config.read_text())
    args.out.mkdir(exist_ok=False,parents=True)
    started = time.perf_counter()
    if args.video:
        frames,samples,metadata = inspect_video(args.video,args.video_id,args.out/"PREUVES_OCR" if args.ocr else None,
                                               config.get("interval_s",1),config.get("roi",(.78125,.8333333333,1,1)),config.get("scale",3))
    else:
        frames,metadata = load_timeline_csv(args.frames_csv,args.video_id)
    write_csv(args.out/"FRAMES_VIDEO.csv",frames)
    decode_seconds = time.perf_counter()-started
    if args.ocr:
        write_csv(args.out/"IMAGES_LUES.csv",samples)
        observations,audit,source_stats = read_samples(samples,args.out/"PREUVES_OCR",config,metadata,args.out)
    else:
        observations,audit = normalize_csv(args.gps_csv,frames,config,args.video_id,metadata["source_video_sha256"])
        source_stats = {"source":"provided_csv","input_sha256":hashlib.sha256(args.gps_csv.read_bytes()).hexdigest(),
                        "rows":len(audit),"accepted":len(observations),"rejected":len(audit)-len(observations)}
    write_csv(args.out/"GPS_OBSERVATIONS.csv",observations)
    write_csv(args.out/"AUDIT_LECTURES.csv",audit)
    source_stats["status_counts"] = dict(Counter(r["status"] for r in audit))
    (args.out/"SOURCE_ET_CONFIG.json").write_text(json.dumps({"video":metadata,"config":config,"input":source_stats},indent=2,ensure_ascii=False))
    if len(observations) >= 2:
        network = RailNetwork.from_geopackage(args.network,observations)
        route,geometry = analyze_trace(network,observations)
    else:
        route = {"route_status":"insufficient_accepted_gps","matches":observations,"gauge_selected_nominal_m":None}
        geometry = {"type":"FeatureCollection","features":[]}
    route.update(video_id=args.video_id,source_video_sha256=metadata["source_video_sha256"],
                 input_provenance=source_stats,frame_timeline_source=metadata)
    (args.out/"RESULTATS_LOCALISATION.json").write_text(json.dumps(route,indent=2,ensure_ascii=False))
    (args.out/"TRAJET_CARTOGRAPHIQUE.geojson").write_text(json.dumps(geometry,ensure_ascii=False))
    write_csv(args.out/"POSITIONS_RACCORDEES.csv",route.get("matches",[]))
    write_csv(args.out/"DISTANCES_ENTRE_OBSERVATIONS.csv",route.get("intervals",[]))
    if args.rails_csv:
        from build_handoff import build_handoff
        build_handoff(args.rails_csv,route,args.out/"GABARIT",args.design_gauge_m,args.max_interpolation_gap_s)
    timing = {"decode_or_timeline_seconds":decode_seconds,"whole_run_seconds":time.perf_counter()-started,
              "meaning":"Offline runtime of GPS processing only; not rail-model end-to-end real-time performance"}
    (args.out/"TEMPS_DE_CALCUL.json").write_text(json.dumps(timing,indent=2))
    (args.out/"RUN_COMPLETED.json").write_text(json.dumps({"status":"complete","video_id":args.video_id,
                                                        "source_video_sha256":metadata["source_video_sha256"]},indent=2))
    print(json.dumps({"accepted_gps":len(observations),"rejected":source_stats["rejected"],"route_status":route["route_status"],
                      "route_distance_estimate_m":route.get("route_distance_estimate_m"),
                      "gauge_selected_nominal_m":route.get("gauge_selected_nominal_m"),"timing":timing},indent=2),flush=True)


if __name__ == "__main__":
    main()
