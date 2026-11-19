"""Extract spaced frames and create a new Labelme annotation session."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from railtsv.common import dump, new_dir, sha256
from railtsv.prepare import extract_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", required=True)
    parser.add_argument("--sequence", action="append", required=True,
                        help="One capture group per --video, in the same order")
    parser.add_argument("--session", required=True)
    parser.add_argument("--frames-root", default="data/frames")
    parser.add_argument("--every", type=float, default=15.)
    parser.add_argument("--max-frames", type=int, default=12)
    args = parser.parse_args()
    if len(args.video) != len(args.sequence) or len(set(args.sequence)) != len(args.sequence):
        parser.error("Provide one distinct --sequence per --video")
    if not math.isfinite(args.every) or args.every <= 0 or args.max_frames < 1:
        parser.error("every and max-frames must be positive")
    session = Path(args.session).resolve()
    frames_root = Path(args.frames_root).resolve()
    videos = [Path(v).resolve() for v in args.video]
    for video, sequence in zip(videos, args.sequence):
        if not sequence or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in sequence):
            parser.error("Sequence names may contain letters, digits, underscore and hyphen")
        if not video.is_file():
            raise FileNotFoundError(video)
        if (frames_root / sequence).exists():
            raise FileExistsError(frames_root / sequence)
        cap = cv2.VideoCapture(str(video))
        try:
            ok, _ = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or not ok or not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"Unreadable video or invalid FPS: {video}")
        finally:
            cap.release()
    new_dir(session)
    (session / "images").mkdir()
    (session / "json").mkdir()
    (session / "labels.txt").write_text("rail_left\nrail_right\nignore\n", encoding="utf-8")
    rows, provenance = [], []
    for video, sequence in zip(videos, args.sequence):
        output = frames_root / sequence
        extract_frames(SimpleNamespace(video=str(video), out=str(output),
                                       every=args.every, max_frames=args.max_frames))
        report = json.loads((output / "extraction.json").read_text())
        report["source"] = os.path.relpath(video, output)
        report["source_sha256"] = sha256(video)
        dump(output / "extraction.json", report)
        for frame in report["frames"]:
            image = output / frame["file"]
            name = f"{sequence}_{image.name}"
            shutil.copyfile(image, session / "images" / name)
            rows.append({"stem": Path(name).stem, "sequence": sequence,
                         "source_video": os.path.relpath(video, session),
                         "frame_index": frame["frame_index"],
                         "estimated_seconds": frame["estimated_seconds"]})
        provenance.append({"sequence": sequence, "video": os.path.relpath(video, session),
                           "sha256": report["source_sha256"], "fps": report["fps"],
                           "frames": len(report["frames"])})
    for filename, fields in [("sequences.csv", ["stem", "sequence"]),
                             ("sources.csv", list(rows[0]))]:
        with (session / filename).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    dump(session / "session.json", {"videos": provenance, "images": len(rows),
         "every_seconds": args.every,
         "grouping": "Grouped by source video. Merge groups if videos share a correlated capture.",
         "annotation_status": "Unannotated; no automatic ground truth"})
    print(f"{len(rows)} images ready for annotation: {session}")


if __name__ == "__main__":
    main()
