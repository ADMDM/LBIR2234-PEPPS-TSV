"""Synthetic fixtures, never a railway validation dataset."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image
from railtsv.common import new_dir, write_manifest, dump


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = new_dir(args.out)
    rng = np.random.default_rng(42)
    binary, tsv = [], []
    for group in range(3):
        for index in range(2):
            i = group*2+index
            image = rng.integers(60, 140, (96, 160, 3), dtype=np.uint8)
            mask = np.zeros((96, 160), dtype=np.uint8)
            cv2.line(mask, (20+i*2, 95), (70, 28), 1, 5)
            cv2.line(mask, (140-i*2, 95), (88, 28), 2, 5)
            image[mask > 0] = [200, 210, 220]
            mask[:22] = 255
            image_path, binary_path, tsv_path = [out / f"{prefix}_{i:02d}.png" for prefix in ("image", "binary", "tsv")]
            Image.fromarray(image).save(image_path)
            Image.fromarray(mask).save(tsv_path)
            Image.fromarray(np.where(mask == 2, 1, mask).astype(np.uint8)).save(binary_path)
            binary.append(dict(image=str(image_path), mask=str(binary_path), sequence=f"synthetic_{group}"))
            tsv.append(dict(image=str(image_path), mask=str(tsv_path), sequence=f"synthetic_{group}"))
    write_manifest(out / "binary.csv", binary)
    write_manifest(out / "tsv.csv", tsv)
    dump(out / "README.json", {"synthetic_only": True, "purpose": "Verify plumbing, no railway accuracy claim"})
    writer = cv2.VideoWriter(str(out / "test_video.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 96))
    if not writer.isOpened():
        raise RuntimeError("Codec MJPG indisponible")
    for i in range(20):
        frame = rng.integers(0, 255, (96, 160, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    print(out)


if __name__ == "__main__":
    main()
