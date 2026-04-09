"""
Extract one frame from a video and save it as an image.

Recommended order (see README.md): run after trim_video.py and crop_video.py so the still
matches the pixels you will process. Use that image in Ultralytics ParkingPtsSelection (GUI) to
export parking polygons, or edit bounding_boxes.json to match the same resolution.

Run:

  python extract_frame.py data/clip_cropped.mp4 -f 100
  python extract_frame.py data/clip_cropped.mp4 -f 0 -o data/reference_frame.jpg

Frame index -f is zero-based (0 = first frame).

Full help:

  python extract_frame.py -h
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Save a single frame from a video file.")
    p.add_argument("video", type=Path, help="Path to input video.")
    p.add_argument(
        "-f",
        "--frame",
        type=int,
        default=0,
        help="Zero-based frame index to extract (default: 0).",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: <video_stem>_frame<N>.jpg next to the video).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    path = args.video.resolve()
    if not path.is_file():
        print(f"Video not found: {path}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Could not open video: {path}", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(args.frame))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print(f"Could not read frame index {args.frame} (past end of video?).", file=sys.stderr)
        return 1

    out = args.out
    if out is None:
        out = path.parent / f"{path.stem}_frame{args.frame}.jpg"
    else:
        out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(out), frame):
        print(f"Failed to write: {out}", file=sys.stderr)
        return 1

    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
