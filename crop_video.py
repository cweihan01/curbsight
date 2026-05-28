"""
Crop every frame of a video (e.g. remove a strip from the bottom) and write a new file.

Recommended order (see README.md): after trim_video.py, use this to drop fixed UI/chrome or
unwanted edges, then extract_frame.py on the cropped video for annotating parking slots.

Run:

  python crop_video.py data/clip.mp4 --remove-bottom 200 -o data/clip_cropped.mp4
  python crop_video.py data/clip.mp4 --remove-top 40 --remove-left 0 --remove-right 0 -o out.mp4

At least one --remove-* value must be > 0.

Full help:

  python crop_video.py -h
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crop each frame vertically and save a new video. "
        "Use --remove-bottom (and optionally --remove-top) to drop strips from the frame edges."
    )
    p.add_argument("video", type=Path, help="Input video path.")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output video (default: <input_stem>_cropped.mp4 next to input).",
    )
    p.add_argument(
        "--remove-bottom",
        type=int,
        default=0,
        metavar="PX",
        help="Pixels to cut off the bottom of each frame (default: 0).",
    )
    p.add_argument(
        "--remove-top",
        type=int,
        default=0,
        metavar="PX",
        help="Pixels to cut off the top of each frame (default: 0).",
    )
    p.add_argument(
        "--remove-left",
        type=int,
        default=0,
        metavar="PX",
        help="Pixels to cut off the left (default: 0).",
    )
    p.add_argument(
        "--remove-right",
        type=int,
        default=0,
        metavar="PX",
        help="Pixels to cut off the right (default: 0).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src = args.video.resolve()
    if not src.is_file():
        print(f"Video not found: {src}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"Could not open video: {src}", file=sys.stderr)
        return 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"[crop] Input: {src} | size={w}x{h} | fps={fps:.3f} | "
        f"frames={frame_count if frame_count > 0 else 'unknown'}"
    )

    t, b, l, r = args.remove_top, args.remove_bottom, args.remove_left, args.remove_right
    if t + b + l + r <= 0:
        print(
            "Specify at least one of --remove-bottom, --remove-top, --remove-left, --remove-right (pixels > 0).",
            file=sys.stderr,
        )
        cap.release()
        return 1

    y0, y1 = t, h - b
    x0, x1 = l, w - r
    if y1 <= y0 or x1 <= x0:
        print(
            f"Invalid crop: frame {w}x{h}, remove top={t} bottom={b} left={l} right={r} "
            f"→ empty or negative region.",
            file=sys.stderr,
        )
        cap.release()
        return 1
    print(
        f"[crop] Crop settings: top={t}, bottom={b}, left={l}, right={r} "
        f"-> output size={x1 - x0}x{y1 - y0}"
    )

    out = args.out
    if out is None:
        out = src.parent / f"{src.stem}_cropped.mp4"
    else:
        out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    cw, ch = x1 - x0, y1 - y0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (cw, ch))
    if not writer.isOpened():
        print(f"Could not open writer: {out}", file=sys.stderr)
        cap.release()
        return 1

    processed = 0
    progress_every = max(1, int(round(fps * 5)))  # print roughly every 5 seconds
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cropped = frame[y0:y1, x0:x1]
            writer.write(cropped)
            processed += 1
            if processed % progress_every == 0:
                if frame_count > 0:
                    pct = (processed / frame_count) * 100
                    print(
                        f"[crop] Progress: {processed}/{frame_count} frame(s) "
                        f"({processed / fps:.1f}s, {pct:.1f}%)"
                    )
                else:
                    print(f"[crop] Progress: {processed} frame(s) ({processed / fps:.1f}s)")
    finally:
        cap.release()
        writer.release()

    print(f"[crop] Done: wrote {processed} frame(s), {cw}x{ch} @ {fps:.3f} fps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
