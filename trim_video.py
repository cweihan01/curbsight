"""
Copy a time range from a video to a new file (trim / extract segment).

Recommended order with other tools (see README.md): run this first on long footage, then
crop_video.py, then extract_frame.py to grab a still for parking-region annotation.

Run:

  python trim_video.py data/raw.MOV --duration 180 -o data/clip.mp4
  python trim_video.py data/raw.MOV --start 120 --end 300 -o data/clip.mp4
  python trim_video.py data/raw.MOV --middle 120 -o data/clip.mp4

You must pass exactly one of --duration, --end, or --middle. Times are in seconds.

Full help:

  python trim_video.py -h
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Write part of a video to a new file. Times are in seconds from the start of the file."
    )
    p.add_argument("video", type=Path, help="Input video path.")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output video (default: <input_stem>_trim.mp4 next to input).",
    )
    p.add_argument(
        "--start",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Where to begin in seconds (default: 0). Ignored when using --middle.",
    )
    duration = p.add_mutually_exclusive_group(required=True)
    duration.add_argument(
        "--duration",
        type=float,
        metavar="SEC",
        help="Length of the clip in seconds (from --start).",
    )
    duration.add_argument(
        "--end",
        type=float,
        metavar="SEC",
        help="End time in seconds (exclusive-ish: we stop before this time).",
    )
    duration.add_argument(
        "--middle",
        type=float,
        metavar="SEC",
        help="Take this many seconds from the middle of the video (--start is ignored).",
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
    total_sec = frame_count / fps if frame_count > 0 else None
    print(
        f"[trim] Input: {src} | size={w}x{h} | fps={fps:.3f} | "
        f"frames={frame_count if frame_count > 0 else 'unknown'}"
    )

    if args.middle is not None:
        if args.middle <= 0:
            print("--middle must be positive.", file=sys.stderr)
            cap.release()
            return 1
        if total_sec is None or total_sec <= 0:
            print("Could not determine video length for --middle (frame count missing).", file=sys.stderr)
            cap.release()
            return 1
        duration_sec = min(args.middle, total_sec)
        start_sec = max(0.0, (total_sec - duration_sec) / 2.0)
        end_sec = start_sec + duration_sec
    else:
        start_sec = max(0.0, args.start)
        if args.duration is not None:
            if args.duration <= 0:
                print("--duration must be positive.", file=sys.stderr)
                cap.release()
                return 1
            end_sec = start_sec + args.duration
        else:
            end_sec = args.end
        if end_sec <= start_sec:
            print("End time must be greater than start time.", file=sys.stderr)
            cap.release()
            return 1

    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))
    if frame_count > 0:
        start_frame = max(0, min(start_frame, frame_count - 1))
        end_frame = max(start_frame + 1, min(end_frame, frame_count))
    n_frames = end_frame - start_frame
    if n_frames <= 0:
        print("No frames to write (check --start / --end / --duration).", file=sys.stderr)
        cap.release()
        return 1
    print(
        f"[trim] Requested range: {start_sec:.2f}s to {end_sec:.2f}s "
        f"(frames {start_frame}-{end_frame - 1}, total {n_frames} frame(s))"
    )

    out = args.out
    if out is None:
        out = src.parent / f"{src.stem}_trim.mp4"
    else:
        out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Could not open writer: {out}", file=sys.stderr)
        cap.release()
        return 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    written = 0
    progress_every = max(1, int(round(fps * 5)))  # print roughly every 5 seconds
    try:
        for _ in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
            if written % progress_every == 0 or written == n_frames:
                pct = (written / n_frames) * 100
                print(
                    f"[trim] Progress: {written}/{n_frames} frame(s) "
                    f"({written / fps:.1f}s, {pct:.1f}%)"
                )
    finally:
        cap.release()
        writer.release()

    print(
        f"Wrote {written} frame(s) (~{written / fps:.2f}s at {fps:.3f} fps): {out} "
        f"(requested {start_sec:.2f}s–{end_sec:.2f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
