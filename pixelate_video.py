"""
Downsample and re-upsample a video to make it look more pixelated (CCTV/street-cam style).

Recommended order (see README.md): trim_video.py -> crop_video.py -> pixelate_video.py ->
extract_frame.py (if you still need a reference frame afterward).

Run:

  python pixelate_video.py data/clip_cropped.mp4 --scale 0.2 -o data/clip_pixelated.mp4
  python pixelate_video.py data/clip.mp4 --target-width 640 --jpeg-quality 35 -o out.mp4

Use either --scale or --target-width to control the downsample level.
Smaller scale and target-width means more pixelation.

Full help:

  python pixelate_video.py -h
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make a video look lower-quality by downsampling and scaling back up."
    )
    p.add_argument("video", type=Path, help="Input video path.")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output video (default: <input_stem>_pixelated.mp4 next to input).",
    )
    size_group = p.add_mutually_exclusive_group(required=False)
    size_group.add_argument(
        "--scale",
        type=float,
        default=0.3,
        metavar="RATIO",
        help="Downsample ratio in (0, 1]. 0.3 means 30%% resolution before upscaling (default: 0.3).",
    )
    size_group.add_argument(
        "--target-width",
        type=int,
        default=None,
        metavar="PX",
        help="Downsample to this width, preserving aspect ratio (overrides --scale).",
    )
    p.add_argument(
        "--fps-scale",
        type=float,
        default=1.0,
        metavar="RATIO",
        help="Multiply output FPS by this factor (<=1.0 can mimic choppier cameras, default: 1.0).",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        metavar="Q",
        help="Optional JPEG recompression quality 1..100 per frame (lower = more artifacts).",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Show side-by-side preview while writing (needs GUI).",
    )
    return p.parse_args()


def compute_downsample_size(
    width: int, height: int, scale: float, target_width: int | None
) -> tuple[int, int]:
    """
    Compute the downsampled width and height based on the input width and height,
    and the scale or target width.

    If the target width is provided, use it to compute the downsampled width and height.
    If the scale is provided, use it to compute the downsampled width and height.

    Returns the downsampled image's width and height.
    """
    if target_width is not None:
        if target_width < 2:
            raise ValueError("--target-width must be >= 2.")
        down_w = min(target_width, width)
        down_h = max(1, int(round(height * (down_w / width))))
        return down_w, down_h

    if not (0.0 < scale <= 1.0):
        raise ValueError("--scale must be in (0, 1].")
    down_w = max(1, int(round(width * scale)))
    down_h = max(1, int(round(height * scale)))
    return down_w, down_h


def maybe_jpeg_recompress(frame, quality: int | None):
    if quality is None:
        return frame
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return frame
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return frame if decoded is None else decoded


def main() -> int:
    args = parse_args()
    src = args.video.resolve()
    if not src.is_file():
        print(f"Video not found: {src}", file=sys.stderr)
        return 1

    if args.fps_scale <= 0:
        print("--fps-scale must be > 0.", file=sys.stderr)
        return 1
    if args.jpeg_quality is not None and not (1 <= args.jpeg_quality <= 100):
        print("--jpeg-quality must be in [1, 100].", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"Could not open video: {src}", file=sys.stderr)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"[pixelate] Input: {src} | size={width}x{height} | fps={fps:.3f} | "
        f"frames={frame_count if frame_count > 0 else 'unknown'}"
    )

    try:
        down_w, down_h = compute_downsample_size(
            width, height, args.scale, args.target_width)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        cap.release()
        return 1
    print(
        f"[pixelate] Settings: downsample={down_w}x{down_h}, fps_scale={args.fps_scale}, "
        f"jpeg_quality={args.jpeg_quality}, preview={args.preview}"
    )

    out = args.out
    if out is None:
        out = src.parent / f"{src.stem}_pixelated.mp4"
    else:
        out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    out_fps = max(1.0, fps * args.fps_scale)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, out_fps, (width, height))
    if not writer.isOpened():
        print(f"Could not open writer: {out}", file=sys.stderr)
        cap.release()
        return 1
    print(f"[pixelate] Writing to: {out} at {out_fps:.3f} fps")

    written = 0
    progress_every = max(1, int(round(fps * 5)))  # print roughly every 5 seconds
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Downsample the frame to the downsampled width and height using linear interpolation
            # and then upsample it to the original width and height using nearest neighbor interpolation
            low_res = cv2.resize(frame, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(low_res, (width, height),
                                   interpolation=cv2.INTER_NEAREST)
            pixelated = maybe_jpeg_recompress(pixelated, args.jpeg_quality)
            writer.write(pixelated)
            written += 1

            if written % progress_every == 0:
                if frame_count > 0:
                    pct = (written / frame_count) * 100
                    print(
                        f"[pixelate] Progress: {written}/{frame_count} frame(s) "
                        f"({written / fps:.1f}s, {pct:.1f}%)"
                    )
                else:
                    print(
                        f"[pixelate] Progress: {written} frame(s) ({written / fps:.1f}s)")

            if args.preview:
                # Side-by-side quick visual check while processing
                preview = cv2.hconcat([frame, pixelated])
                cv2.imshow("Original | Pixelated", preview)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        writer.release()
        if args.preview:
            cv2.destroyAllWindows()

    print(
        f"Wrote {written} frame(s) at {out_fps:.3f} fps: {out} "
        f"(downsampled to {down_w}x{down_h}, jpeg_quality={args.jpeg_quality})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
