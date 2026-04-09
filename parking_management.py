"""
Parking occupancy on video using Ultralytics ParkingManagement and polygon regions in JSON.

Regions default to bounding_boxes.json next to this script. Use --stride to infer only every Nth
frame and reuse the last overlay in between (faster; same output duration as the source).

Typical order (see README.md): trim_video → crop_video → extract_frame → define regions in
bounding_boxes.json (e.g. via Ultralytics ParkingPtsSelection on the extracted still) → run this
on the cropped clip.

Run:

  python parking_management.py data/clip_cropped.mp4 -o data/parking_out.mp4
  python parking_management.py data/clip_cropped.mp4 --stride 5 --no-verbose
  python parking_management.py data/clip_cropped.mp4 -j bounding_boxes.json --classes 2,3,5,7

Full help:

  python parking_management.py -h
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

DEFAULT_WEIGHTS = "yolo26n.pt"
DEFAULT_JSON = Path(__file__).resolve().parent / "bounding_boxes.json"


def parse_classes(s: str | None) -> list[int] | None:
    if not s or not s.strip():
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def open_capture(source: str) -> cv2.VideoCapture:
    cap_src: str | int = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_src)
    return cap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Ultralytics ParkingManagement on video using bounding_boxes.json."
    )
    p.add_argument(
        "source",
        help="Input video path, or webcam index (e.g. 0).",
    )
    p.add_argument(
        "--json",
        "-j",
        type=Path,
        default=DEFAULT_JSON,
        help=f"Parking regions JSON (default: {DEFAULT_JSON}).",
    )
    p.add_argument(
        "--weights",
        "-w",
        default=DEFAULT_WEIGHTS,
        help=f"YOLO weights .pt (default: {DEFAULT_WEIGHTS}).",
    )
    p.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("parking_management_out.mp4"),
        help="Output video path (default: parking_management_out.mp4).",
    )
    p.add_argument(
        "--tracker",
        default="botsort.yaml",
        help="Tracker config (default: botsort.yaml).",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="Detection confidence threshold (default: 0.1).",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold (default: 0.7).",
    )
    p.add_argument(
        "--classes",
        default="",
        help='Comma-separated COCO class ids to track, e.g. "2,3,5,7" for vehicles. Empty = all.',
    )
    p.add_argument(
        "--device",
        default="",
        help="Inference device: cpu, 0, cuda:0, ... (empty = auto).",
    )
    p.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable tracker/detection console spam.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show annotated frames in a window (needs GUI).",
    )
    p.add_argument(
        "--line-width",
        type=int,
        default=None,
        help="Box/line width for visualization (default: auto).",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        metavar="N",
        help="Run the model every N frames (default: 1). Between runs, the last annotated frame "
        "is written again so length and FPS match the input.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="M",
        help="Stop after M frames (optional; useful for quick tests).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import solutions
    except ImportError:
        print("Install dependencies: pip install -r requirements.txt", file=sys.stderr)
        return 1

    json_path = args.json.resolve()
    if not json_path.is_file():
        print(f"JSON not found: {json_path}", file=sys.stderr)
        return 1

    if args.stride < 1:
        print("--stride must be >= 1.", file=sys.stderr)
        return 1
    if args.max_frames is not None and args.max_frames < 1:
        print("--max-frames must be >= 1.", file=sys.stderr)
        return 1

    cap = open_capture(args.source)
    if not cap.isOpened():
        print(f"Error opening video source: {args.source!r}", file=sys.stderr)
        return 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Error opening video writer: {out_path}", file=sys.stderr)
        cap.release()
        return 1

    classes = parse_classes(args.classes)
    pm_kwargs: dict = {
        "model": args.weights,
        "json_file": str(json_path),
        "tracker": args.tracker,
        "conf": args.conf,
        "iou": args.iou,
        "verbose": not args.no_verbose,
        "show": args.show,
    }
    if classes is not None:
        pm_kwargs["classes"] = classes
    if args.device:
        pm_kwargs["device"] = args.device
    if args.line_width is not None:
        pm_kwargs["line_width"] = args.line_width

    parking = solutions.ParkingManagement(**pm_kwargs)

    last_plot = None
    index = 0
    infer_count = 0
    try:
        while cap.isOpened():
            ret, im0 = cap.read()
            if not ret:
                break
            if index % args.stride == 0 or last_plot is None:
                results = parking(im0)
                last_plot = results.plot_im
                infer_count += 1
            writer.write(last_plot)
            index += 1
            if args.max_frames is not None and index >= args.max_frames:
                break
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Wrote: {out_path.resolve()} ({index} frames, {infer_count} inferences, stride={args.stride})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
