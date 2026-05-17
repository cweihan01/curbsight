"""
Parking occupancy on video using Ultralytics ParkingManagement and polygon regions in JSON,
with streaming of per-inference availability events for backend ingestion.

Regions default to bounding_boxes.json next to this script. Use --stride to infer only every Nth
frame and reuse the last overlay in between (faster; same output duration as the source).

This script produces:
- An annotated output video with per-spot occupancy overlays.
- A JSONL stream and folder of inferred JPEG frames for each inference (see --events-out / --publish-every / --inferred-frames-dir).

Typical order of scripts (see README.md):
1. trim_video.py
2. crop_video.py
3. pixelate_video.py (optional)
4. extract_frame.py
5. define regions in bounding_boxes.json (via Ultralytics ParkingPtsSelection on the extracted frame)
6. run this script on the cropped clip from step 3

Run:

  python parking_management.py data/clip_cropped.mp4 -o data/parking_out.mp4 --events-out parking_events.jsonl
  python parking_management.py data/clip_cropped.mp4 --stride 30 --no-verbose
  python parking_management.py data/clip_cropped.mp4 -j bounding_boxes.json --classes 2,3,5,7

Full help:

  python parking_management.py -h
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
from ultralytics.solutions.solutions import SolutionResults

from voting_parking_management import VotingParkingManagement

DEFAULT_WEIGHTS = "yolo26n.pt"
DEFAULT_JSON = Path(__file__).resolve().parent / "bounding_boxes.json"
DEFAULT_SOURCE_ID = "ralphs_garage"
DEFAULT_STREET_ID = "le_conte_ave"
# Global run behavior:
# - True: fresh run (clear frames dir + overwrite events file)
# - False: incremental run (keep frames dir contents + append events file)
OVERWRITE_ON_EACH_RUN = True


def parse_classes(s: str | None) -> list[int] | None:
    """Parse comma-separated list of COCO class ids to track."""
    if not s or not s.strip():
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def open_capture(source: str) -> cv2.VideoCapture:
    """Open a video capture from a source (path, index, or URL)."""
    cap_src: str | int = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_src)
    return cap


def build_inference_event(
    *,
    frame_index: int,
    inference_index: int,
    stride: int,
    source_id: str,
    street_id: str,
    results: SolutionResults,
    inferred_image_path: str | None = None,
) -> dict[str, Any]:
    """
    Build a JSON event for the parking management system.

    Args:
        frame_index: The index of the frame in the video (0-indexed).
        inference_index: The index of the inference (1-indexed). This does not equal the frame_index if stride > 1.
        stride: Number of frames between inferences.
        source_id: The id of the source.
        street_id: The id of the street.
        results: The results of the inference (SolutionResults object from ParkingManagement.process()).

    Returns:
        A dictionary containing the event data.
        Example:
        ```json
        {
            "timestamp_iso": "2026-05-07T05:59:05.888053+00:00",
            "source_id": "ralphs_garage",
            "street_id": "le_conte_ave",
            "frame_index": 20,
            "inference_index": 3,
            "stride": 60,
            "occupied_spots": 6,
            "available_spots": 4,
            "total_spots": 10,
            "occupancy_ratio": 0.6,
            "total_tracks": 6
            "inferred_image_path": "parking_management_frames/inferred_003_frame_000020.jpg"
        }
        ```
    """
    # These are fields from Ultralytics ParkingManagement's SolutionResults object
    occupied_spots = results.filled_slots
    available_spots = results.available_slots
    total_tracks = results.total_tracks

    total_spots = occupied_spots + \
        available_spots if occupied_spots is not None and available_spots is not None else None
    occupancy_ratio = occupied_spots / \
        total_spots if total_spots is not None and total_spots > 0 else None

    return {
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "source_id": source_id,
        "street_id": street_id,
        "frame_index": frame_index,
        "inference_index": inference_index,
        "stride": stride,
        "occupied_spots": occupied_spots,
        "available_spots": available_spots,
        "total_spots": total_spots,
        "occupancy_ratio": occupancy_ratio,
        "total_tracks": total_tracks,
        "inferred_image_path": inferred_image_path,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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
        "--stride",
        type=int,
        default=60,
        metavar="N",
        help="Run the model every N frames (default: 60), i.e. an inference is run every "
        "N frames. Between inferences, the last annotated frame from the previous inference "
        "is duplicated so the output video length and FPS match the input.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="M",
        help="Stop after M frames (optional; useful for quick tests).",
    )
    p.add_argument(
        "--events-out",
        type=Path,
        default=Path("parking_events.jsonl"),
        help="File to write per-inference JSON events here for backend ingestion to"
        "(default: parking_events.jsonl).",
    )
    p.add_argument(
        "--publish-every",
        type=int,
        default=1,
        metavar="N",
        help="Write one event every N inferences (default: 1). If used with --stride <X>, "
        "the event will be written every X*Nth frame.",
    )
    p.add_argument(
        "--inferred-frames-dir",
        type=Path,
        default=Path("parking_management_frames"),
        help="Directory to write a unique inferred/annotated JPEG for every "
        "published event. Default: parking_management_frames/",
    )
    p.add_argument(
        "--vote-radius",
        type=int,
        default=2,
        metavar="R",
        help="At each inference anchor f, majority-vote using (2*R+1) frames "
        "(default: R=2 -> f-4, f-2, f, f+2, f+4). Disabled if --stride is too small "
        "for non-overlapping windows (R=2 needs stride > 8). Use 0 to disable.",
    )
    return p.parse_args()


def run_parking_management(
    source: str,
    json_path: Path = DEFAULT_JSON,
    weights: str = DEFAULT_WEIGHTS,
    out_path: Path = Path("parking_management_out.mp4"),
    conf: float = 0.1,
    iou: float = 0.7,
    classes_csv: str = "",
    no_verbose: bool = False,
    show: bool = False,
    stride: int = 60,
    max_frames: int | None = None,
    events_out_path: Path = Path("parking_events.jsonl"),
    publish_every: int = 1,
    inferred_frames_dir: Path = Path("parking_management_frames"),
    vote_radius: int = 2,
    on_update: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> int:
    """
    Run parking management pipeline.

    Args:
        source: Input video path, URL, or webcam index string.
        json_path: Path to the bounding boxes JSON file.
        weights: Path to the YOLO weights file.
        out_path: Path to the output video file.
        conf: Confidence threshold for object detection.
        iou: IoU threshold for object detection.
        classes_csv: Comma-separated list of COCO class ids to track.
        no_verbose: Whether to disable verbose output.
        show: Whether to show the annotated frames in a window.
        stride: Number of frames between inferences.
        max_frames: Maximum number of frames to process.
        events_out_path: Path to the events output JSONL file.
        publish_every: Number of inferences between events.
        inferred_frames_dir: Path to the inferred frames directory.
        vote_radius: At each anchor f, sample (2*R+1) frames at f+-2, f+-4 for majority vote (0 = off).
        on_update: Optional callback to receive per-inference event data.
        should_stop: Optional callback to check if the run should stop.
    """
    # Resolve the bounding boxes JSON file path
    json_path = json_path.resolve()
    if not json_path.is_file():
        print(f"JSON not found: {json_path}", file=sys.stderr)
        return 1

    # Validate arguments
    if stride < 1:
        print("--stride must be >= 1.", file=sys.stderr)
        return 1
    if max_frames is not None and max_frames < 1:
        print("--max-frames must be >= 1.", file=sys.stderr)
        return 1
    if publish_every < 1:
        print("--publish-every must be >= 1.", file=sys.stderr)
        return 1
    if vote_radius < 0:
        print("--vote-radius must be >= 0.", file=sys.stderr)
        return 1

    # Open the input video source
    cap = open_capture(source)
    if not cap.isOpened():
        print(f"Error opening video source: {source!r}", file=sys.stderr)
        return 1

    # Get the input video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"[parking] Input: {source!r} | size={w}x{h} | fps={fps:.3f} | "
        f"frames={frame_count if frame_count > 0 else 'unknown'}"
    )

    # Open the output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Error opening video writer: {out_path}", file=sys.stderr)
        cap.release()
        return 1

    # Open the events output file (overwrite or append based on global run behavior)
    events_out_path.parent.mkdir(parents=True, exist_ok=True)
    events_file_mode = "w" if OVERWRITE_ON_EACH_RUN else "a"
    try:
        events_file = events_out_path.open(events_file_mode, encoding="utf-8")
    except OSError as e:
        print(
            f"Error opening events output file: {events_out_path} ({e})", file=sys.stderr)
        cap.release()
        writer.release()
        return 1

    # Create the inferred frames directory; clear it if we are overwriting on each run
    inferred_frames_dir.mkdir(parents=True, exist_ok=True)
    if OVERWRITE_ON_EACH_RUN:
        for jpg_path in inferred_frames_dir.glob("*.jpg"):
            try:
                jpg_path.unlink()
            except OSError as e:
                print(f"Warning: could not delete {jpg_path} ({e})", file=sys.stderr)

    # Build the ParkingManagement configs
    classes = parse_classes(classes_csv)
    pm_kwargs: dict = {
        "model": weights,
        "json_file": str(json_path),
        "conf": conf,
        "iou": iou,
        "verbose": not no_verbose,
        "show": show,
    }
    if classes is not None:
        pm_kwargs["classes"] = classes

    print(
        f"[parking] Config: stride={stride}, conf={conf}, iou={iou}, "
        f"weights={weights}, json={json_path}, vote_radius={vote_radius}"
    )
    print(
        f"[parking] Events: out={events_out_path.resolve()} | publish_every={publish_every} "
        f"| source_id={DEFAULT_SOURCE_ID} | street_id={DEFAULT_STREET_ID}"
    )

    parking = VotingParkingManagement(**pm_kwargs)

    last_plot = None
    index = 0
    infer_count = 0
    progress_every = max(1, int(round(fps * 5)))  # print roughly every 5 seconds
    try:
        # TODO: We can probably optimize this by advancing the frame index by the stride,
        # instead of reading each frame sequentially; depends on whether we need the
        # output video to match the input video length

        # Main loop for each frame in the input video
        while cap.isOpened():
            # Check if the run should stop based on the optional callback
            if should_stop is not None and should_stop():
                break

            # Read the next frame from the input video
            ret, im0 = cap.read()
            if not ret:
                break

            # Run the ParkingManagement model on the frame every N frames according to the stride
            if index % stride == 0 or last_plot is None:
                results = parking.process(
                    im0,
                    vote_radius=vote_radius,
                    stride=stride,
                    cap=cap,
                    frame_index=index,
                )
                last_plot = results.plot_im
                infer_count += 1
                should_publish = infer_count % publish_every == 0
                inferred_image_path: Path | None = (
                    None
                    if not should_publish
                    else inferred_frames_dir
                    / f"inferred_{infer_count:03d}_frame_{index:06d}.jpg"
                )
                event = build_inference_event(
                    frame_index=index,
                    inference_index=infer_count,
                    stride=stride,
                    source_id=DEFAULT_SOURCE_ID,
                    street_id=DEFAULT_STREET_ID,
                    results=results,
                    inferred_image_path=None if inferred_image_path is None else str(
                        inferred_image_path),
                )

                # Write the event data to the output JSONL file every N inferences according to publish_every
                if should_publish:
                    if inferred_image_path is not None:
                        cv2.imwrite(str(inferred_image_path), results.plot_im)
                    events_file.write(json.dumps(event) + "\n")
                    events_file.flush()

                # Callback to receive the event data
                if on_update is not None:
                    on_update(event)

            # Write the annotated frame to the output video
            writer.write(last_plot)

            # Print progress roughly every 5 seconds
            index += 1
            if index % progress_every == 0:
                if frame_count > 0:
                    pct = (index / frame_count) * 100
                    print(
                        f"[parking] Progress: {index}/{frame_count} frame(s) "
                        f"({index / fps:.1f}s, {pct:.1f}%) | inferences={infer_count}"
                    )
                else:
                    print(
                        f"[parking] Progress: {index} frame(s) ({index / fps:.1f}s) "
                        f"| inferences={infer_count}"
                    )

            # Stop after the maximum number of frames if specified
            if max_frames is not None and index >= max_frames:
                break
    finally:
        cap.release()
        writer.release()
        events_file.close()
        if show:
            cv2.destroyAllWindows()

    print(
        f"[parking] Done: wrote {index} frame(s), {infer_count} inferences, "
        f"stride={stride} -> {out_path.resolve()}"
    )
    return 0


def main() -> int:
    args = parse_args()
    return run_parking_management(
        source=args.source,
        json_path=args.json,
        weights=args.weights,
        out_path=args.out,
        conf=args.conf,
        iou=args.iou,
        classes_csv=args.classes,
        no_verbose=args.no_verbose,
        show=args.show,
        stride=args.stride,
        max_frames=args.max_frames,
        events_out_path=args.events_out,
        publish_every=args.publish_every,
        inferred_frames_dir=args.inferred_frames_dir,
        vote_radius=args.vote_radius,
    )


if __name__ == "__main__":
    raise SystemExit(main())
