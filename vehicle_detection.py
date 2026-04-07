"""
Vehicle detection with Ultralytics YOLO: cars and other COCO vehicle classes with bounding boxes.

Uses pretrained Detect weights
Vehicle classes only: car, motorcycle, bus, truck.
"""

from __future__ import annotations

import argparse
import sys

# Pretrained Detect weights auto-download on first use
DEFAULT_WEIGHTS = "yolo26n.pt"


# COCO 80-class indices: car, motorcycle, bus, truck
VEHICLE_CLASS_IDS: list[int] = [2, 3, 5, 7]


def resolve_model(weights: str | None) -> str:
    if weights:
        return weights
    return DEFAULT_WEIGHTS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect vehicles (COCO: car, motorcycle, bus, truck) with YOLO."
    )
    p.add_argument(
        "source",
        nargs="?",
        default="0",
        help="Image path, video path, directory, URL, or webcam index (default: 0).",
    )
    p.add_argument(
        "--weights",
        "-w",
        default=None,
        help="Model weights (.pt). Default: yolo26n.pt (override with yolov8n.pt if needed).",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640).",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    p.add_argument(
        "--device",
        default="",
        help="Device, e.g. 0 or cpu (empty = auto).",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Save annotated results to runs/detect/.",
    )
    p.add_argument(
        "--save-txt",
        action="store_true",
        help="Save bounding boxes as YOLO .txt (normalized class cx cy w h) under runs/detect/.../labels/.",
    )
    p.add_argument(
        "--save-conf",
        action="store_true",
        help="Include confidence in each line of saved .txt (use with --save-txt).",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Display window (needs GUI).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install dependencies: pip install -r requirements.txt", file=sys.stderr)
        return 1

    weights = resolve_model(args.weights)
    model = YOLO(weights)
    classes = VEHICLE_CLASS_IDS

    kwargs: dict = {
        "source": args.source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "classes": classes,
        "verbose": True,
    }
    if args.device:
        kwargs["device"] = args.device
    if args.save:
        kwargs["save"] = True
    if args.save_txt:
        kwargs["save_txt"] = True
    if args.save_conf:
        kwargs["save_conf"] = True
    if args.show:
        kwargs["show"] = True

    results = model.predict(**kwargs)

    # TODO: Write the results (raw coordinates) to a file
    for i, r in enumerate(results):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls.item())
            label = names.get(cls_id, str(cls_id))
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()
            print(f"[{i}] {label}: conf={conf:.3f} box(xyxy)={xyxy}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
