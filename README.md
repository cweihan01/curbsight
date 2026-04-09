# Curbsight

Ultralytics YOLO–based tooling: **preprocessing** helpers for video, **`parking_management.py`** for **parking-slot occupancy** using `bounding_boxes.json`, and **`vehicle_detection.py`** for plain vehicle detection.

## Requirements

- Python 3.10+
- Webcam or media files for input

## Install

```bash
cd curbsight
python -m venv .venv
```

Activate the virtual environment:

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **macOS / Linux:** `source .venv/bin/activate`

Then install dependencies:

```bash
pip install -r requirements.txt
```

On first run, YOLO weights (default `yolo26n.pt`) download automatically (if not already present).

## Preprocessing

These scripts only reshape or sample your media (OpenCV). Use them before you annotate parking regions or run the parking pipeline so resolution and framing stay consistent.

| Script | Role |
|--------|------|
| `trim_video.py` | Keep a time range from a long file |
| `crop_video.py` | Crop every frame (e.g. remove a bottom bar) |
| `extract_frame.py` | Save one frame as an image (e.g. for drawing slot polygons) |

### Order when building the parking pipeline

Run preprocessing in this order:

1. **`trim_video.py`** — Shorten the clip (time and file size).
2. **`crop_video.py`** — Drop fixed regions so pixels match what you will analyze.
3. **`extract_frame.py`** — Grab a still from the **cropped** video; use it to define polygons at the same width/height as the cropped clip.

Then define regions by running the following command to open the parking points selection tool:

```python
from ultralytics import solutions
solutions.ParkingPtsSelection()
# Select the image that is a frame from the preprocessed video,
# and save the bounding boxes to the file `bounding_boxes.json`.
```

Finally, we can run the parking occupancy script.

### Example preprocessing commands

```bash
# Trim to a 3-minute clip starting at 2 minutes
python trim_video.py data/raw.MOV --start 120 --duration 180 -o data/clip.mp4

# Remove 200 px from the bottom of every frame
python crop_video.py data/clip.mp4 --remove-bottom 200 -o data/clip_cropped.mp4

# Save frame 100 (0-based) for annotating parking slots
python extract_frame.py data/clip_cropped.mp4 -f 100 -o data/reference_frame.jpg
```

### `trim_video.py`

Requires exactly one of `--duration`, `--end`, or `--middle` (seconds). Optional `--start` (ignored if `--middle`).

```bash
python trim_video.py data/raw.MOV --duration 180 -o data/clip.mp4
python trim_video.py data/raw.MOV --start 60 --end 240 -o data/clip.mp4
python trim_video.py data/raw.MOV --middle 120 -o data/clip.mp4
```

Full help: `python trim_video.py -h`.

### `crop_video.py`

At least one `--remove-top`, `--remove-bottom`, `--remove-left`, or `--remove-right` must be positive.

```bash
python crop_video.py data/clip.mp4 --remove-bottom 200 -o data/clip_cropped.mp4
```

Full help: `python crop_video.py -h`.

### `extract_frame.py`

```bash
python extract_frame.py data/clip_cropped.mp4 -f 100
python extract_frame.py data/clip_cropped.mp4 -f 0 -o data/reference_frame.jpg
```

Full help: `python extract_frame.py -h`.

Each preprocessing script’s module docstring repeats these examples.


## Parking occupancy

**`parking_management.py`** estimates **which parking spaces are occupied**: it runs a YOLO model together with **polygon regions** in `bounding_boxes.json` and overlays occupancy on video. It is **not** the same as plain vehicle detection—slots come from your JSON, not ad-hoc boxes.

Prepare footage with **Preprocessing** above, then export or edit `bounding_boxes.json` (see [Ultralytics parking management](https://docs.ultralytics.com/guides/parking-management/)):

```bash
python -c "from ultralytics import solutions; solutions.ParkingPtsSelection()"
```

### `parking_management.py`

Defaults: `bounding_boxes.json` beside the script, output `parking_management_out.mp4` in the **current working directory**.

```bash
python parking_management.py data/clip_cropped.mp4 -o data/parking_out.mp4
python parking_management.py data/clip_cropped.mp4 -j bounding_boxes.json --classes 2,3,5,7 --stride 10 --no-verbose
python parking_management.py 0 --show
```

Full help: `python parking_management.py -h`.

### End-to-end parking example (after preprocessing)

```bash
python parking_management.py data/clip_cropped.mp4 -o data/parking_out.mp4 --stride 60
```

## Vehicle detection

**`vehicle_detection.py`** is for **detecting vehicles only**: YOLO on an image, video, folder, URL, or webcam with **bounding boxes** for COCO vehicle classes (car, motorcycle, bus, truck). It does **not** read `bounding_boxes.json` and does **not** report per-slot occupancy—use **Parking occupancy** above for that.

### `vehicle_detection.py`

```bash
python vehicle_detection.py
python vehicle_detection.py data/ralphs1.jpg --save --save-txt --save-conf
python vehicle_detection.py --show
python vehicle_detection.py path/to/video.mp4 --save
python vehicle_detection.py https://ultralytics.com/images/bus.jpg
```

Useful options:

| Option | Description |
|--------|-------------|
| `--show` | Open a preview window (needs a display) |
| `--save` | Save annotated results to `runs/detect/` |
| `--weights PATH` | Use different `.pt` weights |
| `--device cpu` or `--device 0` | Force CPU or a specific GPU |
| `--conf 0.25` | Confidence threshold |

Full flags: `python vehicle_detection.py -h`.
