# Curbsight

Vehicle detection using Ultralytics YOLO (COCO classes: car, motorcycle, bus, truck).

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

## Run

```bash
python vehicle_detection.py
```

With no arguments, this uses webcam `0`. You can pass an image path, video, folder, URL, or another camera index.

Useful options:

| Option | Description |
|--------|-------------|
| `--show` | Open a preview window (needs a display) |
| `--save` | Save annotated frames to `runs/detect/` |
| `--weights PATH` | Use different `.pt` weights |
| `--device cpu` or `--device 0` | Force CPU or a specific GPU |
| `--conf 0.25` | Confidence threshold |

Examples:

```bash
python vehicle_detection.py data/ralphs1.jpg --save --save-txt --save-conf
python vehicle_detection.py --show
python vehicle_detection.py path/to/video.mp4 --save
python vehicle_detection.py https://ultralytics.com/images/bus.jpg
```
