# Curbsight

Ultralytics YOLO-based tooling for parking analysis: **preprocessing** and optional **augmentation** helpers for video, with **`parking_management.py`** as the main script for **parking-slot occupancy** using `bounding_boxes.json`. The older **`vehicle_detection.py`** utility is kept for occasional quick experiments.

## Requirements

- Python 3.10+
- Webcam or media files for input

## Installation

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

Preprocessing is the necessary preparation of raw video before annotation/inference. It should eventually be part of the standard pipeline given a video feed.

| Script                                 | Role                                                        |
| -------------------------------------- | ----------------------------------------------------------- |
| [`trim_video.py`](#trim_videopy)       | Keep a time range from a long file                          |
| [`crop_video.py`](#crop_videopy)       | Crop every frame (e.g. remove a bottom bar)                 |
| [`extract_frame.py`](#extract_framepy) | Save one frame as an image (e.g. for drawing slot polygons) |

### Order when building the parking pipeline

Run preprocessing in this order:

1. **`trim_video.py`**: Shorten the clip (time and file size).
2. **`crop_video.py`**: Drop fixed regions so pixels match what you will analyze.
3. **`extract_frame.py`**: Grab a still from the **final preprocessed** video; use it to define polygons at the same width/height as that clip.

### Example preprocessing commands

```bash
# Trim to a 3-minute clip starting at 2 minutes
python trim_video.py data/raw.MOV --start 120 --duration 180 -o data/clip.mp4

# Remove 200 px from the bottom of every frame
python crop_video.py data/clip.mp4 --remove-bottom 200 -o data/clip_cropped.mp4

# Save frame 100 (0-based) for annotating parking spots
python extract_frame.py data/clip_cropped.mp4 -f 100 -o data/reference_frame.jpg
```

### `trim_video.py`

Requires exactly one of `--duration`, `--end`, or `--middle` (seconds). Optional `--start` (ignored if `--middle` is used).

```bash
# Trim to a 3-minute clip
python trim_video.py data/raw.MOV --duration 180 -o data/clip.mp4

# Trim to a 4-minute clip starting at 1 minute
python trim_video.py data/raw.MOV --start 60 --end 240 -o data/clip.mp4

# Trim to a 2-minute clip from the middle of the video
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
# Save frame 100 for annotating parking slots
python extract_frame.py data/clip_cropped.mp4 -f 100

# Save the first frame to the output file path specified
python extract_frame.py data/clip_cropped.mp4 -f 0 -o data/reference_frame.jpg
```

Full help: `python extract_frame.py -h`.

### Define Parking Regions

The parking regions are defined by the bounding boxes of the parking spots.

After extracting a frame from the preprocessed video, run the following command to open the parking points selection tool:

```python
from ultralytics import solutions
solutions.ParkingPtsSelection()
# This will bring up a GUI.
# Upload the image that is a frame from the preprocessed video,
# and save the bounding boxes to the file `bounding_boxes.json`.
```

Next, optionally run some augmentation techniques on the preprocessed video and then run the parking occupancy script `parking_management.py`.

## Augmentation

Augmentation is optional dataset tuning to better resemble real-world operating conditions. Unlike [preprocessing](#preprocessing), augmentation is not required for the core pipeline to run on an input feed. It is suggested to run various augmentation techniques with varying parameters (e.g. different pixelation levels) on the input video before running the parking occupancy script `parking_management.py`.

| Script                                   | Role                                           |
| ---------------------------------------- | ---------------------------------------------- |
| [`pixelate_video.py`](#pixelate_videopy) | Mimic low-res/compressed street-camera footage |

### `pixelate_video.py`

Downsamples each frame, then upscales with nearest-neighbor interpolation to make it look more pixelated.
Optionally adds JPEG artifacts and lowers FPS to better mimic real-world street cameras.

```bash
# Pixelate to 30% of the original resolution
python pixelate_video.py data/clip_cropped.mp4 --scale 0.3 -o data/clip_pixelated.mp4

# Downsample to 640 pixels wide and add JPEG artifacts
python pixelate_video.py data/clip.mp4 --target-width 640 --jpeg-quality 35 -o data/clip_cctv.mp4

# Downsample to 15% of the original resolution and 50% of the original FPS
python pixelate_video.py data/clip.mp4 --scale 0.15 --fps-scale 0.5 -o data/clip_cctv.mp4
```

Full help: `python pixelate_video.py -h`.

## Parking Occupancy Detection

**`parking_management.py`** estimates **which parking spaces are occupied**: it runs a YOLO model together with **polygon regions** in `bounding_boxes.json` and overlays occupancy on the video, outputting a new video with the occupancy.

Prepare footage with **[Preprocessing](#preprocessing)** and optionally **[Augmentation](#augmentation)** above, then export the bounding boxes to `bounding_boxes.json` (see [Ultralytics parking management](https://docs.ultralytics.com/guides/parking-management/)):

```bash
python -c "from ultralytics import solutions; solutions.ParkingPtsSelection()"
```

### `parking_management.py`

Run the script to get the occupancy of the parking spaces. The script will output a video with the occupancy of the parking spaces, and a JSONL file with per-inference events for backend ingestion.

The default JSON file containing the polygon regions bounding boxes is `bounding_boxes.json`.
The default video output is `parking_management_out.mp4` in the current working directory.
The default event output is `parking_events.jsonl` in the current working directory.

Useful script arguments:
| Option | Description |
| ------------------------------ | ---------------------------------------- |
| `--show` | Open a preview window; if omitted, results are saved to an output video file |
| `--iou <iou_threshold>` | IoU threshold for object detection |
| `--out <output_file_path>` | Save the output to the specified file path (default `parking_management_out.mp4`) |
| `--json <json_file_path>` | Use a custom bounding box JSON file (default `bounding_boxes.json`) |
| `--classes <classes>` | Restrict detection to certain vehicle classes |
| `--stride <N>` | Process every Nth frame (default `1`) |
| `--no-verbose` | Disable verbose output |
| `--events-out <path>` | Write per-inference JSON events to a `.jsonl` file (default `parking_events.jsonl`) |
| `--publish-every <N>` | Emit one JSON event every N inferences (default `1`). If used with `--stride <X>`, the event will be published every X\*Nth frame. |

Example commands:

```bash
# Basic usage: process a video and save the video output with overlaid occupancy results to parking_out.mp4
# and stream per-inference availability events to parking_events.jsonl
python parking_management.py data/clip_cropped.mp4 -o parking_out.mp4 --events-out parking_events.jsonl

# Process every 10th frame (output video will remain the same duration as the input video)
# This is recommended for faster processing (fewer inferences)
python parking_management.py data/clip_cropped.mp4 --stride 10

# Use a custom bounding box JSON file
python parking_management.py data/clip_cropped.mp4 -j bounding_boxes.json

# Restrict detection to certain vehicle classes (car, motorcycle, bus, truck)
python parking_management.py data/clip_cropped.mp4 --classes 2,3,5,7

# Run on webcam (index 0) and display the occupancy result live on screen instead of saving to a file
python parking_management.py 0 --show
```

Full help: `python parking_management.py -h`.

### End-to-end parking example (after preprocessing)

```bash
python parking_management.py data/clip_cropped.mp4 -o parking_out.mp4 --events-out parking_events.jsonl --stride 10
```

## Vehicle Detection

**`vehicle_detection.py`** is a legacy utility and is no longer part of the main workflow. We now use **`parking_management.py`** for parking-slot occupancy as the primary pipeline.

If needed for quick experiments, `vehicle_detection.py` runs YOLO vehicle detection only (car, motorcycle, bus, truck) on an image, video, folder, URL, or webcam. It does **not** read `bounding_boxes.json` and does **not** report per-spot occupancy.

### `vehicle_detection.py`

```bash
python vehicle_detection.py
python vehicle_detection.py data/frame.jpg --save --save-txt --save-conf
python vehicle_detection.py --show
python vehicle_detection.py path/to/video.mp4 --save
python vehicle_detection.py https://ultralytics.com/images/bus.jpg
```

Useful options:

| Option                         | Description                              |
| ------------------------------ | ---------------------------------------- |
| `--show`                       | Open a preview window (needs a display)  |
| `--save`                       | Save annotated results to `runs/detect/` |
| `--weights PATH`               | Use different `.pt` weights              |
| `--device cpu` or `--device 0` | Force CPU or a specific GPU              |
| `--conf 0.25`                  | Confidence threshold                     |

Full flags: `python vehicle_detection.py -h`.

## Resources

Object detection documentation:
https://docs.ultralytics.com/tasks/detect/

List of classes:
https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
