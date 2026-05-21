# Curbsight

Ultralytics YOLO-based tooling for parking analysis: **preprocessing** and optional **augmentation** helpers for video, with **`parking_management.py`** as the main script for **parking-slot occupancy**, a **FastAPI** backend, and a **React** operator/signage frontend. The older **`vehicle_detection.py`** utility is kept for occasional quick experiments.

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

**`parking_management.py`** estimates **which parking spaces are occupied**: it runs YOLO via **`VotingParkingManagement`** (extends Ultralytics `ParkingManagement` with per-slot flags and optional majority voting) together with **polygon regions** in `bounding_boxes.json`, then writes an annotated video, per-inference JSON events, and JPEG snapshots for the backend.

Prepare footage with **[Preprocessing](#preprocessing)** and optionally **[Augmentation](#augmentation)** above, then export the bounding boxes to `bounding_boxes.json` (see [Ultralytics parking management](https://docs.ultralytics.com/guides/parking-management/)):

```bash
python -c "from ultralytics import solutions; solutions.ParkingPtsSelection()"
```

### `parking_management.py`

Each run produces:

- **Annotated video** — occupancy overlays; length and FPS match the input (frames between inferences reuse the last overlay).
- **`parking_events.jsonl`** — one JSON object per published inference (occupancy counts, frame indices, optional path to a snapshot JPEG).
- **`inferred_frames/`** — one inferred JPEG per published event (cleared at the start of each run when using default overwrite behavior).

**Default paths** (when flags are omitted):

| Output                                            | Default location                                                  |
| ------------------------------------------------- | ----------------------------------------------------------------- |
| Bounding boxes JSON (`-j`)                        | `<video-parent>/bounding_boxes.json`                              |
| Video (`-o`)                                      | `<video-parent>/output/parking_management_out.mp4`                |
| Events (`--events-out`)                           | `<video-parent>/output/parking_events.jsonl`                      |
| Inferred JPEGs (`--inferred-frames-dir`)          | `<video-parent>/output/inferred_frames/` (cwd)                    |
| Validation metrics (`--metrics-out`, with `--gt`) | `<video-parent>/output/validation_metrics.json`                   |

For a **webcam index** or **URL** source, `<video-parent>` is the current working directory (outputs still go under `output/` there).

Each JSONL event includes `source_id`, `street_id`, `frame_index`, `inference_index`, `stride`, spot counts, `occupancy_ratio`, `total_tracks`, and `inferred_image_path` when a snapshot was written.

Useful script arguments:

| Option                  | Description                                                                                                                                                                                                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--weights`, `-w`       | YOLO weights (default `yolo26n.pt`)                                                                                                                                                                                                                                |
| `--conf`                | Detection confidence threshold (default `0.1`)                                                                                                                                                                                                                     |
| `--iou`                 | IoU threshold for object detection (default `0.7`)                                                                                                                                                                                                                 |
| `--show`                | Open a preview window; if omitted, results are saved to an output video file                                                                                                                                                                                       |
| `--out`, `-o`           | Output video path (default `<source-dir>/output/parking_management_out.mp4`)                                                                                                                                                                                              |
| `--json`, `-j`          | Parking regions JSON (default repo-root `bounding_boxes.json`)                                                                                                                                                                                                     |
| `--classes`             | Comma-separated COCO class ids (e.g. `2,3,5,7` for vehicles)                                                                                                                                                                                                       |
| `--stride <N>`          | Run inference every N frames; output video still matches input length (default `60`)                                                                                                                                                                               |
| `--max-frames <M>`      | Stop after M frames (optional; useful for quick tests)                                                                                                                                                                                                             |
| `--vote-radius <R>`     | Majority-vote occupancy at each anchor using frames `f±2`, `f±4`, … (default `2` → five frames: `f-4`, `f-2`, `f`, `f+2`, `f+4`). Set `0` to disable. Skipped automatically if `--stride` is too small for non-overlapping vote windows (`R=2` needs `stride > 8`) |
| `--no-verbose`          | Disable verbose tracker/detection output                                                                                                                                                                                                                           |
| `--events-out`          | Per-inference JSONL for backend ingestion (default `<source-dir>/output/parking_events.jsonl`)                                                                                                                                                                            |
| `--publish-every <N>`   | Write one JSON event every N inferences (default `1`). With `--stride X`, events land every `X×N` frames.                                                                                                                                                          |
| `--inferred-frames-dir` | Directory for per-event JPEG snapshots (default `inferred_frames/`)                                                                                                                                                                                      |

**Validation** (optional; implemented in `parking_metrics.py`):

| Option                    | Description                                                                                                                                                                                                       |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--gt <csv>`              | Ground-truth CSV: `spot_id`, `start_frame`, `end_frame`, `status` (`occupied` / `available` / `unknown`; aliases like `free` accepted). Spot index _i_ in `bounding_boxes.json` maps to `spot_id` = `str(i + 1)`. |
| `--metrics-out`           | Write accuracy / per-spot metrics JSON (default `<source-dir>/output/validation_metrics.json`)                                                                                                                           |
| `--disagreements-out`     | Optional CSV of `(frame_index, spot_id, gt, pred)` mismatches                                                                                                                                                     |
| `--validation-events-out` | Optional JSONL of per-inference per-spot snapshots (separate from `--events-out`)                                                                                                                                 |
| `--no-video`              | Skip annotated video (faster metrics-only runs with `--gt`)                                                                                                                                                       |
| Option                    | Description                                                                                                                                                                                                       |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--gt <csv>`              | Ground-truth CSV: `spot_id`, `start_frame`, `end_frame`, `status` (`occupied` / `available` / `unknown`; aliases like `free` accepted). Spot index _i_ in `bounding_boxes.json` maps to `spot_id` = `str(i + 1)`. |
| `--metrics-out`           | Write accuracy / per-spot metrics JSON (default `<source-dir>/validation_metrics.json`)                                                                                                                           |
| `--disagreements-out`     | Optional CSV of `(frame_index, spot_id, gt, pred)` mismatches                                                                                                                                                     |
| `--validation-events-out` | Optional JSONL of per-inference per-spot snapshots (separate from `--events-out`)                                                                                                                                 |
| `--no-video`              | Skip annotated video (faster metrics-only runs with `--gt`)                                                                                                                                                       |

Example commands:

```bash
# DEFAULT: inference every 60 frames with 5-frame majority vote; outputs under <video-parent>/output, no metrics
python parking_management.py data/clipped/day_test/recording.mp4

# Finer sampling (more events, slower)
python parking_management.py data/clipped/day_test/recording.mp4 --stride 30

# Disable majority vote (single-frame occupancy per inference step)
python parking_management.py data/clipped/day_test/recording.mp4 --vote-radius 0

# Explicit output paths and event stream
python parking_management.py data/clip_cropped.mp4 -o data/parking_out.mp4 --events-out data/parking_events.jsonl

# Custom regions file and vehicle classes only (car, motorcycle, bus, truck)
python parking_management.py data/clip_cropped.mp4 -j bounding_boxes.json --classes 2,3,5,7

# Quick test on first 300 frames
python parking_management.py data/clip_cropped.mp4 --max-frames 300

# Run on webcam (index 0) with live preview instead of saving video
python parking_management.py 0 --show

# Validate against ground truth CSV (default stride, vote smoothing)
python parking_management.py data/clipped/day_test/recording.mp4 --gt

# Validate against ground truth CSV (frame-accurate; disable vote smoothing)
python parking_management.py data/clipped/day_test/recording.mp4 --gt --stride 1 --vote-radius 0
```

Full help: `python parking_management.py -h`.

### End-to-end parking example (after preprocessing)

```bash
python parking_management.py data/clipped/day_test/recording.mp4
```

## Backend API

The **`api/`** package is a FastAPI server that runs `parking_management` in a **background process** for the operator dashboard. While inference runs, it writes **`parking_events.jsonl`** and annotated JPEGs under **`inferred_frames/`** at the repo root (per-session outputs are planned; see TODO in `api/services.py`).

Start the server from the repo root:

```bash
uvicorn api:app --reload
```

Open **http://127.0.0.1:8000/docs** for interactive API docs and to test the endpoints.

### Session data layout

| File                  | Role                               |
| --------------------- | ---------------------------------- |
| `recording.mp4`       | Source video for inference         |
| `bounding_boxes.json` | Parking slot polygons              |
| `reference_frame.jpg` | Still for region overlay in the UI |

### Endpoints

| Endpoint                                     | Description                                                                      |
| -------------------------------------------- | -------------------------------------------------------------------------------- |
| `GET /health`                                | Health check                                                                     |
| `GET /sessions`                              | `{ "session_ids": ["2025-05-18", ...] }` — complete session folders only         |
| `GET /sessions/{session_id}/regions`         | `list` of `{ "points": [[x, y], ...] }` (`ParkingRegion`)                        |
| `GET /sessions/{session_id}/reference-frame` | Reference JPEG                                                                   |
| `GET /sessions/{session_id}/video`           | Source video (`recording.mp4`)                                                   |
| `GET /videos`                                | **Deprecated** — flat `.mp4`/`.mov` basenames directly under `data/`             |
| `GET /inference/status`                      | `idle`, `running`, `started`, or `stopped`                                       |
| `POST /inference/start`                      | Start inference (`session_id` or legacy `video_filename`)                        |
| `POST /inference/stop`                       | Stop a running job                                                               |
| `GET /frames/{image_name}`                   | Inferred snapshot JPEG from the active run                                       |
| `WS /ws/events`                              | Stream JSONL inference events + lifecycle `{ "type": "status", "state": "..." }` |

### Start inference

**Session** (uses that folder’s `recording.mp4` and `bounding_boxes.json`):

```json
{
  "session_id": "2025-05-18",
  "stride": 60,
  "vote_radius": 2,
  "publish_every": 1
}
```

**Legacy** flat file under `data/` (uses repo-root `bounding_boxes.json`):

```json
{
  "video_filename": "clip_cropped.mp4",
  "stride": 60,
  "vote_radius": 2,
  "publish_every": 1
}
```

Provide **either** `session_id` or `video_filename`. Optional: `max_frames`, `conf` (default `0.1`), `iou` (default `0.7`).

### WebSocket event fields

Each inference line (no `type` field) includes fields such as `timestamp_iso`, `source_id`, `street_id`, `frame_index`, `inference_index`, `stride`, `occupied_spots`, `available_spots`, `total_spots`, `occupancy_ratio`, `total_tracks`, and `inferred_image_path` (basename for `GET /frames/{image_name}`).

## Frontend

The **`frontend/`** app is a React + TypeScript + Vite UI for operators and a driver-facing street sign. It talks to the [Backend API](#backend-api) over HTTP and WebSocket while inference runs.

### Requirements

- **Node.js 18+** and npm
- Backend running at **http://127.0.0.1:8000** (see above)
- At least one `.mp4` or `.mov` file in **`data/`**

### Run locally

From the repo root, start the API first, then the dev server:

```bash
# Terminal 1 — backend
uvicorn api:app --reload

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Open **http://127.0.0.1:5173**. Vite proxies `/api` and `/frames` to port 8000; the WebSocket client connects directly to `ws://localhost:8000/ws/events`.

Production build:

```bash
cd frontend
npm run build
npm run preview   # serves dist/ on port 4173 by default
```

### Pages

| Route                 | Purpose                                                                    |
| --------------------- | -------------------------------------------------------------------------- |
| `/` (Dashboard)       | Start/stop inference, live annotated frame, occupancy gauge, history chart |
| `/sign` (Street Sign) | Same controls plus a large **spaces available** display for signage        |

### Project layout

```
frontend/
  src/pages/          DashboardPage, SignPage
  src/components/     ControlPanel, FrameViewer, OccupancyGauge, …
  src/hooks/          useInferenceSocket, useFramePlayer
  src/api/client.ts   REST helpers (/api → backend)
  vite.config.ts      dev proxies for /api and /frames
```

More detail: [`frontend/README.md`](frontend/README.md).

## Vehicle Detection (Legacy)

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
