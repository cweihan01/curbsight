# Data Directory

The `data/` directory contains the relevant input and output files for **parking_management.py**. The directory is structured as follows:

## `archive/`

Contains old/unused snapshots and footage.

## `clipped/`

Contains short videos (30 sec-2 min) trimmed from full recordings, which can be used for presentations, demonstrations, and brief testing. Each subfolder (e.g. `day_test/`) contains the input and output files associated with a single clip.

## `full_length/`

Contains full-length (untrimmed) recordings, which can be used to obtain accuracy metrics for validation. Each subfolder (e.g. `day/) contains the input and output files associated with a single clip.

## Street labels (`streets.json`)

Maps each `session_id` to a street row on the driver sign. New inference events include
`street_id` and `street_display_name` when the session is listed there.

## Subfolder Structure

Inputs (under `clipped/<subfolder>` and `full_length/<subfolder>`):

| File                  | Description                          |
| --------------------- | ----------------------------------   |
| `bounding_boxes.json` | Parking slot polygons                |
| `recording.mp4`       | Source video for inference           |
| `recording_link.txt`  | Google Drive link to `recording.mp4` |
| `reference_frame.jpg` | Still for drawing bounding boxes, region overlay in UI |
| `gt.csv`              | Ground truth labels (optional, for validation) |

Outputs (under `clipped/<subfolder>/output` and `full_length/<subfolder>/output`):

| File                     | Description                          |
| ---------------------    | ----------------------------------   |
| `inferred_frames/`       | Inferred JPEGs with occupancy annotation       |
| `parking_events.jsonl`   | Scene-level occupancy events           |
| `parking_management_out.mp4`  | Annotated occupancy video |
| `validation_metrics.json` | Spot-level accuracy metrics (optional)  |