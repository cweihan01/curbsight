import multiprocessing as mp
from pathlib import Path

from parking_management import run_parking_management

from api.schemas import InferenceState, StartInferenceRequest

# Repo root is parent of package `api` (this file lives at api/services.py)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
EVENTS_PATH = REPO_ROOT / "parking_events.jsonl"
FRAMES_DIR = REPO_ROOT / "parking_management_frames"
OUT_PATH = REPO_ROOT / "parking_management_out.mp4"

VIDEO_SUFFIXES = frozenset({".mp4", ".mov"})

# Running inference process; None if not running
inference_process: mp.Process | None = None
# True after a running job is terminated by user; set to False on next inference start
stopped_by_user: bool = False


def process_running(proc: mp.Process | None) -> bool:
    return proc is not None and proc.is_alive()


def get_inference_state() -> InferenceState:
    """Current lifecycle state for status polling and WebSocket heartbeats."""
    global inference_process

    if process_running(inference_process):
        return InferenceState.running
    if stopped_by_user:
        return InferenceState.stopped
    # Process natural completion or never started
    return InferenceState.idle


def list_data_videos() -> list[str]:
    """Basenames of video files directly under data/."""
    if not DATA_DIR.is_dir():
        return []
    names: list[str] = []
    for path in DATA_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            names.append(path.name)
    return sorted(names)


def resolve_data_video(filename: str) -> Path:
    """Resolve a basename under data/; raise ValueError if invalid or missing."""
    filename = filename.strip()
    if not filename:
        raise ValueError("video_filename is required.")

    if filename != Path(filename).name:
        raise ValueError("Only a base filename is allowed (no paths).")

    filepath = (DATA_DIR / filename).resolve()
    data_root = DATA_DIR.resolve()
    try:
        filepath.relative_to(data_root)
    except ValueError as exc:
        raise ValueError("Invalid video path.") from exc

    if not filepath.is_file():
        raise ValueError(f"Video not found: {filename}")
    if filepath.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("Not a supported video file type.")

    return filepath


def run_inference_process(req_data: dict[str, object]) -> None:
    req = StartInferenceRequest(**req_data)
    source_path = resolve_data_video(req.video_filename)

    # Note: json_path is assumed to be constant (reused across any chosen video)
    # TODO: Might not want to overwrite out_path (video output) and
    # events_out_path (JSON events output) on each run; may want to write to
    # new files each run
    run_parking_management(
        source=str(source_path),
        out_path=OUT_PATH,
        conf=req.conf,
        iou=req.iou,
        no_verbose=True,
        stride=req.stride,
        max_frames=req.max_frames,
        events_out_path=EVENTS_PATH,
        publish_every=req.publish_every,
        inferred_frames_dir=FRAMES_DIR,
    )


def spawn_inference(req: StartInferenceRequest) -> None:
    """Start inference in a daemon child process."""
    global inference_process, stopped_by_user

    stopped_by_user = False
    inference_process = mp.Process(
        target=run_inference_process,
        args=(req.model_dump(),),
        daemon=True,
    )
    inference_process.start()


def terminate_inference() -> None:
    global inference_process, stopped_by_user

    if not process_running(inference_process):
        return
    assert inference_process is not None

    stopped_by_user = True
    inference_process.terminate()
    inference_process.join(timeout=3.0)
    inference_process = None
