import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

from parking_management import DEFAULT_JSON, run_parking_management

from api.schemas import InferenceState, ParkingRegion, StartInferenceRequest

# Repo root is parent of package `api` (this file lives at api/services.py)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
EVENTS_PATH = REPO_ROOT / "parking_events.jsonl"
FRAMES_DIR = REPO_ROOT / "inferred_frames"
OUT_PATH = REPO_ROOT / "parking_management_out.mp4"

VIDEO_SUFFIXES = frozenset({".mp4", ".mov"})

# Assume names in the data/clipped/<session_id> folder follows this naming convention
# where all 3 files are present
SESSION_VIDEO_NAME = "recording.mp4"
SESSION_REGIONS_NAME = "bounding_boxes.json"
SESSION_REFERENCE_FRAME_NAME = "reference_frame.jpg"

# Running inference process; None if not running
inference_process: mp.Process | None = None
# True after a running job is terminated by user; set to False on next inference start
stopped_by_user: bool = False


@dataclass(frozen=True)
class SessionPaths:
    """Resolved files under data/<session_id>/."""

    session_id: str
    session_dir: Path

    @property
    def video_path(self) -> Path:
        return self.session_dir / SESSION_VIDEO_NAME

    @property
    def regions_path(self) -> Path:
        return self.session_dir / SESSION_REGIONS_NAME

    @property
    def reference_frame_path(self) -> Path:
        return self.session_dir / SESSION_REFERENCE_FRAME_NAME


@dataclass(frozen=True)
class InferencePaths:
    source: Path
    json_path: Path


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


def session_dir_has_required_files(session_dir: Path) -> bool:
    """
    Check if a session directory has all the required files:
    - recording.mp4
    - bounding_boxes.json
    - reference_frame.jpg

    Returns True if all files are present, False otherwise.
    """
    return (
        (session_dir / SESSION_VIDEO_NAME).is_file()
        and (session_dir / SESSION_REGIONS_NAME).is_file()
        and (session_dir / SESSION_REFERENCE_FRAME_NAME).is_file()
    )


def resolve_session(session_id: str) -> SessionPaths:
    """
    Resolve a complete session under data/<session_id>/.

    Requires recording.mp4, bounding_boxes.json, and reference_frame.jpg.
    """
    session_id = session_id.strip()
    if not session_id:
        raise ValueError("session_id is required.")

    session_dir = (DATA_DIR / session_id).resolve()
    try:
        session_dir.relative_to(DATA_DIR.resolve())
    except ValueError as exc:
        raise ValueError(f"Session not found: {session_id}") from exc
    if not session_dir.is_dir():
        raise ValueError(f"Session not found: {session_id}")

    session = SessionPaths(session_id=session_id, session_dir=session_dir)
    if not session_dir_has_required_files(session_dir):
        raise ValueError(f"Session not found: {session_id}")

    return session


def list_sessions() -> list[str]:
    """
    Folder names under data/ that contain all three required session files.

    Returns a list of session IDs (folder names).
    """
    if not DATA_DIR.is_dir():
        return []
    session_ids: list[str] = []
    for path in sorted(DATA_DIR.iterdir()):
        if path.is_dir() and session_dir_has_required_files(path):
            session_ids.append(path.name)
    return session_ids


def load_session_regions(session_id: str) -> list[ParkingRegion]:
    """
    Load bounding_boxes.json for a session.
    """
    session = resolve_session(session_id)
    try:
        raw = json.loads(session.regions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {SESSION_REGIONS_NAME}: {exc}") from exc
    return [ParkingRegion.model_validate(region) for region in raw]


def list_data_videos() -> list[str]:
    """
    Basenames of video files directly under data/ (legacy flat layout).

    Deprecated: prefer GET /sessions.
    """
    if not DATA_DIR.is_dir():
        return []
    names: list[str] = []
    for path in DATA_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            names.append(path.name)
    return sorted(names)


def resolve_data_video(filename: str) -> Path:
    """
    Resolve a basename under data/ (legacy flat layout).

    Deprecated: prefer resolve_session().
    """
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


def resolve_inference_paths(req: StartInferenceRequest) -> InferencePaths:
    """Resolve video source and regions JSON for an inference job."""
    # Session layout
    if req.session_id:
        session = resolve_session(req.session_id)
        return InferencePaths(source=session.video_path, json_path=session.regions_path)

    # Legacy flat layout
    assert req.video_filename is not None
    return InferencePaths(
        source=resolve_data_video(req.video_filename),
        json_path=DEFAULT_JSON,
    )


def run_inference_process(req_data: dict[str, object]) -> None:
    req = StartInferenceRequest(**req_data)
    paths = resolve_inference_paths(req)

    # TODO: per-session events_out_path, inferred_frames_dir, out_path under session_dir
    run_parking_management(
        source=str(paths.source),
        json_path=paths.json_path,
        out_path=OUT_PATH,
        conf=req.conf,
        iou=req.iou,
        no_verbose=True,
        stride=req.stride,
        vote_radius=req.vote_radius,
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
