import multiprocessing as mp
from pathlib import Path

from parking_management import run_parking_management

from api.schemas import StartInferenceRequest

# Repo root is parent of package `api` (this file lives at api/services.py)
REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS_PATH = REPO_ROOT / "parking_events.jsonl"
FRAMES_DIR = REPO_ROOT / "parking_management_frames"

inference_process: mp.Process | None = None


def process_running(proc: mp.Process | None) -> bool:
    return proc is not None and proc.is_alive()


# TODO: Filepath args should not be passed into run_parking_management
# from the API (client will not know these paths)
def run_inference_process(req_data: dict[str, object]) -> None:
    req = StartInferenceRequest(**req_data)
    run_parking_management(
        source=req.source,
        json_path=Path(req.json_path),
        weights=req.weights,
        out_path=Path(req.out),
        conf=req.conf,
        iou=req.iou,
        no_verbose=req.no_verbose,
        stride=req.stride,
        max_frames=req.max_frames,
        events_out_path=EVENTS_PATH,
        publish_every=req.publish_every,
        inferred_frames_dir=FRAMES_DIR,
    )


def spawn_inference(req: StartInferenceRequest) -> None:
    """Start inference in a daemon child process."""
    global inference_process
    inference_process = mp.Process(
        target=run_inference_process,
        args=(req.model_dump(),),
        daemon=True,
    )
    inference_process.start()


def terminate_inference() -> None:
    global inference_process
    if not process_running(inference_process):
        return
    assert inference_process is not None
    inference_process.terminate()
    inference_process.join(timeout=3.0)
