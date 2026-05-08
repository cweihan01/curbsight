"""
CurbSight API for parking management inference.

Run:
  uvicorn api:app --reload
"""
import asyncio
import multiprocessing as mp
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from parking_management import run_parking_management

REPO_ROOT = Path(__file__).resolve().parent
EVENTS_PATH = REPO_ROOT / "parking_events.jsonl"
FRAMES_DIR = REPO_ROOT / "parking_management_frames"

app = FastAPI(title="CurbSight API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference_process: mp.Process | None = None


# TODO: Make this align with CLI arguments and validation rules
class StartInferenceRequest(BaseModel):
    source: str = Field(..., description="Video path")
    out: str = "parking_management_out.mp4"
    stride: int = 1
    publish_every: int = 1
    max_frames: int | None = None
    json_path: str = "bounding_boxes.json"
    weights: str = "yolo26n.pt"
    conf: float = 0.1
    iou: float = 0.7
    no_verbose: bool = False


def process_running(proc: mp.Process | None) -> bool:
    return proc is not None and proc.is_alive()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# TODO: Use Pydantic model for response
# TODO: Track stopped jobs - return stopped instead of idle if stopped halfway
@app.get("/inference/status")
def inference_status() -> dict[str, str]:
    if process_running(inference_process):
        return {"status": "running"}
    return {"status": "idle"}


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


@app.post("/inference/start")
async def start_inference(req: StartInferenceRequest) -> dict[str, str]:
    global inference_process
    if process_running(inference_process):
        raise HTTPException(status_code=409, detail="Inference is already running.")

    inference_process = mp.Process(
        target=run_inference_process,
        args=(req.model_dump(),),
        daemon=True,
    )
    inference_process.start()
    return {"status": "started"}


@app.post("/inference/stop")
async def stop_inference() -> dict[str, str]:
    global inference_process
    if not process_running(inference_process):
        return {"status": "idle"}
    assert inference_process is not None
    inference_process.terminate()
    inference_process.join(timeout=3.0)
    return {"status": "stopped"}


@app.get("/frames/{image_name}")
def get_frame(image_name: str) -> FileResponse:
    image_path = (FRAMES_DIR / image_name).resolve()
    frames_root = FRAMES_DIR.resolve()
    if frames_root not in image_path.parents:
        raise HTTPException(status_code=400, detail="Invalid frame path.")
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Frame not found.")
    return FileResponse(image_path)


@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket) -> None:
    await websocket.accept()
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVENTS_PATH.touch(exist_ok=True)

    try:
        with EVENTS_PATH.open("r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if line:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        await websocket.send_json({"type": "warning", "message": "Malformed JSON event."})
                        continue
                    await websocket.send_json(payload)
                else:
                    if not process_running(inference_process):
                        await websocket.send_json({"type": "status", "state": "idle"})
                    await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return
