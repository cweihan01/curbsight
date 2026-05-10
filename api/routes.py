import asyncio
import json

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from api import services as svc
from api.schemas import StartInferenceRequest

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# TODO: Use Pydantic model for response
# TODO: Track stopped jobs - return stopped instead of idle if stopped halfway
@router.get("/inference/status")
def inference_status() -> dict[str, str]:
    if svc.process_running(svc.inference_process):
        return {"status": "running"}
    return {"status": "idle"}


@router.post("/inference/start")
async def start_inference(req: StartInferenceRequest) -> dict[str, str]:
    if svc.process_running(svc.inference_process):
        raise HTTPException(status_code=409, detail="Inference is already running.")

    svc.spawn_inference(req)
    return {"status": "started"}


@router.post("/inference/stop")
async def stop_inference() -> dict[str, str]:
    if not svc.process_running(svc.inference_process):
        return {"status": "idle"}
    svc.terminate_inference()
    return {"status": "stopped"}


@router.get("/frames/{image_name}")
def get_frame(image_name: str) -> FileResponse:
    image_path = (svc.FRAMES_DIR / image_name).resolve()
    frames_root = svc.FRAMES_DIR.resolve()
    if frames_root not in image_path.parents:
        raise HTTPException(status_code=400, detail="Invalid frame path.")
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Frame not found.")
    return FileResponse(image_path)


@router.websocket("/ws/events")
async def ws_events(websocket: WebSocket) -> None:
    await websocket.accept()
    svc.EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    svc.EVENTS_PATH.touch(exist_ok=True)

    try:
        with svc.EVENTS_PATH.open("r", encoding="utf-8") as f:
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
                    if not svc.process_running(svc.inference_process):
                        await websocket.send_json({"type": "status", "state": "idle"})
                    await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return
