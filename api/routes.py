import asyncio
import json

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse

from api import services as svc
from api.schemas import (
    HealthResponse,
    StartInferenceRequest,
    InferenceStatusResponse,
    InferenceState,
    VideosResponse,
)

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


# NOTE: This assumes video files are stored under data/
@router.get("/videos", status_code=status.HTTP_200_OK)
def list_videos() -> VideosResponse:
    """Available video files to run inference on."""
    return VideosResponse(filenames=svc.list_data_videos())


@router.get("/inference/status", status_code=status.HTTP_200_OK)
def inference_status() -> InferenceStatusResponse:
    return InferenceStatusResponse(status=svc.get_inference_state())


@router.post("/inference/start", status_code=status.HTTP_200_OK)
async def start_inference(req: StartInferenceRequest) -> InferenceStatusResponse:
    if svc.process_running(svc.inference_process):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Inference is already running.",
        )

    try:
        svc.resolve_data_video(req.video_filename)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    svc.spawn_inference(req)
    return InferenceStatusResponse(status=InferenceState.started)


@router.post("/inference/stop", status_code=status.HTTP_200_OK)
async def stop_inference() -> InferenceStatusResponse:
    if svc.process_running(svc.inference_process):
        svc.terminate_inference()
    return InferenceStatusResponse(status=svc.get_inference_state())


@router.get("/frames/{image_name}", status_code=status.HTTP_200_OK)
def get_frame(image_name: str) -> FileResponse:
    image_path = (svc.FRAMES_DIR / image_name).resolve()
    frames_root = svc.FRAMES_DIR.resolve()
    if frames_root not in image_path.parents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid frame path.")
    if not image_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Frame not found.")
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
                        lifecycle = svc.get_inference_state()
                        await websocket.send_json(
                            {"type": "status", "state": lifecycle.value}
                        )
                    await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return
