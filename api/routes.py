import asyncio
import json

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse

from api import services as svc
from api.schemas import (
    HealthResponse,
    InferenceState,
    InferenceStatusResponse,
    ParkingRegion,
    SessionsResponse,
    StartInferenceRequest,
    VideosResponse,
)

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


# NOTE: This assumes video files are stored under data/
@router.get("/videos", status_code=status.HTTP_200_OK, deprecated=True)
def list_videos() -> VideosResponse:
    """
    Legacy: video basenames directly under data/ (flat layout).

    Deprecated: use GET /sessions.
    """
    return VideosResponse(filenames=svc.list_data_videos())


@router.get("/sessions", status_code=status.HTTP_200_OK)
def list_sessions() -> SessionsResponse:
    """Session folder names under data/ with recording.mp4, bounding_boxes.json, and reference_frame.jpg."""
    return SessionsResponse(session_ids=svc.list_sessions())


@router.get(
    "/sessions/{session_id:path}/regions",
    status_code=status.HTTP_200_OK,
    response_model=list[ParkingRegion],
)
def get_session_regions(session_id: str) -> list[ParkingRegion]:
    """Parking region polygons for overlay on the session reference frame."""
    try:
        return svc.load_session_regions(session_id)
    except ValueError as e:
        detail = str(e)
        if detail.startswith("Invalid JSON"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail
            ) from e
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail) from e


@router.get("/sessions/{session_id:path}/reference-frame", status_code=status.HTTP_200_OK)
def get_session_reference_frame(session_id: str) -> FileResponse:
    """Still image aligned with bounding_boxes.json."""
    try:
        session = svc.resolve_session(session_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    return FileResponse(session.reference_frame_path)


@router.get("/sessions/{session_id:path}/video", status_code=status.HTTP_200_OK)
def get_session_video(session_id: str) -> FileResponse:
    """Session source video (recording.mp4)."""
    try:
        session = svc.resolve_session(session_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    return FileResponse(session.video_path)


@router.get("/sessions/{session_id:path}/frames/{image_name}", status_code=status.HTTP_200_OK)
def get_session_frame(session_id: str, image_name: str) -> FileResponse:
    """Inferred frame JPEG for a specific session run."""
    try:
        image_path = svc.resolve_session_frame_path(session_id, image_name)
    except ValueError as e:
        detail = str(e)
        if detail == "Invalid frame path.":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=detail) from e
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail) from e

    if not image_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frame not found.",
        )
    return FileResponse(image_path)


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
        if req.session_id:
            svc.resolve_session(req.session_id)
        else:
            assert req.video_filename is not None
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


@router.get("/frames/{image_name}", status_code=status.HTTP_200_OK, deprecated=True)
def get_frame(image_name: str) -> FileResponse:
    """Deprecated global frame endpoint. Use /sessions/{session_id}/frames/{image_name}."""
    image_path = (svc.FRAMES_DIR / image_name).resolve()
    frames_root = svc.FRAMES_DIR.resolve()
    if frames_root not in image_path.parents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid frame path.",
        )
    if not image_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frame not found.",
        )
    return FileResponse(image_path)


# TODO: This should do more than just send events, it should process events and send only
# what needs to be changed per sign/street
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
                        await websocket.send_json(
                            {"type": "warning", "message": "Malformed JSON event."}
                        )
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
