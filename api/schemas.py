from enum import Enum

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class VideosResponse(BaseModel):
    filenames: list[str]


class InferenceState(str, Enum):
    running = "running"  # inference process is running
    idle = "idle"  # inference process completed naturally, or never started
    started = "started"  # inference process started
    stopped = "stopped"  # inference process terminated by user


class InferenceStatusResponse(BaseModel):
    status: InferenceState


# TODO: Make this align with CLI arguments and validation rules
class StartInferenceRequest(BaseModel):
    video_filename: str = Field(
        ...,
        description="Name of a video file (obtained from GET /videos)",
    )
    out: str = "parking_management_out.mp4"
    stride: int = 1
    publish_every: int = 1
    max_frames: int | None = None
    json_path: str = "bounding_boxes.json"
    weights: str = "yolo26n.pt"
    conf: float = 0.1
    iou: float = 0.7
    no_verbose: bool = False
