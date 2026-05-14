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


class StartInferenceRequest(BaseModel):
    video_filename: str = Field(
        ...,
        description="Name of a video file (obtained from GET /videos)",
    )
    stride: int = Field(
        default=30,
        ge=1,
        description="Run inference every N frames.")
    publish_every: int = Field(
        default=1,
        ge=1,
        description="Write one JSON event every N inferences.",
    )
    max_frames: int | None = Field(
        default=None,
        ge=1,
        description="Stop after M frames (optional).",
    )
    conf: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold.")
    iou: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="IoU threshold.",
    )
