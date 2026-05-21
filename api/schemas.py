from enum import Enum

from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: str


class VideosResponse(BaseModel):
    """Legacy flat list of videos directly under data/."""

    filenames: list[str]


class SessionsResponse(BaseModel):
    """Complete session folders under data/."""

    session_ids: list[str] = Field(
        default_factory=list,
        description="Folder names with recording.mp4, bounding_boxes.json, and reference_frame.jpg.",
    )


class ParkingRegion(BaseModel):
    """One parking slot polygon (Ultralytics bounding_boxes.json format)."""

    points: list[list[int]]


class InferenceState(str, Enum):
    running = "running"  # inference process is running
    idle = "idle"  # inference process completed naturally, or never started
    started = "started"  # inference process started
    stopped = "stopped"  # inference process terminated by user


class InferenceStatusResponse(BaseModel):
    status: InferenceState


class StartInferenceRequest(BaseModel):
    video_filename: str | None = Field(
        default=None,
        description=(
            "Legacy: video basename under data/ (from GET /videos). "
            "Deprecated in favor of session_id when using data/<session_id>/ layouts."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description="Session folder under data/ (from GET /sessions).",
    )
    stride: int = Field(
        default=60,
        ge=1,
        description="Run inference every N frames.",
    )
    vote_radius: int = Field(
        default=2,
        ge=0,
        description=(
            "Majority-vote occupancy at each inference anchor f using frames f+-2, f+-4, ... "
            "(R=2 -> 5 frames: f-4, f-2, f, f+2, f+4). Set 0 to disable. Skipped if stride is too small "
            "for non-overlapping vote windows."
        ),
    )
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
        description="Detection confidence threshold.",
    )
    iou: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="IoU threshold.",
    )

    @model_validator(mode="after")
    def require_video_filename_or_session_id(self) -> "StartInferenceRequest":
        if not self.session_id and not self.video_filename:
            raise ValueError("Either session_id or video_filename is required.")
        return self
