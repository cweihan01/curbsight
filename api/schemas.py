from enum import Enum

from pydantic import BaseModel, Field, model_validator

from constants import (
    DEFAULT_CONF,
    DEFAULT_PUBLISH_EVERY,
    DEFAULT_STRIDE,
    DEFAULT_VOTE_FRAME_STEP,
    DEFAULT_VOTE_RADIUS,
)


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

    @model_validator(mode="after")
    def exactly_four_points(self) -> "ParkingRegion":
        if len(self.points) != 4:
            raise ValueError("Each bounding box must have exactly 4 points.")
        return self


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
    regions: list[ParkingRegion] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Selected parking slot polygons inline (bounding_boxes.json format). "
            "Each region has exactly 4 points. Overrides session/file regions when set."
        ),
    )
    stride: int = Field(
        default=DEFAULT_STRIDE,
        ge=1,
        description="Run inference every N frames.",
    )
    vote_radius: int = Field(
        default=DEFAULT_VOTE_RADIUS,
        ge=0,
        description=(
            "Majority-vote occupancy at each inference anchor f using 2*R+1 frames spaced "
            "vote_frame_step apart. Set 0 to disable. Skipped if stride is too small "
            "for non-overlapping vote windows (needs stride > 2*R*vote_frame_step)."
        ),
    )
    vote_frame_step: int = Field(
        default=DEFAULT_VOTE_FRAME_STEP,
        ge=1,
        description=(
            "Spacing in frames between samples in a vote window. "
            "With R=3, step=15: anchor f samples f-45, f-30, f-15, f, f+15, f+30, f+45."
        ),
    )
    publish_every: int = Field(
        default=DEFAULT_PUBLISH_EVERY,
        ge=1,
        description="Write one JSON event every N inferences.",
    )
    max_frames: int | None = Field(
        default=None,
        ge=1,
        description="Stop after M frames (optional).",
    )
    conf: float = Field(
        default=DEFAULT_CONF,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold.",
    )

    @model_validator(mode="after")
    def require_video_filename_or_session_id(self) -> "StartInferenceRequest":
        if not self.session_id and not self.video_filename:
            raise ValueError("Either session_id or video_filename is required.")
        return self
