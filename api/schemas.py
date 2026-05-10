from pydantic import BaseModel, Field


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
