"""Request and response Pydantic schemas for the HoloGen REST API."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ReconstructRequest(BaseModel):
    """Parameters for triggering a single-view or multi-view 3D reconstruction job."""

    mode: Literal["single_view", "multi_view"] = Field(default="single_view")
    model_name: Literal[
        "depth_anything_v2_small",
        "dpt_hybrid_midas",
        "depth_anything_v2_metric_indoor_small",
    ] = Field(default="depth_anything_v2_small")
    max_dimension: int = Field(default=640, ge=128, le=1024)
    fov_deg: float = Field(default=60.0, gt=10.0, lt=160.0)
    refinement_method: Literal["none", "bilateral", "guided", "guided_bilateral"] = Field(
        default="none"
    )
    mesh_method: Literal["depth_grid", "poisson", "ball_pivoting", "alpha_shape"] = Field(
        default="depth_grid"
    )
    quality_preset: Literal["viewer", "full"] = Field(default="viewer")
    viewer_target_faces: int = Field(default=120000, ge=10000, le=300000)
    smoothing_iterations: int = Field(default=4, ge=0, le=20)
    uncertainty_threshold: float = Field(default=0.75, ge=0.1, le=1.0)
    discontinuity_threshold: float = Field(default=0.08, ge=0.02, le=0.5)
    consistency_rel_depth_thresh: float = Field(default=0.10, ge=0.02, le=0.5)
    consistency_rgb_thresh: float = Field(default=0.18, ge=0.02, le=0.8)
    timeout_seconds: float = Field(default=600.0, ge=5.0, le=3600.0)
    run_synchronously: bool = Field(
        default=False,
        description="If true, waits for the concurrency-1 worker to finish before returning.",
    )


class ProjectStatusResponse(BaseModel):
    """Live status and metadata for a reconstruction project."""

    project_id: str
    name: str
    mode: str
    status: str
    progress_pct: int
    stage_message: str
    queue_position: int = 0
    created_at: str
    updated_at: str
    error_message: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    report: Optional[Dict[str, Any]] = None
    available_exports: List[str] = Field(default_factory=list)


class ProjectListResponse(BaseModel):
    """List of reconstruction projects."""

    projects: List[ProjectStatusResponse]


class HealthResponse(BaseModel):
    """Backend health, hardware, and singleton model status."""

    status: str
    device: str
    gpu_name: str
    loaded_model: Optional[str]
    queue_length: int
    active_job_id: Optional[str]
    max_upload_mb: float
    max_image_pixels: str
