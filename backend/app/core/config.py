"""Pydantic configuration and hardware device auto-selection for HoloGen."""

import os
from pathlib import Path
from typing import Literal, Optional

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def detect_compute_device(preferred: str = "auto") -> str:
    """Detect optimal compute device with CUDA runtime verification and fallback."""
    if preferred == "cpu":
        return "cpu"
    if preferred in ("cuda", "auto"):
        if torch.cuda.is_available():
            try:
                # Verify that a small tensor kernel executes cleanly on this GPU architecture
                test_tensor = torch.zeros((2, 2), device="cuda")
                _ = test_tensor + 1.0
                del test_tensor
                return "cuda"
            except Exception:
                if preferred == "cuda":
                    raise
                return "cpu"
        elif preferred == "cuda":
            raise RuntimeError("CUDA explicitly requested, but torch.cuda.is_available() is False.")
    return "cpu"


class HoloGenSettings(BaseSettings):
    """Global settings for HoloGen pipelines and runtime environment."""

    model_config = SettingsConfigDict(
        env_prefix="HOLOGEN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    project_root: Path = Field(default=PROJECT_ROOT)
    models_dir: Path = Field(default=PROJECT_ROOT / "models" / "cache")
    outputs_dir: Path = Field(default=PROJECT_ROOT / "outputs")
    datasets_dir: Path = Field(default=PROJECT_ROOT / "datasets")

    # Compute & Memory Guards (tuned for 2GB VRAM MX330 & 7.7GB System RAM)
    device_preference: Literal["auto", "cuda", "cpu"] = Field(default="auto")
    use_fp16_on_cuda: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # Preprocessing defaults
    max_image_dimension: int = Field(default=640, ge=64, le=4096)
    blur_laplacian_threshold: float = Field(default=80.0, ge=0.0)
    underexposure_pixel_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    overexposure_pixel_threshold: float = Field(default=0.95, ge=0.0, le=1.0)

    # Camera Intrinsics defaults
    default_fov_deg: float = Field(default=60.0, gt=5.0, lt=175.0)
    default_sensor_width_mm: float = Field(default=36.0, gt=1.0)

    # Depth Estimation defaults
    default_depth_model: str = Field(default="depth_anything_v2_small")
    relative_near_depth: float = Field(default=1.0, gt=0.05)
    relative_far_depth: float = Field(default=10.0, gt=0.5)

    def resolve_device(self) -> str:
        """Return validated PyTorch device string ('cuda' or 'cpu')."""
        return detect_compute_device(self.device_preference)

    def configure_environment(self) -> None:
        """Ensure cache and output directories exist and set HF/Torch cache paths."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.models_dir / "huggingface"))
        os.environ.setdefault("TORCH_HOME", str(self.models_dir / "torch"))


def get_settings() -> HoloGenSettings:
    """Return initialized HoloGenSettings instance."""
    settings = HoloGenSettings()
    settings.configure_environment()
    return settings
