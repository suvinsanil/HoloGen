"""Modular monocular depth estimation interface, model registry, and pretrained implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Type

import numpy as np
from PIL import Image
import psutil
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from app.core.config import get_settings
from app.core.exceptions import DepthEstimationError
from app.core.logging import get_logger

logger = get_logger("vision.depth_estimation")


def disparity_to_relative_depth(
    disparity: np.ndarray,
    near_depth: float = 1.0,
    far_depth: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw relative disparity (higher = closer) to positive relative depth Z (smaller = closer).

    Affine-invariant monocular models (Depth Anything V2, MiDaS) predict relative inverse depth
    (disparity) d ~ a * (1/Z) + b. To preserve 3D surface planarity under pinhole back-projection
    without claiming metric scale, we normalize disparity to [0, 1] (using robust 1st/99th percentiles
    to guard against single-pixel spikes) and map linearly in inverse-depth space [1/far, 1/near].

    Returns:
        (depth_map, normalized_disparity) both float32 of shape (H, W).
    """
    if near_depth <= 0.0 or far_depth <= near_depth:
        raise DepthEstimationError(
            f"Invalid relative depth bounds: near={near_depth}, far={far_depth}"
        )
    disp = np.asarray(disparity, dtype=np.float32)
    if not np.all(np.isfinite(disp)):
        disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)

    d_min = float(np.percentile(disp, 1.0))
    d_max = float(np.percentile(disp, 99.0))
    if d_max - d_min < 1e-6:
        d_min = float(np.min(disp))
        d_max = float(np.max(disp))

    if d_max - d_min < 1e-6:
        norm_disp = np.full_like(disp, 0.5, dtype=np.float32)
    else:
        norm_disp = np.clip((disp - d_min) / (d_max - d_min), 0.0, 1.0).astype(np.float32)

    inv_near = 1.0 / float(near_depth)
    inv_far = 1.0 / float(far_depth)
    inv_z = inv_far + norm_disp * (inv_near - inv_far)
    depth_map = (1.0 / np.maximum(inv_z, 1e-6)).astype(np.float32)
    return depth_map, norm_disp


def metric_depth_to_disparity_vis(metric_depth: np.ndarray) -> np.ndarray:
    """Compute normalized [0, 1] inverse-depth visualization map (higher = closer) from metric depth."""
    depth = np.maximum(np.asarray(metric_depth, dtype=np.float32), 1e-3)
    inv_d = 1.0 / depth
    d_min = float(np.percentile(inv_d, 1.0))
    d_max = float(np.percentile(inv_d, 99.0))
    if d_max - d_min < 1e-6:
        return np.full_like(depth, 0.5, dtype=np.float32)
    return np.clip((inv_d - d_min) / (d_max - d_min), 0.0, 1.0).astype(np.float32)


@dataclass
class DepthPrediction:
    """Structured result from a depth estimation model."""

    depth_map: np.ndarray  # float32 (H, W), positive Z > 0 (smaller = closer to camera)
    disparity_map: np.ndarray  # float32 (H, W) in [0, 1] (larger = closer to camera)
    raw_prediction: np.ndarray  # float32 (H, W), raw network output
    model_name: str
    model_id: str
    license: str
    device: str
    is_metric: bool
    scale_semantics: str
    inference_time_ms: float
    peak_memory_mb: float
    vram_allocated_mb: float

    @property
    def height(self) -> int:
        return int(self.depth_map.shape[0])

    @property
    def width(self) -> int:
        return int(self.depth_map.shape[1])

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable summary of the depth prediction."""
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "license": self.license,
            "device": self.device,
            "is_metric": self.is_metric,
            "scale_semantics": self.scale_semantics,
            "scale_disclaimer": (
                "Metric depth estimate (meters) from indoor-calibrated monocular network."
                if self.is_metric
                else "Monocular relative pseudo-depth; absolute metric scale is unknown."
            ),
            "resolution": {"width": self.width, "height": self.height},
            "inference_time_ms": round(self.inference_time_ms, 3),
            "peak_memory_mb": round(self.peak_memory_mb, 3),
            "vram_allocated_mb": round(self.vram_allocated_mb, 3),
            "depth_stats": {
                "min": round(float(np.min(self.depth_map)), 4),
                "max": round(float(np.max(self.depth_map)), 4),
                "mean": round(float(np.mean(self.depth_map)), 4),
                "median": round(float(np.median(self.depth_map)), 4),
                "std": round(float(np.std(self.depth_map)), 4),
            },
            "raw_prediction_stats": {
                "min": round(float(np.min(self.raw_prediction)), 4),
                "max": round(float(np.max(self.raw_prediction)), 4),
                "mean": round(float(np.mean(self.raw_prediction)), 4),
            },
        }


class DepthModel(ABC):
    """Abstract base interface for all monocular depth estimation models."""

    model_name: str
    model_id: str
    license: str
    is_metric: bool = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights and processor onto the target compute device."""

    @abstractmethod
    def predict(self, rgb_uint8: np.ndarray) -> DepthPrediction:
        """Run depth inference on an HxWx3 RGB uint8 image array."""

    def unload(self) -> None:
        """Release model weights and clear GPU cache to conserve VRAM on constrained hardware."""


class HuggingFaceDepthModelBase(DepthModel):
    """Shared Hugging Face AutoModelForDepthEstimation wrapper with OOM protection."""

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        near_depth: float = 1.0,
        far_depth: float = 10.0,
    ) -> None:
        settings = get_settings()
        self.device = device or settings.resolve_device()
        self.cache_dir = cache_dir or (settings.models_dir / "huggingface")
        self.near_depth = near_depth
        self.far_depth = far_depth
        self._processor: Optional[Any] = None
        self._model: Optional[Any] = None

    def load(self) -> None:
        """Load processor and model weights from local cache or Hugging Face Hub."""
        if self._model is not None and self._processor is not None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Loading %s (%s) on device=%s", self.model_name, self.model_id, self.device)
        try:
            self._processor = AutoImageProcessor.from_pretrained(
                self.model_id,
                cache_dir=str(self.cache_dir),
            )
            self._model = AutoModelForDepthEstimation.from_pretrained(
                self.model_id,
                cache_dir=str(self.cache_dir),
            )
            self._model.to(self.device)
            self._model.eval()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self.device == "cuda":
                logger.warning("CUDA OOM while loading %s; falling back to CPU.", self.model_name)
                torch.cuda.empty_cache()
                self.device = "cpu"
                self._model = AutoModelForDepthEstimation.from_pretrained(
                    self.model_id,
                    cache_dir=str(self.cache_dir),
                )
                self._model.to("cpu")
                self._model.eval()
            else:
                raise DepthEstimationError(f"Failed to load {self.model_id}: {exc}") from exc
        except Exception as exc:
            raise DepthEstimationError(f"Failed to load {self.model_id}: {exc}") from exc

    def unload(self) -> None:
        """Release model references and free GPU VRAM."""
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, rgb_uint8: np.ndarray) -> DepthPrediction:
        """Predict depth map for an HxWx3 RGB uint8 image."""
        if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
            raise DepthEstimationError(f"Expected HxWx3 RGB image, got shape {rgb_uint8.shape}")
        if self._model is None or self._processor is None:
            self.load()

        h, w = int(rgb_uint8.shape[0]), int(rgb_uint8.shape[1])
        pil_img = Image.fromarray(rgb_uint8, mode="RGB")

        process = psutil.Process()
        rss_before_mb = process.memory_info().rss / (1024.0 * 1024.0)

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        try:
            inputs = self._processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                predicted_depth = outputs.predicted_depth
                if predicted_depth.ndim == 2:
                    predicted_depth = predicted_depth.unsqueeze(0)
                prediction = F.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(1).squeeze(0)
                raw_pred = prediction.detach().cpu().numpy().astype(np.float32)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self.device == "cuda":
                logger.warning("CUDA OOM during inference on %s; retrying on CPU.", self.model_name)
                torch.cuda.empty_cache()
                self.device = "cpu"
                self._model.to("cpu")
                inputs = self._processor(images=pil_img, return_tensors="pt")
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    predicted_depth = outputs.predicted_depth
                    if predicted_depth.ndim == 2:
                        predicted_depth = predicted_depth.unsqueeze(0)
                    prediction = F.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=(h, w),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze(1).squeeze(0)
                    raw_pred = prediction.detach().cpu().numpy().astype(np.float32)
            else:
                raise DepthEstimationError(f"Inference failed on {self.model_name}: {exc}") from exc

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            vram_peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        else:
            vram_peak_mb = 0.0

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        rss_after_mb = process.memory_info().rss / (1024.0 * 1024.0)
        peak_mem_mb = max(vram_peak_mb, rss_after_mb, rss_before_mb)

        if self.is_metric:
            depth_map = np.clip(raw_pred, 0.05, 80.0).astype(np.float32)
            norm_disp = metric_depth_to_disparity_vis(depth_map)
            scale_semantics = "metric_meters"
        else:
            depth_map, norm_disp = disparity_to_relative_depth(
                raw_pred,
                near_depth=self.near_depth,
                far_depth=self.far_depth,
            )
            scale_semantics = "relative_unknown_scale"

        return DepthPrediction(
            depth_map=depth_map,
            disparity_map=norm_disp,
            raw_prediction=raw_pred,
            model_name=self.model_name,
            model_id=self.model_id,
            license=self.license,
            device=self.device,
            is_metric=self.is_metric,
            scale_semantics=scale_semantics,
            inference_time_ms=elapsed_ms,
            peak_memory_mb=peak_mem_mb,
            vram_allocated_mb=vram_peak_mb,
        )


class DepthAnythingV2SmallModel(HuggingFaceDepthModelBase):
    """Depth Anything V2 Small (24.8M params, Apache-2.0) relative depth model."""

    model_name = "depth_anything_v2_small"
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    license = "Apache-2.0"
    is_metric = False


class DPTHybridMiDaSModel(HuggingFaceDepthModelBase):
    """Intel DPT-Hybrid MiDaS (123M params, MIT) relative depth model."""

    model_name = "dpt_hybrid_midas"
    model_id = "Intel/dpt-hybrid-midas"
    license = "MIT"
    is_metric = False


class DepthAnythingV2MetricIndoorSmallModel(HuggingFaceDepthModelBase):
    """Depth Anything V2 Metric Indoor Small (24.8M params, Apache-2.0) metric depth model (meters)."""

    model_name = "depth_anything_v2_metric_indoor_small"
    model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    license = "Apache-2.0"
    is_metric = True


DEPTH_MODEL_REGISTRY: Dict[str, Type[HuggingFaceDepthModelBase]] = {
    DepthAnythingV2SmallModel.model_name: DepthAnythingV2SmallModel,
    DPTHybridMiDaSModel.model_name: DPTHybridMiDaSModel,
    DepthAnythingV2MetricIndoorSmallModel.model_name: DepthAnythingV2MetricIndoorSmallModel,
}


def list_available_depth_models() -> List[Dict[str, Any]]:
    """Return metadata for all registered depth estimation models."""
    return [
        {
            "model_name": cls.model_name,
            "model_id": cls.model_id,
            "license": cls.license,
            "is_metric": cls.is_metric,
        }
        for cls in DEPTH_MODEL_REGISTRY.values()
    ]


def create_depth_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    near_depth: Optional[float] = None,
    far_depth: Optional[float] = None,
) -> HuggingFaceDepthModelBase:
    """Instantiate a registered DepthModel by name."""
    settings = get_settings()
    name = (model_name or settings.default_depth_model).strip().lower()
    if name not in DEPTH_MODEL_REGISTRY:
        valid = ", ".join(sorted(DEPTH_MODEL_REGISTRY.keys()))
        raise DepthEstimationError(f"Unknown depth model '{name}'. Available models: {valid}")
    cls = DEPTH_MODEL_REGISTRY[name]
    return cls(
        device=device or settings.resolve_device(),
        cache_dir=cache_dir or (settings.models_dir / "huggingface"),
        near_depth=near_depth if near_depth is not None else settings.relative_near_depth,
        far_depth=far_depth if far_depth is not None else settings.relative_far_depth,
    )
