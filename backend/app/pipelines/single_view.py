"""Single-view 3D reconstruction pipeline."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
from typing import Any, Dict, Optional, Union

from PIL import Image
import torch

from app.core.config import HoloGenSettings, get_settings
from app.core.logging import get_logger
from app.geometry.export import save_depth_visualization_png, save_point_cloud_ply
from app.geometry.point_cloud import PointCloudData, backproject_depth_to_point_cloud
from app.vision.camera import CameraIntrinsics, estimate_intrinsics
from app.vision.depth_estimation import DepthAnythingV2SmallModel, DepthModel, DepthPrediction
from app.vision.preprocessing import PreprocessingResult, preprocess_image

logger = get_logger("pipelines.single_view")


@dataclass
class SingleViewRunResult:
    """Structured result of a single-view reconstruction pipeline execution."""

    run_id: str
    output_dir: Path
    preprocessing: PreprocessingResult
    intrinsics: CameraIntrinsics
    depth_prediction: DepthPrediction
    point_cloud: PointCloudData
    depth_png_path: Path
    ply_path: Path
    report_json_path: Path
    total_time_ms: float
    report_data: Dict[str, Any]


class SingleViewBaselinePipeline:
    """Phase 1 single-view reconstruction pipeline (image -> depth -> colored point cloud -> PLY)."""

    def __init__(
        self,
        settings: Optional[HoloGenSettings] = None,
        depth_model: Optional[DepthModel] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.depth_model = depth_model or DepthAnythingV2SmallModel(
            device=self.settings.resolve_device(),
            cache_dir=self.settings.models_dir / "huggingface",
            near_depth=self.settings.relative_near_depth,
            far_depth=self.settings.relative_far_depth,
        )

    def run(
        self,
        image_input: Union[str, Path, Image.Image],
        run_id: Optional[str] = None,
        max_dimension: Optional[int] = None,
        fov_deg: Optional[float] = None,
        focal_length_px: Optional[float] = None,
        output_base_dir: Optional[Path] = None,
    ) -> SingleViewRunResult:
        """Execute single-view reconstruction and write artifacts to outputs/<run_id>/."""
        t_start = time.perf_counter()
        if run_id is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            stem = Path(image_input).stem if isinstance(image_input, (str, Path)) else "image"
            run_id = f"{stem}_{ts}"

        base_dir = Path(output_base_dir) if output_base_dir else self.settings.outputs_dir
        out_dir = base_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting SingleViewBaselinePipeline run_id=%s", run_id)

        # 1. Preprocessing & Quality Checks
        prep = preprocess_image(
            image_input=image_input,
            max_dimension=max_dimension or self.settings.max_image_dimension,
            blur_threshold=self.settings.blur_laplacian_threshold,
            underexposure_threshold=self.settings.underexposure_pixel_threshold,
            overexposure_threshold=self.settings.overexposure_pixel_threshold,
        )

        # Save preprocessed RGB reference
        preprocessed_png_path = out_dir / "input_preprocessed.png"
        Image.fromarray(prep.rgb_uint8, mode="RGB").save(preprocessed_png_path)

        # 2. Camera Intrinsics Estimation
        intrinsics = estimate_intrinsics(
            width=prep.width,
            height=prep.height,
            exif_metadata=prep.exif_metadata,
            fov_deg=fov_deg or self.settings.default_fov_deg,
            focal_length_px=focal_length_px,
            sensor_width_mm=self.settings.default_sensor_width_mm,
        )

        # 3. Monocular Depth Estimation
        self.depth_model.load()
        depth_pred = self.depth_model.predict(prep.rgb_uint8)

        # 4. Back-projection to Colored 3D Point Cloud
        point_cloud = backproject_depth_to_point_cloud(
            depth_map=depth_pred.depth_map,
            rgb_image=prep.rgb_uint8,
            intrinsics=intrinsics,
        )

        # 5. Export artifacts (Depth visualization PNG & PLY point cloud)
        t_export_start = time.perf_counter()
        depth_png_path = save_depth_visualization_png(
            depth_pred.disparity_map,
            out_dir / "depth_visualization.png",
            mode="heatmap",
        )
        ply_path = save_point_cloud_ply(
            point_cloud,
            out_dir / "point_cloud.ply",
        )
        export_time_ms = (time.perf_counter() - t_export_start) * 1000.0
        total_time_ms = (time.perf_counter() - t_start) * 1000.0

        gpu_name = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "None (CPU only)"
        )

        report_data: Dict[str, Any] = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline_stage": "phase_1_baseline",
            "hardware": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "compute_device": depth_pred.device,
                "gpu_name": gpu_name,
            },
            "preprocessing": prep.to_dict(),
            "camera_intrinsics": intrinsics.to_dict(),
            "depth_estimation": depth_pred.to_metadata_dict(),
            "point_cloud": point_cloud.to_metadata_dict(),
            "timings_ms": {
                "preprocessing_ms": round(prep.processing_time_ms, 3),
                "depth_inference_ms": round(depth_pred.inference_time_ms, 3),
                "backprojection_ms": round(point_cloud.generation_time_ms, 3),
                "export_ms": round(export_time_ms, 3),
                "total_pipeline_ms": round(total_time_ms, 3),
            },
            "artifacts": {
                "preprocessed_image": str(preprocessed_png_path.name),
                "depth_visualization_png": str(depth_png_path.name),
                "point_cloud_ply": str(ply_path.name),
                "ply_file_size_bytes": int(ply_path.stat().st_size),
            },
        }

        report_json_path = out_dir / "run_report.json"
        report_json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
        logger.info(
            "Completed run_id=%s in %.1f ms (%d points, PLY=%d bytes)",
            run_id,
            total_time_ms,
            point_cloud.num_points,
            ply_path.stat().st_size,
        )

        return SingleViewRunResult(
            run_id=run_id,
            output_dir=out_dir,
            preprocessing=prep,
            intrinsics=intrinsics,
            depth_prediction=depth_pred,
            point_cloud=point_cloud,
            depth_png_path=depth_png_path,
            ply_path=ply_path,
            report_json_path=report_json_path,
            total_time_ms=total_time_ms,
            report_data=report_data,
        )
