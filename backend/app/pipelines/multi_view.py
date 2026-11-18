"""Multi-View 3D Reconstruction Pipeline (Phase 7d).

Orchestrates:
1. N-view Incremental SfM via pycolmap (SIMPLE_RADIAL default, automatic OPENCV fallback,
   PINHOLE undistortion for downstream dense fusion).
2. Sequential sparse-aligned monocular depth + discontinuity & multi-view consistency filtering +
   Open3D ScalableTSDFVolume fusion with view-angle-weighted color blending.
3. Viewer-quality quadric decimation (~120k faces target -> scene.glb) while keeping full-res
   exports (scene_full.glb, scene.obj, mesh.ply, mesh.stl, point_cloud.ply).
4. Whole-pipeline peak RSS tracking, cooperative cancellation, hard timeout, and 3D camera frustum
   metadata for the interactive Three.js viewer.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil

from app.core.config import HoloGenSettings, get_settings
from app.core.logging import get_logger
from app.geometry.export import (
    export_mesh_glb,
    export_mesh_obj,
    export_mesh_ply,
    export_mesh_stl,
    save_point_cloud_ply,
    validate_exported_glb,
)
from app.geometry.mesh import MeshData
from app.geometry.multi_view_fusion import (
    MultiViewFusionConfig,
    MultiViewFusionResult,
    fuse_multiview_tsdf_and_mesh,
)
from app.geometry.point_cloud import PointCloudData
from app.geometry.refinement import clean_and_refine_mesh
from app.geometry.texture import TexturedMeshData
from app.core.exceptions import JobCancelledError, JobTimeoutError
from app.vision.depth_estimation import DepthModel, create_depth_model
from app.vision.sfm import SfmReconstructionResult, run_incremental_sfm

logger = get_logger("pipelines.multi_view")


@dataclass
class MultiViewResult:
    """Complete artifacts and telemetry from a multi-view reconstruction run."""

    project_id: str
    output_dir: Path
    sfm_result: SfmReconstructionResult
    fusion_result: MultiViewFusionResult
    viewer_mesh: MeshData
    full_mesh: MeshData
    point_cloud: PointCloudData
    camera_frustums: List[Dict[str, Any]]
    report_data: Dict[str, Any]
    whole_pipeline_peak_rss_mb: float
    total_runtime_ms: float


def _check_abort(cancel_event: Optional[threading.Event], deadline: Optional[float]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelledError("Reconstruction cancelled by user.")
    if deadline is not None and time.monotonic() > deadline:
        raise JobTimeoutError("Reconstruction exceeded hard job timeout.")


def _extract_camera_frustums_gltf(sfm_res: SfmReconstructionResult) -> List[Dict[str, Any]]:
    """Build camera frustum descriptors in glTF Y-up coordinates (+X, -Y, -Z from camera/OpenCV)."""
    frustums: List[Dict[str, Any]] = []
    # Flip Y and Z to convert OpenCV/COLMAP (+X right, +Y down, +Z forward) -> glTF (+X right, +Y up, -Z forward)
    flip_yz = np.diag([1.0, -1.0, -1.0])
    for name in sorted(sfm_res.registered_poses.keys()):
        pose = sfm_res.registered_poses[name]
        center_gltf = (flip_yz @ pose.camera_center_w).tolist()
        # Camera-to-world basis columns in glTF space
        R_c2w_cv = pose.R_w2c.T
        right_gltf = (flip_yz @ R_c2w_cv[:, 0]).tolist()
        down_gltf = (flip_yz @ R_c2w_cv[:, 1]).tolist()
        forward_gltf = (flip_yz @ R_c2w_cv[:, 2]).tolist()
        frustums.append(
            {
                "image_name": name,
                "center": [round(float(x), 5) for x in center_gltf],
                "right": [round(float(x), 5) for x in right_gltf],
                "up": [round(float(-x), 5) for x in down_gltf],
                "forward": [round(float(x), 5) for x in forward_gltf],
                "fov_x_deg": round(float(pose.intrinsics.fov_x_deg), 2),
                "fov_y_deg": round(float(pose.intrinsics.fov_y_deg), 2),
            }
        )
    return frustums


class MultiViewPipeline:
    """End-to-end multi-view reconstruction pipeline with sequential memory discipline."""

    def __init__(
        self,
        settings: Optional[HoloGenSettings] = None,
        depth_model: Optional[DepthModel] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.depth_model = depth_model

    def run(
        self,
        image_paths: List[Path],
        run_id: str,
        output_base_dir: Optional[Path] = None,
        sfm_max_image_size: int = 1600,
        depth_max_dim: int = 640,
        discontinuity_threshold: float = 0.08,
        consistency_rel_depth_thresh: float = 0.10,
        consistency_rgb_thresh: float = 0.18,
        viewer_target_faces: int = 120000,
        timeout_seconds: float = 600.0,
        cancel_event: Optional[threading.Event] = None,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
    ) -> MultiViewResult:
        t_start = time.perf_counter()
        deadline = time.monotonic() + float(timeout_seconds) if timeout_seconds > 0 else None

        proc = psutil.Process(os.getpid())
        baseline_rss = proc.memory_info().rss
        peak_rss = baseline_rss
        stop_evt = threading.Event()

        def _poll_rss() -> None:
            nonlocal peak_rss
            while not stop_evt.is_set():
                curr = proc.memory_info().rss
                if curr > peak_rss:
                    peak_rss = curr
                stop_evt.wait(0.02)

        sampler = threading.Thread(target=_poll_rss, daemon=True)
        sampler.start()

        out_root = output_base_dir or self.settings.outputs_dir
        out_dir = out_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        sfm_work_dir = out_dir / "_sfm_work"

        def _report(status: str, pct: int, msg: str) -> None:
            if progress_callback is not None:
                progress_callback(status, pct, msg)

        try:
            _check_abort(cancel_event, deadline)
            _report("running_sfm", 15, f"Running incremental SfM on {len(image_paths)} images...")
            sfm_res = run_incremental_sfm(
                image_paths=image_paths,
                sfm_max_image_size=sfm_max_image_size,
                camera_model="SIMPLE_RADIAL",
                fallback_camera_model="OPENCV",
                single_camera=True,
                max_num_features=4096,
                num_threads=4,
                undistort=True,
                undistorted_dir=sfm_work_dir / "undistorted",
                work_dir=sfm_work_dir,
            )

            _check_abort(cancel_event, deadline)
            _report(
                "running_depth_alignment",
                42,
                f"SfM registered {sfm_res.num_registered_images}/{sfm_res.num_input_images} views; aligning monocular depth...",
            )
            if self.depth_model is None:
                self.depth_model = create_depth_model("depth_anything_v2_small")

            fusion_cfg = MultiViewFusionConfig(
                depth_max_dim=depth_max_dim,
                discontinuity_threshold=discontinuity_threshold,
                consistency_rel_depth_thresh=consistency_rel_depth_thresh,
                consistency_rgb_thresh=consistency_rgb_thresh,
                min_consistent_views=1,
            )

            _report("running_fusion", 65, "Filtering consistency & fusing TSDF volume sequentially...")
            fusion_res = fuse_multiview_tsdf_and_mesh(sfm_res, self.depth_model, fusion_cfg)

            _check_abort(cancel_event, deadline)
            _report("running_export", 88, "Decimating viewer mesh & exporting 3D assets...")

            full_mesh = fusion_res.mesh
            if full_mesh.num_faces > viewer_target_faces:
                viewer_mesh = clean_and_refine_mesh(
                    full_mesh,
                    min_component_faces=0,
                    repair_holes=False,
                    taubin_smoothing_iterations=0,
                    target_faces=viewer_target_faces,
                )
            else:
                viewer_mesh = full_mesh

            def _as_textured(m: MeshData) -> TexturedMeshData:
                return TexturedMeshData(
                    mesh=m,
                    texture_image=np.zeros((2, 2, 3), dtype=np.uint8),
                    original_texture_size=(2, 2),
                    final_texture_size=(2, 2),
                    view_frustum_coverage_pct=100.0,
                    texturing_time_ms=0.0,
                )

            full_tex = _as_textured(full_mesh)
            viewer_tex = _as_textured(viewer_mesh)

            # Export full-resolution and viewer-preset assets
            export_mesh_glb(full_tex, out_dir / "scene_full.glb")
            export_mesh_glb(viewer_tex, out_dir / "scene.glb")
            glb_val = validate_exported_glb(out_dir / "scene.glb")
            export_mesh_obj(full_tex, out_dir / "scene.obj")
            export_mesh_ply(full_mesh, out_dir / "mesh.ply")
            export_mesh_stl(full_mesh, out_dir / "mesh.stl")
            save_point_cloud_ply(fusion_res.point_cloud, out_dir / "point_cloud.ply")

            camera_frustums = _extract_camera_frustums_gltf(sfm_res)
            total_ms = (time.perf_counter() - t_start) * 1000.0
            stop_evt.set()
            sampler.join(timeout=1.0)
            whole_peak_rss_mb = round(float(peak_rss) / (1024.0 * 1024.0), 2)
            whole_peak_delta_mb = round(float(peak_rss - baseline_rss) / (1024.0 * 1024.0), 2)

            report_data: Dict[str, Any] = {
                "project_id": run_id,
                "mode": "multi_view",
                "sfm": sfm_res.to_metadata_dict(),
                "fusion": fusion_res.to_metadata_dict(),
                "mesh_full": {"vertices": full_mesh.num_vertices, "faces": full_mesh.num_faces},
                "mesh_viewer": {"vertices": viewer_mesh.num_vertices, "faces": viewer_mesh.num_faces},
                "glb_validation": glb_val,
                "camera_frustums": camera_frustums,
                "whole_pipeline_peak_rss_mb": whole_peak_rss_mb,
                "whole_pipeline_peak_above_baseline_mb": whole_peak_delta_mb,
                "total_runtime_ms": round(total_ms, 2),
            }
            (out_dir / "reconstruction_report.json").write_text(
                json.dumps(report_data, indent=2), encoding="utf-8"
            )

            return MultiViewResult(
                project_id=run_id,
                output_dir=out_dir,
                sfm_result=sfm_res,
                fusion_result=fusion_res,
                viewer_mesh=viewer_mesh,
                full_mesh=full_mesh,
                point_cloud=fusion_res.point_cloud,
                camera_frustums=camera_frustums,
                report_data=report_data,
                whole_pipeline_peak_rss_mb=whole_peak_rss_mb,
                total_runtime_ms=total_ms,
            )
        finally:
            stop_evt.set()
            if sfm_work_dir.exists():
                shutil.rmtree(sfm_work_dir, ignore_errors=True)
