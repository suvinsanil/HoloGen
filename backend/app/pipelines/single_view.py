"""Full single-view 3D reconstruction pipeline (Phases 1-5.5)."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
import threading
from typing import Any, Callable, Dict, Literal, Optional, Union

from PIL import Image
import torch

from app.core.config import HoloGenSettings, get_settings
from app.core.exceptions import JobCancelledError, JobTimeoutError
from app.core.logging import get_logger
from app.geometry.export import (
    export_mesh_glb,
    export_mesh_obj,
    export_mesh_ply,
    export_mesh_stl,
    generate_markdown_reconstruction_report,
    save_depth_visualization_png,
    save_point_cloud_ply,
    validate_exported_glb,
)
from app.geometry.filtering import (
    compute_point_cloud_density_stats,
    normal_grazing_angle_filter,
    statistical_outlier_removal,
)
from app.geometry.mesh import (
    MeshData,
    reconstruct_mesh_alpha_shape,
    reconstruct_mesh_ball_pivoting,
    reconstruct_mesh_depth_grid,
    reconstruct_mesh_poisson,
)
from app.geometry.normals import estimate_and_orient_normals
from app.geometry.point_cloud import PointCloudData, backproject_depth_to_point_cloud
from app.geometry.refinement import analyze_mesh_topology, clean_and_refine_mesh
from app.geometry.texture import TexturedMeshData, apply_projective_texturing
from app.vision.camera import CameraIntrinsics, estimate_intrinsics
from app.vision.depth_estimation import DepthModel, DepthPrediction, create_depth_model
from app.vision.depth_refinement import DepthRefinementResult, refine_depth_map
from app.vision.preprocessing import PreprocessingResult, preprocess_image
from app.vision.uncertainty import UncertaintyProxyResult, compute_uncertainty_proxy

logger = get_logger("pipelines.single_view")


@dataclass
class SingleViewRunResult:
    """Structured result of a single-view reconstruction pipeline execution."""

    run_id: str
    output_dir: Path
    preprocessing: PreprocessingResult
    intrinsics: CameraIntrinsics
    depth_prediction: DepthPrediction
    depth_refinement: DepthRefinementResult
    uncertainty_proxy: UncertaintyProxyResult
    point_cloud: PointCloudData
    mesh: MeshData
    textured_mesh: TexturedMeshData
    depth_png_path: Path
    ply_path: Path
    glb_path: Path
    glb_full_path: Optional[Path]
    obj_path: Path
    stl_path: Path
    report_json_path: Path
    report_md_path: Path
    total_time_ms: float
    report_data: Dict[str, Any]


def _check_abort(cancel_event: Optional[threading.Event], deadline: Optional[float]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelledError("Reconstruction cancelled by user.")
    if deadline is not None and time.monotonic() > deadline:
        raise JobTimeoutError("Reconstruction exceeded hard job timeout.")


class SingleViewPipeline:
    """End-to-end single-view 3D reconstruction pipeline (Phases 1-5.5)."""

    def __init__(
        self,
        settings: Optional[HoloGenSettings] = None,
        depth_model: Optional[DepthModel] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.depth_model = depth_model or create_depth_model(
            model_name=model_name or self.settings.default_depth_model,
            device=self.settings.resolve_device(),
        )

    def run(
        self,
        image_input: Union[str, Path, Image.Image],
        run_id: Optional[str] = None,
        max_dimension: Optional[int] = None,
        fov_deg: Optional[float] = None,
        focal_length_px: Optional[float] = None,
        refinement_method: Optional[Literal["none", "bilateral", "guided", "guided_bilateral"]] = None,
        mesh_method: Literal["depth_grid", "poisson", "ball_pivoting", "alpha_shape"] = "depth_grid",
        quality_preset: Literal["viewer", "full"] = "viewer",
        viewer_target_faces: int = 120000,
        export_both_presets: bool = True,
        discontinuity_threshold: float = 0.08,
        uncertainty_threshold: float = 0.75,
        smoothing_iterations: int = 4,
        target_faces: Optional[int] = None,
        texture_max_dimension: Optional[int] = None,
        output_base_dir: Optional[Path] = None,
        timeout_seconds: float = 300.0,
        cancel_event: Optional[threading.Event] = None,
    ) -> SingleViewRunResult:
        """Execute full single-view 3D reconstruction and write all assets + reports."""
        t_start = time.perf_counter()
        deadline = time.monotonic() + float(timeout_seconds) if timeout_seconds > 0 else None
        _check_abort(cancel_event, deadline)

        if run_id is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            stem = Path(image_input).stem if isinstance(image_input, (str, Path)) else "image"
            run_id = f"{stem}_{ts}"

        ref_method = refinement_method or self.settings.default_refinement_method
        base_dir = Path(output_base_dir) if output_base_dir else self.settings.outputs_dir
        out_dir = base_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting SingleViewPipeline run_id=%s (refinement=%s, preset=%s)",
            run_id,
            ref_method,
            quality_preset,
        )

        # 1. Preprocessing & Quality Checks
        prep = preprocess_image(
            image_input=image_input,
            max_dimension=max_dimension or self.settings.max_image_dimension,
            blur_threshold=self.settings.blur_laplacian_threshold,
            underexposure_threshold=self.settings.underexposure_pixel_threshold,
            overexposure_threshold=self.settings.overexposure_pixel_threshold,
        )
        preprocessed_png_path = out_dir / "input_preprocessed.png"
        Image.fromarray(prep.rgb_uint8, mode="RGB").save(preprocessed_png_path)
        _check_abort(cancel_event, deadline)

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
        _check_abort(cancel_event, deadline)

        # 4. Edge-Preserving Depth Refinement (defaults to "none" per Experiment A findings)
        depth_ref = refine_depth_map(
            depth_map=depth_pred.depth_map,
            rgb_uint8=prep.rgb_uint8,
            method=ref_method,
            discontinuity_threshold=discontinuity_threshold,
        )

        # 5. Monocular Uncertainty Proxy
        uncertainty = compute_uncertainty_proxy(
            depth_map=depth_ref.refined_depth,
            uncertainty_threshold=uncertainty_threshold,
        )
        _check_abort(cancel_event, deadline)

        # 6. Back-Projection & Point Cloud Cleaning + Oriented Normals
        raw_cloud = backproject_depth_to_point_cloud(
            depth_map=depth_ref.refined_depth,
            rgb_image=prep.rgb_uint8,
            intrinsics=intrinsics,
        )
        t_clean0 = time.perf_counter()
        cloud_with_normals = estimate_and_orient_normals(raw_cloud, max_nn=30)
        sor_res = statistical_outlier_removal(cloud_with_normals, nb_neighbors=20, std_ratio=2.0)
        graz_res = normal_grazing_angle_filter(
            sor_res.cloud, max_grazing_angle_deg=82.0
        )
        clean_cloud = graz_res.cloud
        density_stats = compute_point_cloud_density_stats(clean_cloud)
        clean_time_ms = (time.perf_counter() - t_clean0) * 1000.0
        _check_abort(cancel_event, deadline)

        # 7. Surface Mesh Reconstruction & Cleanup (Full Resolution + Viewer Decimation Preset)
        t_mesh0 = time.perf_counter()
        if mesh_method == "depth_grid":
            raw_mesh = reconstruct_mesh_depth_grid(
                depth_map=depth_ref.refined_depth,
                rgb_image=prep.rgb_uint8,
                intrinsics=intrinsics,
                relative_depth_jump_threshold=discontinuity_threshold,
                stride=1,
            )
            full_mesh = clean_and_refine_mesh(
                raw_mesh,
                min_component_faces=40,
                repair_holes=True,
                max_hole_loop_edges=36,
                taubin_smoothing_iterations=smoothing_iterations,
                target_faces=None,
            )
        elif mesh_method == "poisson":
            raw_mesh = reconstruct_mesh_poisson(
                clean_cloud, octree_depth=8, density_quantile_trim=0.08
            )
            full_mesh = clean_and_refine_mesh(
                raw_mesh,
                min_component_faces=40,
                repair_holes=False,
                taubin_smoothing_iterations=max(0, smoothing_iterations // 2),
                target_faces=None,
            )
        elif mesh_method == "ball_pivoting":
            raw_mesh = reconstruct_mesh_ball_pivoting(clean_cloud)
            full_mesh = clean_and_refine_mesh(
                raw_mesh,
                min_component_faces=20,
                repair_holes=True,
                taubin_smoothing_iterations=smoothing_iterations,
                target_faces=None,
            )
        elif mesh_method == "alpha_shape":
            raw_mesh = reconstruct_mesh_alpha_shape(clean_cloud)
            full_mesh = clean_and_refine_mesh(
                raw_mesh,
                min_component_faces=10,
                repair_holes=False,
                taubin_smoothing_iterations=smoothing_iterations,
                target_faces=None,
            )
        else:
            raise ValueError(f"Unsupported mesh_method: {mesh_method}")

        effective_target_faces = target_faces
        if effective_target_faces is None and quality_preset == "viewer":
            effective_target_faces = int(viewer_target_faces)

        if effective_target_faces is not None and full_mesh.num_faces > effective_target_faces:
            active_mesh = clean_and_refine_mesh(
                full_mesh,
                min_component_faces=20,
                repair_holes=False,
                taubin_smoothing_iterations=1,
                target_faces=effective_target_faces,
            )
            active_mesh.method = f"{mesh_method}_viewer_decimated"
        else:
            active_mesh = full_mesh

        topo = analyze_mesh_topology(active_mesh)
        meshing_total_ms = (time.perf_counter() - t_mesh0) * 1000.0
        _check_abort(cancel_event, deadline)

        # 8. Projective UV Mapping & Texture Assignment
        textured = apply_projective_texturing(
            mesh=active_mesh,
            rgb_uint8=prep.rgb_uint8,
            intrinsics=intrinsics,
            texture_max_dimension=texture_max_dimension,
        )
        _check_abort(cancel_event, deadline)

        # 9. Multi-Format Asset Export & Independent GLB Validation
        t_export_start = time.perf_counter()
        depth_png_path = save_depth_visualization_png(
            depth_pred.disparity_map,
            out_dir / "depth_visualization.png",
            mode="heatmap",
        )
        uncertainty_png_path = save_depth_visualization_png(
            uncertainty.uncertainty_proxy_map,
            out_dir / "uncertainty_proxy.png",
            mode="heatmap",
        )
        ply_path = save_point_cloud_ply(
            clean_cloud,
            out_dir / "point_cloud.ply",
        )
        mesh_ply_path = export_mesh_ply(
            textured.mesh,
            out_dir / "mesh.ply",
        )
        glb_path = export_mesh_glb(
            textured,
            out_dir / "scene.glb",
        )

        glb_full_path: Optional[Path] = None
        if export_both_presets and full_mesh.num_faces != active_mesh.num_faces:
            textured_full = apply_projective_texturing(
                mesh=full_mesh,
                rgb_uint8=prep.rgb_uint8,
                intrinsics=intrinsics,
                texture_max_dimension=texture_max_dimension,
            )
            glb_full_path = export_mesh_glb(
                textured_full,
                out_dir / "scene_full.glb",
            )
        else:
            glb_full_path = glb_path

        obj_path = export_mesh_obj(
            textured,
            out_dir / "scene.obj",
        )
        stl_path = export_mesh_stl(
            textured.mesh,
            out_dir / "mesh.stl",
        )
        glb_validation = validate_exported_glb(glb_path)
        export_time_ms = (time.perf_counter() - t_export_start) * 1000.0
        total_time_ms = (time.perf_counter() - t_start) * 1000.0

        gpu_name = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "None (CPU only)"
        )

        pc_meta = clean_cloud.to_metadata_dict()
        pc_meta["raw_points"] = raw_cloud.num_points
        pc_meta["retention_ratio"] = round(
            float(clean_cloud.num_points) / float(raw_cloud.num_points), 4
        )
        pc_meta["density_stats"] = density_stats.to_dict()

        mesh_meta = textured.mesh.to_metadata_dict()
        mesh_meta["quality_preset"] = quality_preset
        mesh_meta["full_res_vertices"] = full_mesh.num_vertices
        mesh_meta["full_res_faces"] = full_mesh.num_faces
        mesh_meta["topology"] = topo.to_dict()

        export_availability: Dict[str, Any] = {
            "glb": {
                "filename": glb_path.name,
                "preset": quality_preset,
                "available": glb_path.exists(),
                "size_bytes": int(glb_path.stat().st_size),
                "size_mb": round(glb_path.stat().st_size / (1024.0 * 1024.0), 3),
                "validation": glb_validation,
            },
            "glb_full": {
                "filename": glb_full_path.name if glb_full_path else glb_path.name,
                "preset": "full",
                "available": bool(glb_full_path and glb_full_path.exists()),
                "size_bytes": int(glb_full_path.stat().st_size) if glb_full_path else 0,
                "size_mb": round(glb_full_path.stat().st_size / (1024.0 * 1024.0), 3)
                if glb_full_path
                else 0.0,
            },
            "obj": {
                "filename": obj_path.name,
                "available": obj_path.exists(),
                "size_bytes": int(obj_path.stat().st_size),
            },
            "point_cloud_ply": {
                "filename": ply_path.name,
                "available": ply_path.exists(),
                "size_bytes": int(ply_path.stat().st_size),
            },
            "mesh_ply": {
                "filename": mesh_ply_path.name,
                "available": mesh_ply_path.exists(),
                "size_bytes": int(mesh_ply_path.stat().st_size),
            },
            "stl": {
                "filename": stl_path.name,
                "available": stl_path.exists(),
                "size_bytes": int(stl_path.stat().st_size),
            },
            "depth_png": {
                "filename": depth_png_path.name,
                "available": depth_png_path.exists(),
                "size_bytes": int(depth_png_path.stat().st_size),
            },
            "uncertainty_proxy_png": {
                "filename": uncertainty_png_path.name,
                "available": uncertainty_png_path.exists(),
                "size_bytes": int(uncertainty_png_path.stat().st_size),
            },
        }

        report_data: Dict[str, Any] = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline_stage": "phase_5.5_hardened_single_view",
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
            "depth_refinement": depth_ref.to_metadata_dict(),
            "uncertainty_proxy": uncertainty.to_metadata_dict(),
            "point_cloud": pc_meta,
            "mesh": mesh_meta,
            "texturing": textured.to_metadata_dict(),
            "estimated_coverage": {
                "view_frustum_coverage_pct": round(textured.view_frustum_coverage_pct, 2),
                "cleaned_point_retention_pct": round(
                    float(clean_cloud.num_points) * 100.0 / float(raw_cloud.num_points), 2
                ),
                "definition": (
                    "View-Frustum Coverage is the percentage of 2D input camera image pixels "
                    "(W x H) covered when rasterizing all valid reconstructed mesh triangles "
                    "in projective UV space. NOTE: It measures 2D view-frustum projection "
                    "occupancy only and does NOT measure 3D scene completeness."
                ),
            },
            "timings_ms": {
                "preprocessing_ms": round(prep.processing_time_ms, 3),
                "depth_inference_ms": round(depth_pred.inference_time_ms, 3),
                "depth_refinement_ms": round(depth_ref.processing_time_ms, 3),
                "uncertainty_proxy_ms": round(uncertainty.processing_time_ms, 3),
                "backprojection_ms": round(raw_cloud.generation_time_ms, 3),
                "point_cloud_cleaning_ms": round(clean_time_ms, 3),
                "meshing_ms": round(meshing_total_ms, 3),
                "texturing_ms": round(textured.texturing_time_ms, 3),
                "export_ms": round(export_time_ms, 3),
                "total_pipeline_ms": round(total_time_ms, 3),
            },
            "export_availability": export_availability,
            "artifacts": {
                "preprocessed_image": str(preprocessed_png_path.name),
                "depth_visualization_png": str(depth_png_path.name),
                "uncertainty_proxy_png": str(uncertainty_png_path.name),
                "point_cloud_ply": str(ply_path.name),
                "mesh_ply": str(mesh_ply_path.name),
                "scene_glb": str(glb_path.name),
                "scene_full_glb": str(glb_full_path.name) if glb_full_path else str(glb_path.name),
                "scene_obj": str(obj_path.name),
                "mesh_stl": str(stl_path.name),
                "ply_file_size_bytes": int(ply_path.stat().st_size),
                "glb_file_size_bytes": int(glb_path.stat().st_size),
                "glb_full_file_size_bytes": int(glb_full_path.stat().st_size)
                if glb_full_path
                else int(glb_path.stat().st_size),
            },
        }

        report_json_path = out_dir / "run_report.json"
        report_json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
        (out_dir / "reconstruction_report.json").write_text(
            json.dumps(report_data, indent=2), encoding="utf-8"
        )
        md_text = generate_markdown_reconstruction_report(report_data)
        report_md_path = out_dir / "reconstruction_report.md"
        report_md_path.write_text(md_text, encoding="utf-8")

        logger.info(
            "Completed SingleViewPipeline run_id=%s in %.1f ms (Viewer Faces=%d [%.2f MB], Full Faces=%d [%.2f MB])",
            run_id,
            total_time_ms,
            textured.mesh.num_faces,
            glb_path.stat().st_size / (1024.0 * 1024.0),
            full_mesh.num_faces,
            (glb_full_path.stat().st_size if glb_full_path else glb_path.stat().st_size)
            / (1024.0 * 1024.0),
        )

        return SingleViewRunResult(
            run_id=run_id,
            output_dir=out_dir,
            preprocessing=prep,
            intrinsics=intrinsics,
            depth_prediction=depth_pred,
            depth_refinement=depth_ref,
            uncertainty_proxy=uncertainty,
            point_cloud=clean_cloud,
            mesh=textured.mesh,
            textured_mesh=textured,
            depth_png_path=depth_png_path,
            ply_path=ply_path,
            glb_path=glb_path,
            glb_full_path=glb_full_path,
            obj_path=obj_path,
            stl_path=stl_path,
            report_json_path=report_json_path,
            report_md_path=report_md_path,
            total_time_ms=total_time_ms,
            report_data=report_data,
        )


SingleViewBaselinePipeline = SingleViewPipeline
