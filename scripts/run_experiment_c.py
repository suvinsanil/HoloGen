"""Experiment C Runner: Point Cloud Processing & Filtering Ablation Study.

Evaluates 6 point cloud processing variants on real sample reconstructions:
1. raw (unfiltered back-projected cloud with oriented normals)
2. voxel_downsampled (voxel_size=0.03)
3. statistical_outlier_removal (nb_neighbors=20, std_ratio=2.0)
4. radius_outlier_removal (nb_points=12, radius=3.5*mean_nn)
5. normal_grazing_filtered (SOR + grazing normal filter at 80 deg)
6. full_clean_smoothed (SOR + grazing filter + tangent-plane normal smoothing)
"""

import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.core.logging import get_logger
from app.geometry.export import save_point_cloud_ply
from app.geometry.filtering import (
    compute_point_cloud_density_stats,
    normal_grazing_angle_filter,
    radius_outlier_removal,
    smooth_point_cloud_tangent_plane,
    statistical_outlier_removal,
    voxel_downsample_cloud,
)
from app.geometry.normals import estimate_and_orient_normals
from app.geometry.point_cloud import PointCloudData, backproject_depth_to_point_cloud
from app.vision.camera import estimate_intrinsics
from app.vision.depth_estimation import create_depth_model
from app.vision.depth_refinement import refine_depth_map
from app.vision.preprocessing import preprocess_image

logger = get_logger("scripts.run_experiment_c")


def render_point_cloud_orthographic_preview(
    cloud: PointCloudData,
    output_path: Path,
    canvas_size: int = 480,
) -> None:
    """Render side-by-side Front (X-Y) and Side/Oblique (Z-Y) views of a colored point cloud to PNG."""
    canvas = np.full((canvas_size, canvas_size * 2, 3), 24, dtype=np.uint8)
    if cloud.num_points == 0:
        cv2.imwrite(str(output_path), canvas)
        return

    pts = cloud.points
    cols_bgr = np.clip(cloud.colors[:, ::-1] * 255.0, 0, 255).astype(np.uint8)

    # Sort back-to-front by Z for painter's algorithm in Front view
    order_front = np.argsort(-pts[:, 2])
    p_front = pts[order_front]
    c_front = cols_bgr[order_front]

    min_b = np.percentile(pts, 1.0, axis=0)
    max_b = np.percentile(pts, 99.0, axis=0)
    span = np.maximum(max_b - min_b, 1e-4)

    margin = 24
    usable = canvas_size - 2 * margin

    # Panel 1: Front View (X vs Y)
    u_f = margin + ((p_front[:, 0] - min_b[0]) / span[0] * usable).astype(np.int32)
    v_f = margin + ((p_front[:, 1] - min_b[1]) / span[1] * usable).astype(np.int32)
    valid_f = (u_f >= 0) & (u_f < canvas_size) & (v_f >= 0) & (v_f < canvas_size)
    canvas[v_f[valid_f], u_f[valid_f]] = c_front[valid_f]

    # Panel 2: Side Depth Profile View (Z vs Y) — reveals flying pixels and depth noise!
    u_s = canvas_size + margin + ((p_front[:, 2] - min_b[2]) / span[2] * usable).astype(np.int32)
    v_s = margin + ((p_front[:, 1] - min_b[1]) / span[1] * usable).astype(np.int32)
    valid_s = (
        (u_s >= canvas_size)
        & (u_s < canvas_size * 2)
        & (v_s >= 0)
        & (v_s < canvas_size)
    )
    canvas[v_s[valid_s], u_s[valid_s]] = c_front[valid_s]

    cv2.putText(
        canvas,
        f"Front (XY) | N={cloud.num_points}",
        (14, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Side Depth Profile (ZY)",
        (canvas_size + 14, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def get_git_commit_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        )
        return out.strip()
    except Exception:
        return "unknown"


def main() -> int:
    settings = get_settings()
    device = settings.resolve_device()
    commit_hash = get_git_commit_short()

    sample_path = settings.datasets_dir / "samples" / "sample_02_aloe_indoor.jpg"
    prep = preprocess_image(sample_path, max_dimension=512)
    intrinsics = estimate_intrinsics(prep.width, prep.height, exif_metadata=prep.exif_metadata)

    model = create_depth_model("depth_anything_v2_small", device=device)
    model.load()
    pred = model.predict(prep.rgb_uint8)
    refined = refine_depth_map(pred.depth_map, prep.rgb_uint8, method="bilateral")

    raw_cloud = backproject_depth_to_point_cloud(
        depth_map=refined.refined_depth,
        rgb_image=prep.rgb_uint8,
        intrinsics=intrinsics,
    )
    raw_with_normals = estimate_and_orient_normals(raw_cloud, max_nn=30)

    exp_dir = settings.outputs_dir / "experiment_c"
    exp_dir.mkdir(parents=True, exist_ok=True)

    variants = []

    # 1. Raw with oriented normals
    variants.append(
        (
            "1_raw_backprojected",
            raw_with_normals,
            raw_with_normals.generation_time_ms,
            1.0,
            {"description": "Unfiltered back-projected cloud with PCA camera-oriented normals"},
        )
    )

    # 2. Voxel downsampled
    v_res = voxel_downsample_cloud(raw_with_normals, voxel_size=0.035)
    variants.append(
        (
            "2_voxel_downsampled",
            v_res.cloud,
            v_res.processing_time_ms,
            v_res.retention_ratio,
            v_res.parameters,
        )
    )

    # 3. Statistical Outlier Removal (SOR)
    sor_res = statistical_outlier_removal(raw_with_normals, nb_neighbors=20, std_ratio=2.0)
    variants.append(
        (
            "3_statistical_outlier_removal",
            sor_res.cloud,
            sor_res.processing_time_ms,
            sor_res.retention_ratio,
            sor_res.parameters,
        )
    )

    # 4. Radius Outlier Removal (ROR)
    ror_res = radius_outlier_removal(raw_with_normals, nb_points=12, radius=None)
    variants.append(
        (
            "4_radius_outlier_removal",
            ror_res.cloud,
            ror_res.processing_time_ms,
            ror_res.retention_ratio,
            ror_res.parameters,
        )
    )

    # 5. SOR + Normal Grazing-Angle Filtering
    graz_res = normal_grazing_angle_filter(sor_res.cloud, max_grazing_angle_deg=80.0)
    variants.append(
        (
            "5_sor_plus_normal_grazing_filter",
            graz_res.cloud,
            sor_res.processing_time_ms + graz_res.processing_time_ms,
            float(graz_res.output_points) / float(raw_with_normals.num_points),
            {**sor_res.parameters, **graz_res.parameters},
        )
    )

    # 6. Full Pipeline: SOR + Grazing Filter + Tangent-Plane Normal Smoothing
    smooth_res = smooth_point_cloud_tangent_plane(graz_res.cloud, alpha=0.35)
    variants.append(
        (
            "6_full_cleaned_and_smoothed",
            smooth_res.cloud,
            sor_res.processing_time_ms + graz_res.processing_time_ms + smooth_res.processing_time_ms,
            float(smooth_res.output_points) / float(raw_with_normals.num_points),
            {**sor_res.parameters, **graz_res.parameters, **smooth_res.parameters},
        )
    )

    records = []
    for name, cloud_obj, proc_ms, retention, params in variants:
        ply_path = save_point_cloud_ply(cloud_obj, exp_dir / f"{name}.ply")
        preview_path = exp_dir / f"{name}_preview.png"
        render_point_cloud_orthographic_preview(cloud_obj, preview_path)

        d_stats = compute_point_cloud_density_stats(cloud_obj)
        records.append(
            {
                "variant": name,
                "parameters": params,
                "num_points": cloud_obj.num_points,
                "retention_pct": round(retention * 100.0, 2),
                "processing_time_ms": round(proc_ms, 2),
                "nn_distance_mean": round(d_stats.nn_distance_mean, 5),
                "nn_distance_std": round(d_stats.nn_distance_std, 5),
                "nn_distance_p95": round(d_stats.nn_distance_p95, 5),
                "estimated_surface_density": round(d_stats.estimated_surface_density, 1),
                "bbox_extent": [round(x, 3) for x in d_stats.bbox_extent],
                "ply_file": ply_path.name,
                "preview_png": preview_path.name,
            }
        )

    payload = {
        "experiment": "Experiment C: Point Cloud Filtering, Normal Orientation & Smoothing Ablation",
        "git_commit": commit_hash,
        "input_image": sample_path.name,
        "processed_resolution": f"{prep.width}x{prep.height}",
        "depth_model": "depth_anything_v2_small",
        "variants": records,
    }

    out_json = exp_dir / "experiment_c_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved Experiment C results to %s", out_json)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
