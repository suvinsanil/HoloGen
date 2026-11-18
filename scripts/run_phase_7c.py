"""Phase 7c: Dense Multi-View Fusion (sparse-aligned monocular depth + consistency filtering + TSDF)
on Fountain (10 views) and ETH3D 'pipes' (14 views), plus a small consistency-threshold ablation.
"""

import json
from pathlib import Path
import sys
import tempfile

import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.evaluation.multiview_datasets import load_eth3d_pipes_dataset, load_fountain_dataset  # noqa: E402
from app.geometry.multi_view_fusion import MultiViewFusionConfig, fuse_multiview_tsdf_and_mesh  # noqa: E402
from app.vision.depth_estimation import create_depth_model  # noqa: E402
from app.vision.sfm import run_incremental_sfm  # noqa: E402


def _run_scene(
    scene_name: str,
    image_paths: list[Path],
    depth_model,
    out_dir: Path,
    cfg: MultiViewFusionConfig,
    save_geometry: bool = True,
) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"hologen_7c_{scene_name}_") as tmp_u:
        sfm_res = run_incremental_sfm(
            image_paths=image_paths,
            sfm_max_image_size=1600,
            camera_model="SIMPLE_RADIAL",
            fallback_camera_model="OPENCV",
            single_camera=True,
            max_num_features=4096,
            num_threads=4,
            undistort=True,
            undistorted_dir=Path(tmp_u) / "undist",
        )
        fusion_res = fuse_multiview_tsdf_and_mesh(sfm_res, depth_model, cfg)

    if save_geometry:
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(fusion_res.point_cloud.points.astype("float64"))
        o3d_pcd.colors = o3d.utility.Vector3dVector(fusion_res.point_cloud.colors.astype("float64"))
        o3d.io.write_point_cloud(str(out_dir / f"{scene_name}_fused_cloud.ply"), o3d_pcd)
        o3d.io.write_triangle_mesh(str(out_dir / f"{scene_name}_fused_mesh.ply"), fusion_res.mesh.to_open3d())

    return {
        "scene": scene_name,
        "sfm_summary": {
            "registered": sfm_res.num_registered_images,
            "total": sfm_res.num_input_images,
            "camera_model": sfm_res.shared_camera_model,
            "fallback_triggered": sfm_res.fallback_triggered,
            "undistorted": sfm_res.undistorted,
            "sfm_runtime_ms": round(sfm_res.total_runtime_ms, 2),
            "sfm_peak_ram_mb": round(sfm_res.peak_ram_mb, 2),
        },
        "fusion": fusion_res.to_metadata_dict(),
    }


def main() -> int:
    settings = get_settings()
    out_dir = settings.outputs_dir / "phase_7c"
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_model = create_depth_model("depth_anything_v2_small")
    fountain_views, _ = load_fountain_dataset(settings.datasets_dir)
    pipes_views, _ = load_eth3d_pipes_dataset(settings.datasets_dir)

    fountain_sub = fountain_views[::3][:10]
    default_cfg = MultiViewFusionConfig(
        depth_max_dim=640,
        discontinuity_threshold=0.08,
        consistency_rel_depth_thresh=0.10,
        consistency_rgb_thresh=0.18,
        min_consistent_views=1,
    )

    fountain_res = _run_scene(
        "fountain_10views",
        [v.image_path for v in fountain_sub],
        depth_model,
        out_dir,
        default_cfg,
        save_geometry=True,
    )
    pipes_res = _run_scene(
        "eth3d_pipes_14views",
        [v.image_path for v in pipes_views],
        depth_model,
        out_dir,
        default_cfg,
        save_geometry=True,
    )

    # Small consistency threshold ablation on Fountain 10 views
    ablation = []
    ablation_configs = [
        ("strict", 0.05, 0.12),
        ("default", 0.10, 0.18),
        ("relaxed", 0.20, 0.30),
    ]
    with tempfile.TemporaryDirectory(prefix="hologen_7c_abl_") as tmp_u:
        sfm_abl = run_incremental_sfm(
            image_paths=[v.image_path for v in fountain_sub],
            sfm_max_image_size=1600,
            camera_model="SIMPLE_RADIAL",
            fallback_camera_model="OPENCV",
            single_camera=True,
            max_num_features=4096,
            num_threads=4,
            undistort=True,
            undistorted_dir=Path(tmp_u) / "undist",
        )
        for label, rel_d, rgb_t in ablation_configs:
            cfg_a = MultiViewFusionConfig(
                depth_max_dim=640,
                discontinuity_threshold=0.08,
                consistency_rel_depth_thresh=rel_d,
                consistency_rgb_thresh=rgb_t,
                min_consistent_views=1,
            )
            f_a = fuse_multiview_tsdf_and_mesh(sfm_abl, depth_model, cfg_a)
            meta = f_a.to_metadata_dict()
            ablation.append({
                "setting": label,
                "consistency_rel_depth_thresh": rel_d,
                "consistency_rgb_thresh": rgb_t,
                "rejected_consistency_pct": meta["rejected_consistency_pct"],
                "valid_fused_pct": meta["valid_fused_pct"],
                "fused_points": meta["fused_points"],
                "mesh_faces": meta["mesh_faces"],
                "total_runtime_ms": meta["total_runtime_ms"],
                "peak_ram_mb": meta["peak_ram_mb"],
            })

    payload = {
        "phase": "phase_7c_dense_multiview_fusion",
        "scenes": [fountain_res, pipes_res],
        "consistency_ablation_fountain_10views": ablation,
    }
    out_json = out_dir / "phase_7c_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(ablation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
