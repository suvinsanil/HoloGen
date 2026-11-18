"""Phase 7c.5 Validation:
1. 80/20 held-out sparse-depth alignment check (Abs Rel and inlier ratio on held-out 20% vs in-sample fit error).
2. Whole-pipeline peak RSS across SfM + undistortion + depth + alignment + consistency + TSDF fusion + meshing.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import threading

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.evaluation.multiview_datasets import load_eth3d_pipes_dataset, load_fountain_dataset  # noqa: E402
from app.geometry.multi_view_fusion import MultiViewFusionConfig, fuse_multiview_tsdf_and_mesh  # noqa: E402
from app.vision.depth_estimation import create_depth_model  # noqa: E402
from app.vision.sfm import run_incremental_sfm  # noqa: E402


def _eval_scene(scene_name: str, image_paths: list[Path], depth_model, cfg: MultiViewFusionConfig) -> dict:
    proc = psutil.Process(os.getpid())
    baseline_rss = proc.memory_info().rss
    peak_rss = baseline_rss
    stop_evt = threading.Event()

    def _poll():
        nonlocal peak_rss
        while not stop_evt.is_set():
            curr = proc.memory_info().rss
            if curr > peak_rss:
                peak_rss = curr
            stop_evt.wait(0.02)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    try:
        with tempfile.TemporaryDirectory(prefix=f"hologen_7c5_{scene_name}_") as tmp_u:
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
    finally:
        stop_evt.set()
        t.join(timeout=1.0)

    whole_peak_rss_mb = round(peak_rss / (1024.0 * 1024.0), 2)
    whole_peak_delta_mb = round((peak_rss - baseline_rss) / (1024.0 * 1024.0), 2)

    in_sample_abs_rel = [v.sparse_abs_rel for v in fusion_res.view_stats if not v.rejected]
    in_sample_inl = [v.inlier_ratio for v in fusion_res.view_stats if not v.rejected]
    ho_abs_rel = [v.heldout_abs_rel for v in fusion_res.view_stats if v.heldout_abs_rel is not None]
    ho_inl = [v.heldout_inlier_ratio for v in fusion_res.view_stats if v.heldout_inlier_ratio is not None]

    return {
        "scene": scene_name,
        "registered_views": sfm_res.num_registered_images,
        "total_views": sfm_res.num_input_images,
        "camera_model": sfm_res.shared_camera_model,
        "fallback_triggered": sfm_res.fallback_triggered,
        "in_sample_fit_abs_rel_median": round(float(np.median(in_sample_abs_rel)), 4),
        "in_sample_inlier_ratio_median": round(float(np.median(in_sample_inl)), 4),
        "heldout_20pct_abs_rel_median": round(float(np.median(ho_abs_rel)), 4) if ho_abs_rel else None,
        "heldout_20pct_abs_rel_min_max": [round(min(ho_abs_rel), 4), round(max(ho_abs_rel), 4)] if ho_abs_rel else None,
        "heldout_20pct_inlier_ratio_median": round(float(np.median(ho_inl)), 4) if ho_inl else None,
        "heldout_20pct_inlier_ratio_min_max": [round(min(ho_inl), 4), round(max(ho_inl), 4)] if ho_inl else None,
        "sfm_peak_rss_mb": round(sfm_res.peak_ram_mb, 2),
        "fusion_peak_rss_mb": round(fusion_res.peak_ram_mb, 2),
        "whole_pipeline_peak_rss_mb": whole_peak_rss_mb,
        "whole_pipeline_peak_above_baseline_mb": whole_peak_delta_mb,
        "sfm_runtime_ms": round(sfm_res.total_runtime_ms, 2),
        "fusion_runtime_ms": round(fusion_res.total_runtime_ms, 2),
        "whole_pipeline_runtime_ms": round(sfm_res.total_runtime_ms + fusion_res.total_runtime_ms, 2),
    }


def main() -> int:
    settings = get_settings()
    out_dir = settings.outputs_dir / "phase_7c5"
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_model = create_depth_model("depth_anything_v2_small")
    fountain_views, _ = load_fountain_dataset(settings.datasets_dir)
    pipes_views, _ = load_eth3d_pipes_dataset(settings.datasets_dir)

    cfg = MultiViewFusionConfig(
        depth_max_dim=640,
        discontinuity_threshold=0.08,
        consistency_rel_depth_thresh=0.10,
        consistency_rgb_thresh=0.18,
        min_consistent_views=1,
    )

    fountain_sub = fountain_views[::3][:10]
    res_f = _eval_scene("fountain_10views", [v.image_path for v in fountain_sub], depth_model, cfg)
    res_p = _eval_scene("eth3d_pipes_14views", [v.image_path for v in pipes_views], depth_model, cfg)

    payload = {"phase": "phase_7c5_validation", "scenes": [res_f, res_p]}
    (out_dir / "phase_7c5_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
