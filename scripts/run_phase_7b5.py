"""Phase 7b.5 Fix-Up Benchmark:
1. Two-view multi-pair (10 pairs/scene) x multi-seed (3 RANSAC seeds) evaluation on:
   - Fountain (10 pairs, RGB-D reference trajectory)
   - ETH3D 'pipes' (10 pairs on undistorted PINHOLE images, ground-truth calibrated poses)
   Evaluates the exact same SIFT matches across:
     - from_scratch_8pt_no_gn (iterative 8-pt F refit on inliers, no GN)
     - from_scratch_8pt_with_gn (ablation: + 5-DoF Sampson Gauss-Newton on SO(3) x S^2)
     - opencv_5pt_baseline
     - pycolmap_baseline
   Reports median and IQR (Q75 - Q25) for rotation error, translation-direction error, and reprojection RMSE.
2. N-view SfM camera-model policy (default SIMPLE_RADIAL, auto-fallback to OPENCV if < 100% registered)
   + pycolmap undistortion into PINHOLE images & intrinsics, with peak RSS telemetry.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List

import cv2
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.evaluation.multiview_datasets import (  # noqa: E402
    load_eth3d_pipes_dataset,
    load_fountain_dataset,
    relative_pose_between_views,
)
from app.geometry.two_view import (  # noqa: E402
    compute_pose_errors_deg,
    estimate_two_view_geometry_from_scratch,
    estimate_two_view_opencv_baseline,
    estimate_two_view_pycolmap_baseline,
)
from app.vision.features import extract_features, match_features  # noqa: E402
from app.vision.sfm import evaluate_sfm_poses_against_reference, run_incremental_sfm  # noqa: E402


def _summary_stats(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    q25, med, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
    return {
        "median": round(float(med), 4),
        "iqr": round(float(q75 - q25), 4),
        "q25": round(float(q25), 4),
        "q75": round(float(q75), 4),
    }


def _eval_pairs_multiseed(
    scene_name: str,
    pose_semantics: str,
    pair_specs: List[Dict[str, Any]],
    seeds: List[int],
    sampson_thresh: float,
) -> Dict[str, Any]:
    methods = [
        "from_scratch_8pt_no_gn",
        "from_scratch_8pt_with_gn",
        "opencv_5pt_baseline",
        "pycolmap_baseline",
    ]
    per_method_metrics: Dict[str, Dict[str, List[float]]] = {
        m: {"rot_err_deg": [], "trans_err_deg": [], "reproj_rmse_px": [], "runtime_ms": [], "inliers": []}
        for m in methods
    }

    for spec in pair_specs:
        pts1 = spec["pts1"]
        pts2 = spec["pts2"]
        intr = spec["intr"]
        R_gt = spec["R_gt"]
        t_gt = spec["t_gt"]

        for seed in seeds:
            r_no_gn = estimate_two_view_geometry_from_scratch(
                pts1, pts2, intr, sampson_threshold_px=sampson_thresh, seed=seed, refine_essential_gn=False
            )
            r_gn = estimate_two_view_geometry_from_scratch(
                pts1, pts2, intr, sampson_threshold_px=sampson_thresh, seed=seed, refine_essential_gn=True
            )
            r_cv = estimate_two_view_opencv_baseline(pts1, pts2, intr, ransac_threshold_px=sampson_thresh)
            r_col = estimate_two_view_pycolmap_baseline(pts1, pts2, intr, max_error_px=sampson_thresh, seed=seed)

            for key, res in (
                ("from_scratch_8pt_no_gn", r_no_gn),
                ("from_scratch_8pt_with_gn", r_gn),
                ("opencv_5pt_baseline", r_cv),
                ("pycolmap_baseline", r_col),
            ):
                errs = compute_pose_errors_deg(res.R, res.t, R_gt, t_gt)
                per_method_metrics[key]["rot_err_deg"].append(float(errs["rotation_error_deg"]))
                per_method_metrics[key]["trans_err_deg"].append(float(errs["translation_direction_error_deg"]))
                per_method_metrics[key]["reproj_rmse_px"].append(float(res.rmse_reprojection_error_px))
                per_method_metrics[key]["runtime_ms"].append(float(res.runtime_ms))
                per_method_metrics[key]["inliers"].append(float(res.num_inliers))

    summary = {}
    for m in methods:
        summary[m] = {
            "num_runs": len(pair_specs) * len(seeds),
            "rotation_error_deg": _summary_stats(per_method_metrics[m]["rot_err_deg"]),
            "translation_direction_error_deg": _summary_stats(per_method_metrics[m]["trans_err_deg"]),
            "reprojection_rmse_px": _summary_stats(per_method_metrics[m]["reproj_rmse_px"]),
            "runtime_ms": _summary_stats(per_method_metrics[m]["runtime_ms"]),
            "num_inliers": _summary_stats(per_method_metrics[m]["inliers"]),
        }

    return {
        "scene": scene_name,
        "pose_semantics": pose_semantics,
        "num_pairs": len(pair_specs),
        "pair_names": [p["pair_name"] for p in pair_specs],
        "seeds": seeds,
        "sampson_threshold_px": sampson_thresh,
        "summary_median_iqr": summary,
    }


def main() -> int:
    settings = get_settings()
    out_dir = settings.outputs_dir / "phase_7b5"
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process(os.getpid())

    seeds = [42, 101, 2026]
    fountain_views, _ = load_fountain_dataset(settings.datasets_dir)
    pipes_views, _ = load_eth3d_pipes_dataset(settings.datasets_dir)

    # 1. N-view SfM with automatic camera-model fallback + undistortion
    sfm_reports = []
    with tempfile.TemporaryDirectory(prefix="hologen_7b5_undist_") as tmp_u:
        tmp_u_path = Path(tmp_u)
        fountain_undist = tmp_u_path / "fountain_undist"
        pipes_undist = tmp_u_path / "pipes_undist"

        sfm_fountain = run_incremental_sfm(
            image_paths=[v.image_path for v in fountain_views],
            sfm_max_image_size=1600,
            camera_model="SIMPLE_RADIAL",
            fallback_camera_model="OPENCV",
            single_camera=True,
            max_num_features=4096,
            num_threads=4,
            undistort=True,
            undistorted_dir=fountain_undist,
        )
        f_pose_eval, _ = evaluate_sfm_poses_against_reference(
            sfm_fountain, {v.image_path.name: v for v in fountain_views}
        )
        sfm_reports.append({
            "case": "fountain_all33_auto_policy",
            "pose_semantics": "reference_rgbd_trajectory (key.log)",
            "sfm": sfm_fountain.to_metadata_dict(),
            "pose_eval": f_pose_eval,
        })

        sfm_pipes = run_incremental_sfm(
            image_paths=[v.image_path for v in pipes_views],
            sfm_max_image_size=1600,
            camera_model="SIMPLE_RADIAL",
            fallback_camera_model="OPENCV",
            single_camera=True,
            max_num_features=4096,
            num_threads=4,
            undistort=True,
            undistorted_dir=pipes_undist,
        )
        p_pose_eval, _ = evaluate_sfm_poses_against_reference(
            sfm_pipes, {v.image_path.name: v for v in pipes_views}
        )
        sfm_reports.append({
            "case": "eth3d_pipes_all14_auto_policy",
            "pose_semantics": "eth3d_ground_truth_calibrated_pose",
            "sfm": sfm_pipes.to_metadata_dict(),
            "pose_eval": p_pose_eval,
        })

        # 2. Build 10 overlapping pairs on Fountain (step=2 across the sequence, >= 30 matches)
        fountain_pairs = []
        for idx in range(0, len(fountain_views) - 2, 2):
            if len(fountain_pairs) >= 10:
                break
            v1, v2 = fountain_views[idx], fountain_views[idx + 2]
            rgb1, intr1 = v1.load_rgb_resized(max_dim=640)
            rgb2, _ = v2.load_rgb_resized(max_dim=640)
            R_gt, t_gt = relative_pose_between_views(v1, v2)
            f1 = extract_features(rgb1, method="sift", max_keypoints=4096, sfm_max_dim=640)
            f2 = extract_features(rgb2, method="sift", max_keypoints=4096, sfm_max_dim=640)
            m = match_features(f1, f2, ratio_threshold=0.75, mutual_check=True)
            if m.num_matches < 30:
                continue
            fountain_pairs.append({
                "pair_name": f"fountain_{idx:02d}_vs_{idx+2:02d}",
                "pts1": m.pts1_xy,
                "pts2": m.pts2_xy,
                "intr": intr1,
                "R_gt": R_gt,
                "t_gt": t_gt,
            })

        # 3. Build 10 overlapping pairs on ETH3D 'pipes' using UNDISTORTED PINHOLE images & PINHOLE K
        pipes_gt_by_name = {v.image_path.name: v for v in pipes_views}
        reg_names = sorted(sfm_pipes.registered_poses.keys())
        pipes_feats = {}
        pipes_rgbs = {}
        for n in reg_names:
            pose_u = sfm_pipes.registered_poses[n]
            rgb = cv2.imread(str(pose_u.image_path))[:, :, ::-1]
            pipes_rgbs[n] = rgb
            pipes_feats[n] = extract_features(rgb, method="sift", max_keypoints=4096, sfm_max_dim=1600)

        candidate_pairs = []
        for i1 in range(len(reg_names)):
            for i2 in range(i1 + 1, len(reg_names)):
                n1, n2 = reg_names[i1], reg_names[i2]
                m = match_features(pipes_feats[n1], pipes_feats[n2], ratio_threshold=0.75, mutual_check=True)
                if m.num_matches >= 40:
                    candidate_pairs.append((m.num_matches, n1, n2, m))
        candidate_pairs.sort(key=lambda x: x[0], reverse=True)
        pipes_pairs = []
        for _, n1, n2, m in candidate_pairs[:10]:
            pose_u1 = sfm_pipes.registered_poses[n1]
            R_gt, t_gt = relative_pose_between_views(pipes_gt_by_name[n1], pipes_gt_by_name[n2])
            pipes_pairs.append({
                "pair_name": f"{n1}_vs_{n2}",
                "pts1": m.pts1_xy,
                "pts2": m.pts2_xy,
                "intr": pose_u1.intrinsics,
                "R_gt": R_gt,
                "t_gt": t_gt,
            })

        two_view_fountain = _eval_pairs_multiseed(
            "fountain_10pairs_x_3seeds",
            "reference_rgbd_trajectory (key.log)",
            fountain_pairs,
            seeds,
            sampson_thresh=1.5,
        )
        two_view_pipes = _eval_pairs_multiseed(
            "eth3d_pipes_undistorted_pinhole_10pairs_x_3seeds",
            "eth3d_ground_truth_calibrated_pose",
            pipes_pairs,
            seeds,
            sampson_thresh=1.5,
        )

    payload = {
        "phase": "phase_7b5_fixup_benchmark",
        "peak_rss_mb": round(proc.memory_info().rss / (1024.0 * 1024.0), 2),
        "sfm_camera_policy_and_undistortion": sfm_reports,
        "two_view_multiseed_benchmarks": [two_view_fountain, two_view_pipes],
    }
    out_json = out_dir / "phase_7b5_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
