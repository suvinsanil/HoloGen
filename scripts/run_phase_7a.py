"""Phase 7a Benchmark & Visualization Runner:
Evaluates two-view pose recovery and triangulation across:
1. Synthetic scenes with known ground-truth pose (noise-free + outliers, and noisy + outliers)
2. Real Pair 1: Open3D SampleFountainRGBDImages (views 00 and 02, reference RGB-D pose)
3. Real Pair 2: ETH3D 'pipes' DSLR (views DSC_0635 and DSC_0636, ground-truth calibrated pose)

Compares:
- `from_scratch_8pt_sampson` (Our from-scratch normalized 8-point + Sampson RANSAC + E + cheirality + DLT)
- `opencv_5pt_baseline` (`cv2.findEssentialMat` + `cv2.recoverPose`)
- `pycolmap_baseline` (`pycolmap.estimate_calibrated_two_view_geometry`)
"""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.core.logging import get_logger
from app.evaluation.multiview_datasets import (
    load_eth3d_pipes_dataset,
    load_fountain_dataset,
    relative_pose_between_views,
)
from app.geometry.two_view import (
    compute_pose_errors_deg,
    estimate_two_view_geometry_from_scratch,
    estimate_two_view_opencv_baseline,
    estimate_two_view_pycolmap_baseline,
    render_epipolar_lines,
)
from app.vision.features import (
    extract_features,
    match_features,
    render_feature_matches,
)
from tests.test_two_view_geometry import _generate_synthetic_two_view_scene

logger = get_logger("scripts.run_phase_7a")


def evaluate_three_methods(
    pair_name: str,
    pose_semantics: str,
    pts1: np.ndarray,
    pts2: np.ndarray,
    intr1,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    sampson_thresh: float = 1.5,
) -> dict:
    res_scratch = estimate_two_view_geometry_from_scratch(
        pts1, pts2, intr1, sampson_threshold_px=sampson_thresh, max_ransac_iterations=1500, seed=42
    )
    res_opencv = estimate_two_view_opencv_baseline(
        pts1, pts2, intr1, ransac_threshold_px=sampson_thresh
    )
    res_colmap = estimate_two_view_pycolmap_baseline(
        pts1, pts2, intr1, max_error_px=sampson_thresh
    )

    methods_out = []
    for res in (res_scratch, res_opencv, res_colmap):
        pose_err = compute_pose_errors_deg(res.R, res.t, R_gt, t_gt)
        # Also compare against pycolmap's recovered pose directly
        vs_colmap = compute_pose_errors_deg(res.R, res.t, res_colmap.R, res_colmap.t)
        methods_out.append(
            {
                **res.to_metadata_dict(),
                "pose_error_vs_gt_or_ref": pose_err,
                "pose_diff_vs_pycolmap": vs_colmap,
            }
        )

    return {
        "pair_name": pair_name,
        "pose_semantics": pose_semantics,
        "num_input_matches": int(len(pts1)),
        "results": methods_out,
        "_scratch_obj": res_scratch,
    }


def main() -> int:
    settings = get_settings()
    out_dir = settings.outputs_dir / "phase_7a"
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = []

    # 1. Synthetic Scene A: Noise-Free + 15% Outliers
    intr_syn1, _, R_gt1, t_gt1, pts1_syn1, pts2_syn1 = _generate_synthetic_two_view_scene(
        num_inliers=100, num_outliers=18, noise_px=0.0, seed=42
    )
    b1 = evaluate_three_methods(
        "synthetic_noisefree_15pct_outliers",
        "exact_synthetic_ground_truth",
        pts1_syn1,
        pts2_syn1,
        intr_syn1,
        R_gt1,
        t_gt1,
    )
    del b1["_scratch_obj"]
    benchmarks.append(b1)

    # 2. Synthetic Scene B: Gaussian Noise (sigma=0.35 px) + 15% Outliers
    intr_syn2, _, R_gt2, t_gt2, pts1_syn2, pts2_syn2 = _generate_synthetic_two_view_scene(
        num_inliers=120, num_outliers=20, noise_px=0.35, seed=101
    )
    b2 = evaluate_three_methods(
        "synthetic_noisy_0.35px_15pct_outliers",
        "exact_synthetic_ground_truth",
        pts1_syn2,
        pts2_syn2,
        intr_syn2,
        R_gt2,
        t_gt2,
    )
    del b2["_scratch_obj"]
    benchmarks.append(b2)

    # 3. Real Pair 1: Fountain (view 00 and view 02 — reference RGB-D trajectory pose)
    fountain_views, _ = load_fountain_dataset(settings.datasets_dir)
    fv1, fv2 = fountain_views[0], fountain_views[2]
    rgb_f1, intr_f1 = fv1.load_rgb_resized(max_dim=640)
    rgb_f2, _ = fv2.load_rgb_resized(max_dim=640)
    R_f_ref, t_f_ref = relative_pose_between_views(fv1, fv2)

    feats_f1 = extract_features(rgb_f1, method="sift", max_keypoints=4096, sfm_max_dim=640)
    feats_f2 = extract_features(rgb_f2, method="sift", max_keypoints=4096, sfm_max_dim=640)
    matches_f = match_features(feats_f1, feats_f2, ratio_threshold=0.75, mutual_check=True)

    b3 = evaluate_three_methods(
        "real_fountain_views_00_02",
        "reference_rgbd_trajectory (not laser ground truth)",
        matches_f.pts1_xy,
        matches_f.pts2_xy,
        intr_f1,
        R_f_ref,
        t_f_ref,
        sampson_thresh=1.5,
    )
    res_f_scratch = b3.pop("_scratch_obj")
    benchmarks.append(b3)

    render_feature_matches(
        rgb_f1,
        rgb_f2,
        matches_f.pts1_xy,
        matches_f.pts2_xy,
        inlier_mask=res_f_scratch.inlier_mask,
        output_path=out_dir / "fountain_00_02_matches.png",
    )
    render_epipolar_lines(
        rgb_f1,
        rgb_f2,
        matches_f.pts1_xy[res_f_scratch.inlier_mask],
        matches_f.pts2_xy[res_f_scratch.inlier_mask],
        res_f_scratch.F,
        output_path=out_dir / "fountain_00_02_epipolar_lines.png",
    )

    # 4. Real Pair 2: ETH3D 'pipes' DSLR (DSC_0635 and DSC_0636 at 1600 px — ground-truth pose)
    pipes_views, _ = load_eth3d_pipes_dataset(settings.datasets_dir)
    pv1, pv2 = pipes_views[1], pipes_views[2]  # DSC_0635 and DSC_0636
    rgb_p1, intr_p1 = pv1.load_rgb_resized(max_dim=1600)
    rgb_p2, _ = pv2.load_rgb_resized(max_dim=1600)
    R_p_gt, t_p_gt = relative_pose_between_views(pv1, pv2)

    feats_p1 = extract_features(rgb_p1, method="sift", max_keypoints=4096, sfm_max_dim=1600)
    feats_p2 = extract_features(rgb_p2, method="sift", max_keypoints=4096, sfm_max_dim=1600)
    matches_p = match_features(feats_p1, feats_p2, ratio_threshold=0.75, mutual_check=True)

    b4 = evaluate_three_methods(
        f"real_eth3d_pipes_{pv1.view_id}_vs_{pv2.view_id}",
        "eth3d_ground_truth_calibrated_pose",
        matches_p.pts1_xy,
        matches_p.pts2_xy,
        intr_p1,
        R_p_gt,
        t_p_gt,
        sampson_thresh=2.0,
    )
    res_p_scratch = b4.pop("_scratch_obj")
    benchmarks.append(b4)

    render_feature_matches(
        rgb_p1,
        rgb_p2,
        matches_p.pts1_xy,
        matches_p.pts2_xy,
        inlier_mask=res_p_scratch.inlier_mask,
        output_path=out_dir / "eth3d_pipes_matches.png",
    )
    render_epipolar_lines(
        rgb_p1,
        rgb_p2,
        matches_p.pts1_xy[res_p_scratch.inlier_mask],
        matches_p.pts2_xy[res_p_scratch.inlier_mask],
        res_p_scratch.F,
        output_path=out_dir / "eth3d_pipes_epipolar_lines.png",
    )

    payload = {
        "phase": "phase_7a_two_view_geometry_benchmark",
        "from_scratch_components": [
            "Hartley isotropic 2D point normalization (normalize_points_2d)",
            "Normalized 8-point algorithm with SVD rank-2 enforcement (eight_point_fundamental_matrix)",
            "First-order Sampson epipolar distance (compute_sampson_distance)",
            "Adaptive RANSAC loop with MSAC scoring and inlier least-squares refit (ransac_fundamental_matrix)",
            "Essential matrix projection onto diag(1,1,0) from F and K (essential_from_fundamental)",
            "4-hypothesis Essential SVD decomposition onto SO(3) x S^2 (decompose_essential_matrix)",
            "Row-normalized homogeneous SVD DLT triangulation (triangulate_points_dlt)",
            "Cheirality check (Z1 > 0 and Z2 > 0) and symmetric reprojection error (recover_pose_cheirality)",
        ],
        "comparison_baselines": [
            "opencv_5pt_baseline (cv2.findEssentialMat + cv2.recoverPose)",
            "pycolmap_baseline (pycolmap.estimate_calibrated_two_view_geometry)",
        ],
        "benchmarks": benchmarks,
    }

    out_json = out_dir / "phase_7a_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
