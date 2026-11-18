"""Unit tests for Phase 7a from-scratch two-view geometry (normalized 8-point, Sampson RANSAC,
Essential decomposition, cheirality, DLT triangulation) and comparison against OpenCV & pycolmap.
"""

import math
import numpy as np
import pytest

from app.geometry.two_view import (
    compute_pose_errors_deg,
    compute_sampson_distance,
    decompose_essential_matrix,
    eight_point_fundamental_matrix,
    essential_from_fundamental,
    estimate_two_view_geometry_from_scratch,
    estimate_two_view_opencv_baseline,
    estimate_two_view_pycolmap_baseline,
    normalize_points_2d,
    triangulate_points_dlt,
)
from app.vision.camera import CameraIntrinsics, estimate_intrinsics
from app.vision.features import extract_features, match_features


def _rotation_matrix_y_x(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    Ry = np.array(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ],
        dtype=np.float64,
    )
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float64,
    )
    return Rx @ Ry


def _generate_synthetic_two_view_scene(
    num_inliers: int = 80,
    num_outliers: int = 15,
    noise_px: float = 0.0,
    seed: int = 123,
) -> tuple[CameraIntrinsics, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic 3D point cloud projected into two pinhole cameras with known pose."""
    rng = np.random.default_rng(seed)
    intr = estimate_intrinsics(width=800, height=600, focal_length_px=700.0)

    # 3D points well inside the shared viewing frustum
    X_gt = np.column_stack(
        [
            rng.uniform(-1.5, 1.5, size=num_inliers),
            rng.uniform(-1.1, 1.1, size=num_inliers),
            rng.uniform(3.0, 7.5, size=num_inliers),
        ]
    )

    R_gt = _rotation_matrix_y_x(yaw_deg=14.0, pitch_deg=-4.0)
    t_gt = np.array([-0.85, 0.15, 0.25], dtype=np.float64)

    # Project into Camera 1: [I | 0]
    pts1_clean = intr.project(X_gt)
    # Project into Camera 2: X_cam2 = R_gt @ X_gt + t_gt
    X_cam2 = (R_gt @ X_gt.T).T + t_gt[None, :]
    pts2_clean = intr.project(X_cam2)

    if noise_px > 0.0:
        pts1_in = pts1_clean + rng.normal(0.0, noise_px, size=pts1_clean.shape)
        pts2_in = pts2_clean + rng.normal(0.0, noise_px, size=pts2_clean.shape)
    else:
        pts1_in = pts1_clean
        pts2_in = pts2_clean

    if num_outliers > 0:
        out1 = np.column_stack(
            [
                rng.uniform(20.0, 780.0, size=num_outliers),
                rng.uniform(20.0, 580.0, size=num_outliers),
            ]
        )
        out2 = np.column_stack(
            [
                rng.uniform(20.0, 780.0, size=num_outliers),
                rng.uniform(20.0, 580.0, size=num_outliers),
            ]
        )
        pts1 = np.vstack([pts1_in, out1])
        pts2 = np.vstack([pts2_in, out2])
    else:
        pts1 = pts1_in
        pts2 = pts2_in

    return intr, X_gt, R_gt, t_gt, pts1, pts2


def test_hand_computed_hartley_isotropic_normalization() -> None:
    """Hand-computed check:
    For 4 corners of a square centered at (10, 20) with half-side 3:
    (7, 17), (13, 17), (7, 23), (13, 23):
      Centroid = (10, 20).
      Distance of each corner from centroid = sqrt(3^2 + 3^2) = 3 * sqrt(2).
      Required scale s = sqrt(2) / (3 * sqrt(2)) = 1/3.
      Normalized points must be (-1, -1, 1), (+1, -1, 1), (-1, +1, 1), (+1, +1, 1).
    """
    pts = np.array(
        [[7.0, 17.0], [13.0, 17.0], [7.0, 23.0], [13.0, 23.0]],
        dtype=np.float64,
    )
    norm_hom, T = normalize_points_2d(pts)

    assert T[0, 0] == pytest.approx(1.0 / 3.0, rel=1e-12)
    assert T[1, 1] == pytest.approx(1.0 / 3.0, rel=1e-12)
    assert T[0, 2] == pytest.approx(-10.0 / 3.0, rel=1e-12)
    assert T[1, 2] == pytest.approx(-20.0 / 3.0, rel=1e-12)

    expected = np.array(
        [
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(norm_hom, expected, atol=1e-12)
    # Mean radial distance must equal sqrt(2)
    radial = np.linalg.norm(norm_hom[:, :2], axis=1)
    assert float(np.mean(radial)) == pytest.approx(math.sqrt(2.0), rel=1e-12)


def test_eight_point_rank2_and_sampson_distance_zero_on_exact_correspondences() -> None:
    """Verify normalized 8-point produces a rank-2 F with 0 Sampson error on noise-free points."""
    intr, _, _, _, pts1, pts2 = _generate_synthetic_two_view_scene(
        num_inliers=24, num_outliers=0, noise_px=0.0
    )
    F = eight_point_fundamental_matrix(pts1, pts2)
    # Determinant of rank-2 matrix F must be 0
    assert abs(float(np.linalg.det(F))) < 1e-12

    sampson = compute_sampson_distance(pts1, pts2, F)
    np.testing.assert_allclose(sampson, 0.0, atol=1e-9)

    E = essential_from_fundamental(F, intr.K, intr.K)
    _, s_vals, _ = np.linalg.svd(E)
    assert s_vals[0] == pytest.approx(1.0, abs=1e-9)
    assert s_vals[1] == pytest.approx(1.0, abs=1e-9)
    assert s_vals[2] == pytest.approx(0.0, abs=1e-9)

    hyps = decompose_essential_matrix(E)
    assert len(hyps) == 4
    for R_cand, t_cand in hyps:
        assert float(np.linalg.det(R_cand)) == pytest.approx(1.0, abs=1e-9)
        assert float(np.linalg.norm(t_cand)) == pytest.approx(1.0, abs=1e-9)


def test_from_scratch_vs_opencv_vs_pycolmap_on_synthetic_scene_with_outliers() -> None:
    """Verify from-scratch 2-view geometry, OpenCV, and pycolmap recover exact pose with outliers."""
    intr, X_gt, R_gt, t_gt, pts1, pts2 = _generate_synthetic_two_view_scene(
        num_inliers=80, num_outliers=15, noise_px=0.0, seed=99
    )

    res_scratch = estimate_two_view_geometry_from_scratch(pts1, pts2, intr, seed=42)
    res_opencv = estimate_two_view_opencv_baseline(pts1, pts2, intr)
    res_colmap = estimate_two_view_pycolmap_baseline(pts1, pts2, intr)

    for res in (res_scratch, res_opencv, res_colmap):
        errs = compute_pose_errors_deg(res.R, res.t, R_gt, t_gt)
        assert errs["rotation_error_deg"] < 0.05, f"{res.method} rot err: {errs}"
        assert errs["translation_direction_error_deg"] < 0.05, f"{res.method} trans err: {errs}"
        # At least 78/80 true inliers recovered; at most 2 random outliers that happen to lie within the
        # threshold of the epipolar line (two-view geometry cannot reject those).
        assert int(np.sum(res.inlier_mask[:80])) >= 78, f"{res.method} missed true inliers"
        assert int(np.sum(res.inlier_mask[80:])) <= 2, f"{res.method} false inliers"
        assert res.median_reprojection_error_px < 1e-3
        assert res.num_cheirality_valid >= 78
    # Ours and OpenCV recover all 80 true inliers and reject every synthetic outlier on this scene.
    for res in (res_scratch, res_opencv):
        assert bool(np.all(res.inlier_mask[:80]))
        assert res.rmse_reprojection_error_px < 1e-3

    # Verify DLT triangulated points from our from-scratch pipeline match X_gt up to baseline scale
    scale_gt = float(np.linalg.norm(t_gt))
    X_recovered_scaled = res_scratch.points_3d[:80] * scale_gt
    np.testing.assert_allclose(X_recovered_scaled, X_gt, atol=1e-4)


def test_from_scratch_pose_recovery_under_gaussian_pixel_noise() -> None:
    """Verify sub-degree rotation recovery under sigma=0.25 px Gaussian noise and 10% outliers."""
    intr, _, R_gt, t_gt, pts1, pts2 = _generate_synthetic_two_view_scene(
        num_inliers=120, num_outliers=12, noise_px=0.25, seed=2026
    )
    res_scratch = estimate_two_view_geometry_from_scratch(
        pts1, pts2, intr, sampson_threshold_px=1.5, seed=42
    )
    errs = compute_pose_errors_deg(res_scratch.R, res_scratch.t, R_gt, t_gt)
    assert errs["rotation_error_deg"] < 0.8
    assert errs["translation_direction_error_deg"] < 2.0
    assert res_scratch.rmse_reprojection_error_px < 1.0


def test_sift_and_orb_feature_matching_with_mutual_ratio_test() -> None:
    """Verify SIFT and ORB feature extraction and mutual Lowe ratio matching on shifted test pattern."""
    rng = np.random.default_rng(7)
    base_img = rng.integers(30, 220, size=(240, 320, 3), dtype=np.uint8)
    # Apply small 2D translation shift of (+8, +4) pixels
    M = np.float32([[1, 0, 8], [0, 1, 4]])
    import cv2

    shifted_img = cv2.warpAffine(base_img, M, (320, 240))

    for method in ("sift", "orb"):
        f1 = extract_features(base_img, method=method, max_keypoints=1000)
        f2 = extract_features(shifted_img, method=method, max_keypoints=1000)
        assert f1.num_keypoints > 50
        assert f2.num_keypoints > 50

        matches = match_features(f1, f2, ratio_threshold=0.75, mutual_check=True)
        assert matches.num_matches > 20
        diffs = matches.pts2_xy - matches.pts1_xy
        med_dx = float(np.median(diffs[:, 0]))
        med_dy = float(np.median(diffs[:, 1]))
        assert med_dx == pytest.approx(8.0, abs=1.5)
        assert med_dy == pytest.approx(4.0, abs=1.5)
