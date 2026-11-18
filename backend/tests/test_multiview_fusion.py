"""Unit tests for Phase 7c sparse-aligned monocular depth, fallback/rejection, consistency filtering, and TSDF fusion."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from app.geometry.multi_view_fusion import (
    MultiViewFusionConfig,
    align_monocular_disparity_to_sparse_depth,
    filter_depth_by_discontinuity_and_consistency,
    fuse_multiview_tsdf_and_mesh,
)
from app.vision.camera import estimate_intrinsics
from app.vision.depth_estimation import DepthPrediction
from app.vision.sfm import SfmCameraPose, SfmReconstructionResult


def _make_pose(name: str, img_path: Path, tx: float, xyz_w: np.ndarray, intr) -> SfmCameraPose:
    R = np.eye(3, dtype=np.float64)
    t = np.array([tx, 0.0, 0.0], dtype=np.float64)
    pts_c = (R @ xyz_w.T).T + t[None, :]
    z = pts_c[:, 2]
    uv = np.column_stack([intr.fx * pts_c[:, 0] / z + intr.cx, intr.fy * pts_c[:, 1] / z + intr.cy])
    return SfmCameraPose(
        image_id=1,
        image_name=name,
        image_path=img_path,
        camera_id=1,
        intrinsics=intr,
        R_w2c=R,
        t_w2c=t,
        camera_center_w=-R.T @ t,
        observed_uv=uv,
        observed_point3d_ids=np.arange(len(xyz_w), dtype=np.int64),
        observed_xyz_w=xyz_w,
    )


def test_sparse_depth_alignment_ransac_and_scale_only_fallback(tmp_path) -> None:
    """Verify RANSAC affine inverse-depth fit recovers exact scale+shift with outliers,
    falls back to scale-only when shift produces non-positive disparity, and rejects bad views."""
    intr = estimate_intrinsics(width=160, height=120, focal_length_px=140.0)
    rng = np.random.default_rng(7)
    xyz_w = np.column_stack([rng.uniform(-0.8, 0.8, 40), rng.uniform(-0.6, 0.6, 40), rng.uniform(2.0, 5.0, 40)])
    pose = _make_pose("v0.png", tmp_path / "v0.png", 0.0, xyz_w, intr)

    # Create disparity map d_mono where 1/Z_true = 0.4 * d_mono + 0.1 -> d_mono = (1/Z - 0.1)/0.4
    Z_true = np.full((120, 160), 3.0, dtype=np.float32)
    # Vary depth across columns so d_mono has gradient
    xs = np.linspace(2.0, 5.0, 160, dtype=np.float32)[None, :]
    Z_true[:] = xs
    # Set sparse Z to match Z_true at projected pixels (with 20% outliers)
    u_px = np.clip(np.round(pose.observed_uv[:, 0]).astype(int), 0, 159)
    z_at_uv = Z_true[0, u_px].astype(np.float64)
    # Inject 6 gross outliers
    z_at_uv[:6] = 12.0
    pose.observed_xyz_w[:, 2] = z_at_uv
    pose.observed_xyz_w[:, 0] = (pose.observed_uv[:, 0] - intr.cx) * z_at_uv / intr.fx
    pose.observed_xyz_w[:, 1] = (pose.observed_uv[:, 1] - intr.cy) * z_at_uv / intr.fy

    d_mono = np.clip((1.0 / Z_true - 0.1) / 0.4, 0.02, 1.0).astype(np.float32)
    cfg = MultiViewFusionConfig(min_inlier_ratio=0.25, ransac_iterations=200)
    aligned, mode, s, t, n_pts, inl_ratio, abs_rel, rmse, _ = align_monocular_disparity_to_sparse_depth(
        d_mono, pose, intr, cfg
    )
    assert aligned is not None
    assert mode == "scale_shift_ransac"
    assert s == pytest.approx(0.4, abs=1e-2)
    assert t == pytest.approx(0.1, abs=1e-2)
    assert inl_ratio >= 0.80
    np.testing.assert_allclose(aligned, Z_true, rtol=1e-2)

    from app.geometry.multi_view_fusion import evaluate_heldout_sparse_alignment

    ho_abs_rel, ho_inl = evaluate_heldout_sparse_alignment(d_mono, pose, cfg, train_frac=0.8)
    assert ho_abs_rel is not None and ho_inl is not None
    assert ho_inl >= 0.70
    assert ho_abs_rel < 0.25

    # Test view rejection when sparse points are completely random / corrupted
    pose_bad = _make_pose("bad.png", tmp_path / "bad.png", 0.0, xyz_w[:12].copy(), intr)
    pose_bad.observed_xyz_w[:, 2] = rng.uniform(0.5, 50.0, 12)
    aligned_bad, mode_bad, *_ = align_monocular_disparity_to_sparse_depth(d_mono, pose_bad, intr, cfg)
    assert aligned_bad is None
    assert mode_bad == "rejected"


def test_consistency_filter_rejects_discontinuity_and_inconsistent_pixels(tmp_path) -> None:
    """Verify high-discontinuity step edges and neighbor-inconsistent pixels are masked out."""
    intr = estimate_intrinsics(width=80, height=60, focal_length_px=70.0)
    rgb = np.full((60, 80, 3), 180, dtype=np.uint8)
    d1 = np.full((60, 80), 2.5, dtype=np.float32)
    # Create sharp step edge discontinuity on left half vs right half
    d1[:, 40:] = 6.0

    # Neighbor view sees depth 2.5 everywhere except a corrupted patch at (10:20, 10:20)
    d2 = d1.copy()
    d2[10:20, 10:20] = 8.0
    rgb2 = rgb.copy()

    p1 = _make_pose("v1.png", tmp_path / "v1.png", 0.0, np.array([[0, 0, 2.5], [0.2, 0, 2.5], [0, 0.2, 2.5]]), intr)
    p2 = _make_pose("v2.png", tmp_path / "v2.png", 0.0, np.array([[0, 0, 2.5], [0.2, 0, 2.5], [0, 0.2, 2.5]]), intr)

    cfg = MultiViewFusionConfig(discontinuity_threshold=0.08, consistency_rel_depth_thresh=0.10, min_consistent_views=1)
    filt, tot, rej_disc, rej_cons, valid_px = filter_depth_by_discontinuity_and_consistency(
        "v1.png", rgb, d1, intr, p1, [(p2, intr, d2, rgb2)], cfg
    )
    assert tot == 80 * 60
    assert rej_disc > 0  # step edge at x=40 rejected
    assert rej_cons >= 80  # corrupted 10x10 patch rejected by geometric consistency
    assert valid_px > 0
    assert np.all(filt[10:20, 10:20] == 0.0)


def test_end_to_end_sequential_tsdf_fusion_on_synthetic_scene(tmp_path) -> None:
    """Verify fuse_multiview_tsdf_and_mesh fuses 3 planar views into a clean mesh & cloud."""
    intr = estimate_intrinsics(width=120, height=90, focal_length_px=100.0)
    rng = np.random.default_rng(123)
    xyz_w = np.column_stack([rng.uniform(-0.6, 0.6, 30), rng.uniform(-0.4, 0.4, 30), np.full(30, 2.0)])

    poses = {}
    for idx, tx in enumerate((-0.15, 0.0, 0.15)):
        name = f"view_{idx}.png"
        img_p = tmp_path / name
        img = np.full((90, 120, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(img_p), img)
        poses[name] = _make_pose(name, img_p, tx, xyz_w.copy(), intr)

    sfm_res = SfmReconstructionResult(
        num_input_images=3, num_registered_images=3, registration_ratio=1.0,
        shared_camera_model="PINHOLE", single_camera_mode=True, sfm_max_image_size=120,
        shared_intrinsics=intr, registered_poses=poses, unregistered_images=[],
        sparse_points_xyz=xyz_w, sparse_points_rgb=np.full((30, 3), 200, dtype=np.uint8),
        sparse_point_errors_px=np.zeros(30), sparse_point_track_lengths=np.full(30, 3, dtype=np.int32),
        mean_reprojection_error_px=0.2, mean_track_length=3.0,
        feature_extraction_ms=1.0, feature_matching_ms=1.0, incremental_mapping_ms=1.0,
        total_runtime_ms=3.0, peak_ram_mb=100.0, undistorted=True,
    )

    class _MockDepth:
        def predict(self, rgb):
            h, w = rgb.shape[:2]
            disp = np.full((h, w), 0.5, dtype=np.float32)
            # Small smooth variation so RANSAC 2-pt can fit s and t cleanly
            disp += np.linspace(-0.02, 0.02, w, dtype=np.float32)[None, :]
            return DepthPrediction(
                depth_map=np.full((h, w), 2.0, dtype=np.float32),
                disparity_map=disp,
                raw_prediction=disp,
                model_id="mock",
                model_name="mock",
                license="MIT",
                is_metric=False,
                scale_semantics="relative",
                device="cpu",
                inference_time_ms=1.0,
                peak_memory_mb=10.0,
                vram_allocated_mb=0.0,
            )

    cfg = MultiViewFusionConfig(
        depth_max_dim=120, tsdf_voxel_length=0.03, discontinuity_threshold=0.08, min_consistent_views=1
    )
    res = fuse_multiview_tsdf_and_mesh(sfm_res, _MockDepth(), cfg)
    assert res.num_accepted_views == 3
    assert res.num_rejected_views == 0
    assert res.mesh.num_faces > 100
    assert res.point_cloud.num_points > 100
    assert res.peak_ram_mb > 0.0
