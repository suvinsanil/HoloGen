"""Unit tests for Phase 7b: Sim(3) Umeyama alignment (with hand-computed values) and
N-view incremental Structure-from-Motion via pycolmap with single shared camera model.
"""

import numpy as np
import pytest

from app.core.config import get_settings
from app.evaluation.multiview_datasets import load_fountain_dataset
from app.vision.sfm import (
    align_umeyama_sim3,
    evaluate_sfm_poses_against_reference,
    run_incremental_sfm,
)


def test_hand_computed_umeyama_sim3_alignment() -> None:
    """Hand-computed check for Sim(3) Umeyama alignment:
    Source points:
      p0 = (0, 0, 0), p1 = (1, 0, 0), p2 = (0, 2, 0), p3 = (0, 0, 3)
    True transformation:
      scale s = 2.5
      R = 90 deg rotation around Z axis: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
      t = [10.0, -5.0, 3.0]
    Target points y_i = 2.5 * R @ p_i + t:
      y0 = (10.0, -5.0, 3.0)
      y1 = (10.0, -2.5, 3.0)
      y2 = ( 5.0, -5.0, 3.0)
      y3 = (10.0, -5.0, 10.5)
    """
    src = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=np.float64,
    )
    dst = np.array(
        [
            [10.0, -5.0, 3.0],
            [10.0, -2.5, 3.0],
            [5.0, -5.0, 3.0],
            [10.0, -5.0, 10.5],
        ],
        dtype=np.float64,
    )
    R_expected = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    t_expected = np.array([10.0, -5.0, 3.0], dtype=np.float64)

    res = align_umeyama_sim3(src, dst, with_scale=True)
    assert res.scale == pytest.approx(2.5, rel=1e-12)
    np.testing.assert_allclose(res.R, R_expected, atol=1e-12)
    np.testing.assert_allclose(res.t, t_expected, atol=1e-12)
    np.testing.assert_allclose(res.aligned_points, dst, atol=1e-12)
    assert res.ate_rmse == pytest.approx(0.0, abs=1e-12)
    assert float(np.linalg.det(res.R)) == pytest.approx(1.0, abs=1e-12)


def test_umeyama_sim3_prevents_improper_reflection() -> None:
    """Verify Umeyama Sim(3) enforces det(R) = +1 even when destination points are reflected."""
    src = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )
    # Mirror Z coordinate (det = -1 reflection)
    dst_reflected = src * np.array([1.0, 1.0, -1.0])
    res = align_umeyama_sim3(src, dst_reflected, with_scale=True)
    assert float(np.linalg.det(res.R)) == pytest.approx(1.0, abs=1e-12)


def test_incremental_sfm_single_shared_camera_on_fountain_subset() -> None:
    """Run pycolmap incremental SfM on 5 Fountain views with single shared camera model."""
    settings = get_settings()
    views, _ = load_fountain_dataset(settings.datasets_dir)
    subset = views[:5]
    img_paths = [v.image_path for v in subset]
    ref_map = {v.image_path.name: v for v in subset}

    sfm_res = run_incremental_sfm(
        image_paths=img_paths,
        sfm_max_image_size=640,
        camera_model="SIMPLE_RADIAL",
        single_camera=True,
        max_num_features=2048,
        num_threads=4,
    )

    assert sfm_res.num_input_images == 5
    assert sfm_res.num_registered_images == 5
    assert sfm_res.registration_ratio == pytest.approx(1.0)
    assert sfm_res.single_camera_mode is True
    assert sfm_res.shared_intrinsics is not None
    assert sfm_res.num_sparse_points > 200
    assert sfm_res.mean_reprojection_error_px < 1.5
    assert sfm_res.total_runtime_ms > 0.0
    assert sfm_res.peak_ram_mb > 50.0

    # Verify all registered views share camera_id == 1
    cam_ids = {pose.camera_id for pose in sfm_res.registered_poses.values()}
    assert len(cam_ids) == 1

    # Verify Sim(3) Umeyama trajectory evaluation against reference RGB-D trajectory
    pose_metrics, sim3 = evaluate_sfm_poses_against_reference(sfm_res, ref_map)
    assert pose_metrics["num_aligned_cameras"] == 5
    assert pose_metrics["camera_center_ate_rmse"] < 0.05
    # Per-camera rotation accuracy, measured independent of centre-based alignment conditioning.
    assert pose_metrics["rotation_error_after_best_global_rotation_mean_deg"] < 2.0
    assert pose_metrics["pairwise_relative_rotation_error_mean_deg"] < 2.0
    # Centre-based Umeyama rotation includes alignment error on a near-collinear trajectory: sanity bound only.
    assert pose_metrics["aligned_rotation_error_mean_deg"] < 5.0
    assert sim3.scale > 0.0


def test_best_global_rotation_recovers_known_frame_rotation() -> None:
    """If R_sfm_i = R_ref_i @ R_g for a fixed R_g (90 deg about Z), the solver must return R_g exactly."""
    from app.vision.sfm import best_global_rotation, trajectory_sv_ratio

    R_g = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    c, s = np.cos(0.3), np.sin(0.3)
    R_ref = [
        np.eye(3),
        np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]),
        np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]),
    ]
    R_sfm = [R @ R_g for R in R_ref]
    np.testing.assert_allclose(best_global_rotation(R_sfm, R_ref), R_g, atol=1e-12)

    # Hand-computed: perfectly collinear centres -> sigma_2 / sigma_1 = 0
    assert trajectory_sv_ratio(np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])) == pytest.approx(0.0, abs=1e-12)
