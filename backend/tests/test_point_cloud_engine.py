"""Unit tests for Phase 3 point cloud engine: filtering, normals, density stats, and smoothing."""

import numpy as np
import pytest

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


def _make_planar_cloud(width: int = 24, height: int = 24, z_val: float = 3.0) -> PointCloudData:
    intr = estimate_intrinsics(width=width, height=height, focal_length_px=20.0)
    depth = np.full((height, width), z_val, dtype=np.float32)
    rgb = np.full((height, width, 3), 180, dtype=np.uint8)
    return backproject_depth_to_point_cloud(depth, rgb, intr)


def test_analytical_plane_normals_face_camera_origin() -> None:
    """Hand-computed check:
    For a fronto-parallel plane Z = 3.0 in front of camera at (0, 0, 0),
    the unit surface normal oriented toward the camera must be [0.0, 0.0, -1.0].
    """
    cloud = _make_planar_cloud(width=20, height=20, z_val=3.0)
    with_normals = estimate_and_orient_normals(cloud, max_nn=20)
    assert with_normals.normals is not None
    assert with_normals.normals.shape == (400, 3)

    # Every normal must have unit length and point in -Z direction ([0, 0, -1])
    lengths = np.linalg.norm(with_normals.normals, axis=1)
    np.testing.assert_allclose(lengths, np.ones(400), atol=1e-5)
    np.testing.assert_allclose(with_normals.normals[:, 0], 0.0, atol=1e-4)
    np.testing.assert_allclose(with_normals.normals[:, 1], 0.0, atol=1e-4)
    np.testing.assert_allclose(with_normals.normals[:, 2], -1.0, atol=1e-4)


def test_statistical_and_radius_outlier_removal_prune_synthetic_spikes() -> None:
    """Verify SOR and ROR remove isolated far-away outlier points while preserving the plane."""
    base = _make_planar_cloud(width=20, height=20, z_val=2.0)
    outliers = np.array(
        [
            [50.0, 50.0, 50.0],
            [-40.0, 30.0, 60.0],
            [0.0, -45.0, 80.0],
        ],
        dtype=np.float32,
    )
    outlier_colors = np.ones((3, 3), dtype=np.float32)
    outlier_coords = np.zeros((3, 2), dtype=np.int32)

    corrupted = PointCloudData(
        points=np.vstack([base.points, outliers]),
        colors=np.vstack([base.colors, outlier_colors]),
        pixel_coords=np.vstack([base.pixel_coords, outlier_coords]),
        valid_mask=base.valid_mask,
        generation_time_ms=base.generation_time_ms,
    )
    assert corrupted.num_points == 403

    sor_res = statistical_outlier_removal(corrupted, nb_neighbors=15, std_ratio=2.0)
    assert sor_res.output_points == 400
    assert sor_res.removed_points == 3
    assert float(np.max(sor_res.cloud.points[:, 2])) == pytest.approx(2.0, abs=1e-5)

    ror_res = radius_outlier_removal(corrupted, nb_points=5, radius=0.8)
    assert ror_res.output_points == 400
    assert ror_res.removed_points == 3


def test_voxel_downsampling_and_density_statistics() -> None:
    """Verify voxel downsampling reduces dense point count and density stats are positive."""
    cloud = _make_planar_cloud(width=40, height=40, z_val=2.0)
    stats_before = compute_point_cloud_density_stats(cloud)
    assert stats_before.num_points == 1600
    assert stats_before.nn_distance_median == pytest.approx(0.1, rel=1e-3)

    down_res = voxel_downsample_cloud(cloud, voxel_size=0.25)
    assert 0 < down_res.output_points < 600
    assert down_res.retention_ratio < 0.4


def test_normal_grazing_filter_and_tangent_smoothing() -> None:
    """Verify grazing filter removes steep depth-ramp veil points and smoothing reduces Z jitter."""
    rng = np.random.default_rng(7)
    intr = estimate_intrinsics(width=30, height=30, focal_length_px=30.0)
    depth = np.full((30, 30), 2.0, dtype=np.float32)
    # Create a steep tilted ramp on columns 15..29 where adjacent columns are within normal search radius
    for col_idx in range(15, 30):
        depth[:, col_idx] = 2.0 + 0.18 * float(col_idx - 14)
    rgb = np.full((30, 30, 3), 200, dtype=np.uint8)

    cloud = backproject_depth_to_point_cloud(depth, rgb, intr)
    grazing_res = normal_grazing_angle_filter(cloud, max_grazing_angle_deg=45.0)
    # Points along the steep tilted surface have grazing normals (>45 deg from viewing ray) and get removed
    assert grazing_res.removed_points > 0
    assert grazing_res.output_points < cloud.num_points

    # Test tangent smoothing on a noisy plane
    noisy_plane_d = np.full((24, 24), 3.0, dtype=np.float32) + rng.normal(
        0.0, 0.02, size=(24, 24)
    ).astype(np.float32)
    noisy_cloud = backproject_depth_to_point_cloud(
        noisy_plane_d, np.full((24, 24, 3), 150, dtype=np.uint8), estimate_intrinsics(24, 24, focal_length_px=24.0)
    )
    std_before = float(np.std(noisy_cloud.points[:, 2]))
    smooth_res = smooth_point_cloud_tangent_plane(noisy_cloud, alpha=0.5)
    std_after = float(np.std(smooth_res.cloud.points[:, 2]))
    assert std_after < std_before
