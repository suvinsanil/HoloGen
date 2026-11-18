"""Unit tests for 3D point cloud back-projection, disparity conversion, and PLY export."""

import numpy as np
import open3d as o3d
import pytest

from app.geometry.export import save_depth_visualization_png, save_point_cloud_ply
from app.geometry.point_cloud import backproject_depth_to_point_cloud
from app.vision.camera import estimate_intrinsics
from app.vision.depth_estimation import disparity_to_relative_depth


def test_hand_computed_2x2_backprojection() -> None:
    """Hand-computed 2x2 grid back-projection test:
    Let W=2, H=2, fx=2.0, fy=2.0, principal point=(0.5, 0.5).
    Pixels (u, v) in {(0,0), (1,0), (0,1), (1,1)}:
    Depth D:
      D(0,0) = 2.0 -> X = (0 - 0.5)*2.0/2.0 = -0.5, Y = (0 - 0.5)*2.0/2.0 = -0.5, Z = 2.0
      D(1,0) = 4.0 -> X = (1 - 0.5)*4.0/2.0 = +1.0, Y = (0 - 0.5)*4.0/2.0 = -1.0, Z = 4.0
      D(0,1) = 6.0 -> X = (0 - 0.5)*6.0/2.0 = -1.5, Y = (1 - 0.5)*6.0/2.0 = +1.5, Z = 6.0
      D(1,1) = 8.0 -> X = (1 - 0.5)*8.0/2.0 = +2.0, Y = (1 - 0.5)*8.0/2.0 = +2.0, Z = 8.0
    """
    intr = estimate_intrinsics(
        width=2,
        height=2,
        focal_length_px=2.0,
        principal_point=(0.5, 0.5),
    )
    depth = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ],
        dtype=np.uint8,
    )

    cloud = backproject_depth_to_point_cloud(depth_map=depth, rgb_image=rgb, intrinsics=intr)
    assert cloud.num_points == 4

    expected_points = np.array(
        [
            [-0.5, -0.5, 2.0],
            [1.0, -1.0, 4.0],
            [-1.5, 1.5, 6.0],
            [2.0, 2.0, 8.0],
        ],
        dtype=np.float32,
    )
    expected_colors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(cloud.points, expected_points, atol=1e-6)
    np.testing.assert_allclose(cloud.colors, expected_colors, atol=1e-6)


def test_disparity_to_relative_depth_ordering() -> None:
    """Verify higher disparity (foreground) maps to smaller depth Z, within [near, far]."""
    disp = np.linspace(10.0, 100.0, 100, dtype=np.float32).reshape(10, 10)
    depth, norm_disp = disparity_to_relative_depth(disp, near_depth=1.0, far_depth=10.0)

    # Smallest disparity (background) -> far_depth (10.0)
    # Largest disparity (foreground) -> near_depth (1.0)
    assert float(depth[0, 0]) == pytest.approx(10.0, rel=1e-2)
    assert float(depth[-1, -1]) == pytest.approx(1.0, rel=1e-2)
    assert float(norm_disp[0, 0]) == pytest.approx(0.0, abs=1e-2)
    assert float(norm_disp[-1, -1]) == pytest.approx(1.0, abs=1e-2)
    assert np.all(depth >= 1.0 - 1e-4) and np.all(depth <= 10.0 + 1e-4)


def test_ply_and_png_export_roundtrip(tmp_path) -> None:
    """Verify exported PLY and depth PNG can be read back with exact point count."""
    intr = estimate_intrinsics(width=16, height=12, fov_deg=60.0)
    depth = np.full((12, 16), 3.0, dtype=np.float32)
    rgb = np.full((12, 16, 3), 128, dtype=np.uint8)

    cloud = backproject_depth_to_point_cloud(depth, rgb, intr)
    ply_path = save_point_cloud_ply(cloud, tmp_path / "test.ply")
    png_path = save_depth_visualization_png(depth, tmp_path / "depth.png", mode="heatmap")

    assert ply_path.exists() and ply_path.stat().st_size > 100
    assert png_path.exists() and png_path.stat().st_size > 50

    loaded_pcd = o3d.io.read_point_cloud(str(ply_path))
    assert len(loaded_pcd.points) == 16 * 12
    assert len(loaded_pcd.colors) == 16 * 12
