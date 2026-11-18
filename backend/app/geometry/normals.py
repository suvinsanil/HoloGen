"""Surface normal estimation and camera-consistent orientation for 3D point clouds."""

import time
from typing import Optional

import numpy as np
import open3d as o3d

from app.core.exceptions import GeometryProcessingError
from app.geometry.point_cloud import PointCloudData


def estimate_and_orient_normals(
    cloud: PointCloudData,
    max_nn: int = 30,
    radius: Optional[float] = None,
    camera_location: Optional[np.ndarray] = None,
) -> PointCloudData:
    """Estimate local PCA surface normals and orient them consistently toward the camera origin.

    Parameters:
        cloud: Input PointCloudData with at least 3 points.
        max_nn: Maximum nearest neighbors for local covariance PCA (default: 30).
        radius: Search radius in 3D camera units. If None, automatically set to 4x the mean
            nearest-neighbor spacing (or 5% of bounding-box diagonal).
        camera_location: 3D coordinates of the pinhole camera center (default: [0, 0, 0]).

    Returns:
        New PointCloudData instance with unit normals (N, 3) oriented toward `camera_location`.
    """
    t0 = time.perf_counter()
    if cloud.num_points < 3:
        raise GeometryProcessingError(
            f"At least 3 points required for normal estimation, got {cloud.num_points}"
        )
    if max_nn < 3:
        raise GeometryProcessingError(f"max_nn must be >= 3, got {max_nn}")

    pcd = cloud.to_open3d()

    if radius is None:
        dists = np.asarray(pcd.compute_nearest_neighbor_distance(), dtype=np.float64)
        mean_nn = float(np.mean(dists)) if dists.size > 0 else 0.0
        diag = float(np.linalg.norm(cloud.extent))
        search_radius = max(mean_nn * 4.5, diag * 0.02, 1e-3)
    else:
        if radius <= 0.0:
            raise GeometryProcessingError(f"radius must be > 0, got {radius}")
        search_radius = float(radius)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=search_radius,
            max_nn=int(max_nn),
        )
    )

    cam_loc = (
        np.zeros(3, dtype=np.float64)
        if camera_location is None
        else np.asarray(camera_location, dtype=np.float64)
    )
    pcd.orient_normals_towards_camera_location(cam_loc)

    normals = np.asarray(pcd.normals, dtype=np.float32)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return PointCloudData(
        points=cloud.points.copy(),
        colors=cloud.colors.copy(),
        pixel_coords=cloud.pixel_coords.copy(),
        valid_mask=cloud.valid_mask.copy(),
        generation_time_ms=cloud.generation_time_ms + elapsed_ms,
        normals=normals,
    )
