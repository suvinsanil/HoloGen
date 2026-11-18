"""Depth map back-projection to colored 3D point clouds."""

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional

import numpy as np
import open3d as o3d

from app.core.exceptions import GeometryProcessingError
from app.vision.camera import CameraIntrinsics


@dataclass
class PointCloudData:
    """Container for a 3D colored point cloud and associated spatial metadata."""

    points: np.ndarray  # float32 (N, 3) [X, Y, Z] in camera coordinate frame
    colors: np.ndarray  # float32 (N, 3) [R, G, B] in [0.0, 1.0]
    pixel_coords: np.ndarray  # int32 (N, 2) [u, v] source pixel coordinates
    valid_mask: np.ndarray  # bool (H, W) mask of valid back-projected pixels
    generation_time_ms: float
    normals: Optional[np.ndarray] = None  # float32 (N, 3) unit normals if computed

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def min_bound(self) -> np.ndarray:
        if self.num_points == 0:
            return np.zeros(3, dtype=np.float32)
        return np.min(self.points, axis=0)

    @property
    def max_bound(self) -> np.ndarray:
        if self.num_points == 0:
            return np.zeros(3, dtype=np.float32)
        return np.max(self.points, axis=0)

    @property
    def centroid(self) -> np.ndarray:
        if self.num_points == 0:
            return np.zeros(3, dtype=np.float32)
        return np.mean(self.points, axis=0)

    @property
    def extent(self) -> np.ndarray:
        return self.max_bound - self.min_bound

    def to_open3d(self) -> o3d.geometry.PointCloud:
        """Convert to an Open3D PointCloud instance."""
        pcd = o3d.geometry.PointCloud()
        if self.num_points > 0:
            pcd.points = o3d.utility.Vector3dVector(self.points.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(
                np.clip(self.colors, 0.0, 1.0).astype(np.float64)
            )
            if self.normals is not None and len(self.normals) == self.num_points:
                pcd.normals = o3d.utility.Vector3dVector(self.normals.astype(np.float64))
        return pcd

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Serialize point cloud statistics for reporting."""
        return {
            "num_points": self.num_points,
            "has_normals": self.normals is not None,
            "bounding_box": {
                "min": [round(float(x), 4) for x in self.min_bound],
                "max": [round(float(x), 4) for x in self.max_bound],
                "extent": [round(float(x), 4) for x in self.extent],
                "centroid": [round(float(x), 4) for x in self.centroid],
            },
            "generation_time_ms": round(self.generation_time_ms, 3),
        }


def backproject_depth_to_point_cloud(
    depth_map: np.ndarray,
    rgb_image: np.ndarray,
    intrinsics: CameraIntrinsics,
    min_depth: float = 1e-4,
    max_depth: float = 1e4,
    stride: int = 1,
    confidence_mask: Optional[np.ndarray] = None,
) -> PointCloudData:
    """Back-project a 2D depth map and aligned RGB image into a 3D colored point cloud.

    Mathematical formulation (pinhole camera model):
        Z = D(u, v)
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

    Args:
        depth_map: 2D array (H, W) of positive depth values Z.
        rgb_image: 3D array (H, W, 3) of RGB colors (uint8 [0, 255] or float [0.0, 1.0]).
        intrinsics: CameraIntrinsics with fx, fy, cx, cy matching (W, H).
        min_depth: Minimum valid depth value (> 0).
        max_depth: Maximum valid depth value.
        stride: Pixel subsampling stride (>= 1).
        confidence_mask: Optional boolean mask (H, W) filtering out uncertain pixels.
    """
    t0 = time.perf_counter()
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim != 2:
        raise GeometryProcessingError(f"depth_map must be 2D (H, W), got shape {depth.shape}")

    h, w = depth.shape
    if intrinsics.width != w or intrinsics.height != h:
        raise GeometryProcessingError(
            f"Intrinsics dimensions ({intrinsics.width}x{intrinsics.height}) do not match "
            f"depth_map dimensions ({w}x{h})."
        )

    rgb = np.asarray(rgb_image)
    if rgb.ndim != 3 or rgb.shape[0] != h or rgb.shape[1] != w or rgb.shape[2] != 3:
        raise GeometryProcessingError(
            f"rgb_image shape {rgb.shape} does not match depth_map ({h}, {w}, 3)."
        )

    if stride < 1:
        raise GeometryProcessingError(f"stride must be >= 1, got {stride}")

    if rgb.dtype == np.uint8:
        colors_norm = rgb.astype(np.float32) / 255.0
    else:
        colors_norm = np.clip(rgb.astype(np.float32), 0.0, 1.0)

    valid_mask = np.isfinite(depth) & (depth > float(min_depth)) & (depth <= float(max_depth))
    if confidence_mask is not None:
        if confidence_mask.shape != (h, w):
            raise GeometryProcessingError(
                f"confidence_mask shape {confidence_mask.shape} != ({h}, {w})"
            )
        valid_mask = valid_mask & confidence_mask.astype(bool)

    if stride > 1:
        stride_mask = np.zeros((h, w), dtype=bool)
        stride_mask[::stride, ::stride] = True
        valid_mask = valid_mask & stride_mask

    u_grid, v_grid = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
        indexing="xy",
    )

    u_valid = u_grid[valid_mask]
    v_valid = v_grid[valid_mask]
    z_valid = depth[valid_mask]

    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    cx = float(intrinsics.cx)
    cy = float(intrinsics.cy)

    x_valid = (u_valid - cx) * z_valid / fx
    y_valid = (v_valid - cy) * z_valid / fy

    points = np.stack([x_valid, y_valid, z_valid], axis=-1).astype(np.float32)
    colors = colors_norm[valid_mask].astype(np.float32)
    pixel_coords = np.stack([u_valid.astype(np.int32), v_valid.astype(np.int32)], axis=-1)

    if points.shape[0] == 0:
        raise GeometryProcessingError(
            "Back-projection produced 0 valid points; check depth range or mask."
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return PointCloudData(
        points=points,
        colors=colors,
        pixel_coords=pixel_coords,
        valid_mask=valid_mask,
        generation_time_ms=elapsed_ms,
    )
