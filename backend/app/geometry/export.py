"""Export utilities for point clouds (PLY) and depth visualizations (PNG)."""

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import open3d as o3d

from app.core.exceptions import ExportError
from app.geometry.point_cloud import PointCloudData


def render_depth_visualization(
    depth_or_disparity: np.ndarray,
    mode: Literal["heatmap", "grayscale", "inverted"] = "heatmap",
    colormap: int = cv2.COLORMAP_INFERNO,
) -> np.ndarray:
    """Render a 2D depth or disparity map to an HxWx3 RGB uint8 image."""
    arr = np.asarray(depth_or_disparity, dtype=np.float32)
    if arr.ndim != 2:
        raise ExportError(f"Expected 2D array for depth visualization, got shape {arr.shape}")

    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)

    vmin = float(np.min(arr[finite_mask]))
    vmax = float(np.max(arr[finite_mask]))
    if vmax - vmin < 1e-8:
        norm = np.zeros_like(arr, dtype=np.float32)
    else:
        norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)

    if mode == "inverted":
        norm = 1.0 - norm

    gray_u8 = np.round(norm * 255.0).astype(np.uint8)
    if mode in ("grayscale", "inverted"):
        return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)

    bgr = cv2.applyColorMap(gray_u8, colormap)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def save_depth_visualization_png(
    depth_or_disparity: np.ndarray,
    output_path: Path,
    mode: Literal["heatmap", "grayscale", "inverted"] = "heatmap",
) -> Path:
    """Save a visualized depth/disparity map to a PNG file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = render_depth_visualization(depth_or_disparity, mode=mode)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(output_path), bgr)
    if not ok or not output_path.exists():
        raise ExportError(f"Failed to write depth visualization PNG to {output_path}")
    return output_path


def save_point_cloud_ply(
    cloud: PointCloudData,
    output_path: Path,
    write_ascii: bool = False,
) -> Path:
    """Save a PointCloudData object to a standard PLY file via Open3D."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cloud.num_points == 0:
        raise ExportError("Cannot export an empty point cloud to PLY.")

    pcd = cloud.to_open3d()
    ok = o3d.io.write_point_cloud(
        str(output_path),
        pcd,
        write_ascii=write_ascii,
        compressed=False,
    )
    if not ok or not output_path.exists():
        raise ExportError(f"Failed to write PLY point cloud to {output_path}")
    return output_path
