"""Point cloud filtering, outlier removal, downsampling, smoothing, and spatial density statistics."""

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, Optional

import numpy as np
import open3d as o3d

from app.core.exceptions import GeometryProcessingError
from app.geometry.normals import estimate_and_orient_normals
from app.geometry.point_cloud import PointCloudData


@dataclass
class PointCloudDensityStats:
    """Spatial density and nearest-neighbor spacing statistics for a 3D point cloud."""

    num_points: int
    nn_distance_mean: float
    nn_distance_median: float
    nn_distance_std: float
    nn_distance_p5: float
    nn_distance_p95: float
    estimated_surface_density: float  # Approx points per unit area: 1 / (pi * median_nn^2)
    bbox_min: list[float]
    bbox_max: list[float]
    bbox_extent: list[float]
    bbox_diagonal: float
    bbox_volume: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_points": self.num_points,
            "nn_distance_mean": round(self.nn_distance_mean, 5),
            "nn_distance_median": round(self.nn_distance_median, 5),
            "nn_distance_std": round(self.nn_distance_std, 5),
            "nn_distance_p5": round(self.nn_distance_p5, 5),
            "nn_distance_p95": round(self.nn_distance_p95, 5),
            "estimated_surface_density": round(self.estimated_surface_density, 2),
            "bounding_box": {
                "min": [round(v, 4) for v in self.bbox_min],
                "max": [round(v, 4) for v in self.bbox_max],
                "extent": [round(v, 4) for v in self.bbox_extent],
                "diagonal": round(self.bbox_diagonal, 4),
                "volume": round(self.bbox_volume, 4),
            },
        }


@dataclass
class PointCloudFilterResult:
    """Structured result from a point cloud filtering or cleaning operation."""

    cloud: PointCloudData
    method: str
    parameters: Dict[str, Any]
    input_points: int
    output_points: int
    removed_points: int
    retention_ratio: float
    processing_time_ms: float

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "parameters": self.parameters,
            "input_points": self.input_points,
            "output_points": self.output_points,
            "removed_points": self.removed_points,
            "retention_ratio": round(self.retention_ratio, 4),
            "processing_time_ms": round(self.processing_time_ms, 3),
        }


def compute_point_cloud_density_stats(cloud: PointCloudData) -> PointCloudDensityStats:
    """Compute nearest-neighbor spacing distribution, surface density, and bounding box stats."""
    if cloud.num_points == 0:
        raise GeometryProcessingError("Cannot compute density stats on empty point cloud.")

    pcd = cloud.to_open3d()
    if cloud.num_points >= 2:
        dists = np.asarray(pcd.compute_nearest_neighbor_distance(), dtype=np.float64)
        mean_d = float(np.mean(dists))
        med_d = float(np.median(dists))
        std_d = float(np.std(dists))
        p5_d, p95_d = [float(x) for x in np.percentile(dists, [5.0, 95.0])]
        surf_density = float(1.0 / max(math.pi * (med_d ** 2), 1e-9))
    else:
        mean_d = med_d = std_d = p5_d = p95_d = 0.0
        surf_density = 0.0

    ext = cloud.extent.astype(np.float64)
    diag = float(np.linalg.norm(ext))
    vol = float(np.prod(np.maximum(ext, 0.0)))

    return PointCloudDensityStats(
        num_points=cloud.num_points,
        nn_distance_mean=mean_d,
        nn_distance_median=med_d,
        nn_distance_std=std_d,
        nn_distance_p5=p5_d,
        nn_distance_p95=p95_d,
        estimated_surface_density=surf_density,
        bbox_min=[float(v) for v in cloud.min_bound],
        bbox_max=[float(v) for v in cloud.max_bound],
        bbox_extent=[float(v) for v in ext],
        bbox_diagonal=diag,
        bbox_volume=vol,
    )


def _subset_cloud_by_indices(
    cloud: PointCloudData,
    indices: np.ndarray,
    elapsed_ms: float,
) -> PointCloudData:
    """Create a filtered PointCloudData subset preserving points, colors, pixel_coords, and normals."""
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        raise GeometryProcessingError("Filtering removed all points from the point cloud.")

    new_normals = cloud.normals[idx].copy() if cloud.normals is not None else None
    new_pixel_coords = (
        cloud.pixel_coords[idx].copy()
        if len(cloud.pixel_coords) == cloud.num_points
        else np.zeros((idx.size, 2), dtype=np.int32)
    )
    return PointCloudData(
        points=cloud.points[idx].copy(),
        colors=cloud.colors[idx].copy(),
        pixel_coords=new_pixel_coords,
        valid_mask=cloud.valid_mask.copy(),
        generation_time_ms=cloud.generation_time_ms + elapsed_ms,
        normals=new_normals,
    )


def statistical_outlier_removal(
    cloud: PointCloudData,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> PointCloudFilterResult:
    """Remove points whose mean distance to `nb_neighbors` exceeds `mean + std_ratio * std`.

    Parameters:
        cloud: Input PointCloudData.
        nb_neighbors: Number of nearest neighbors analyzed per point (default: 20).
        std_ratio: Standard deviation multiplier threshold; smaller values filter more aggressively (default: 2.0).
    """
    t0 = time.perf_counter()
    if nb_neighbors < 1 or std_ratio <= 0.0:
        raise GeometryProcessingError(
            f"Invalid SOR parameters: nb_neighbors={nb_neighbors}, std_ratio={std_ratio}"
        )
    pcd = cloud.to_open3d()
    _, ind = pcd.remove_statistical_outlier(
        nb_neighbors=int(nb_neighbors),
        std_ratio=float(std_ratio),
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    sub = _subset_cloud_by_indices(cloud, np.asarray(ind, dtype=np.int64), elapsed_ms)
    return PointCloudFilterResult(
        cloud=sub,
        method="statistical_outlier_removal",
        parameters={"nb_neighbors": nb_neighbors, "std_ratio": std_ratio},
        input_points=cloud.num_points,
        output_points=sub.num_points,
        removed_points=cloud.num_points - sub.num_points,
        retention_ratio=float(sub.num_points) / float(cloud.num_points),
        processing_time_ms=elapsed_ms,
    )


def radius_outlier_removal(
    cloud: PointCloudData,
    nb_points: int = 12,
    radius: Optional[float] = None,
) -> PointCloudFilterResult:
    """Remove isolated points that have fewer than `nb_points` neighbors within `radius`.

    Parameters:
        cloud: Input PointCloudData.
        nb_points: Minimum number of neighbor points required inside the sphere (default: 12).
        radius: Sphere radius in 3D camera units. If None, defaults to 3.5x mean NN spacing.
    """
    t0 = time.perf_counter()
    if nb_points < 1:
        raise GeometryProcessingError(f"nb_points must be >= 1, got {nb_points}")
    pcd = cloud.to_open3d()

    if radius is None:
        dists = np.asarray(pcd.compute_nearest_neighbor_distance(), dtype=np.float64)
        r = max(float(np.mean(dists)) * 3.5, 0.02)
    else:
        if radius <= 0.0:
            raise GeometryProcessingError(f"radius must be > 0, got {radius}")
        r = float(radius)

    _, ind = pcd.remove_radius_outlier(nb_points=int(nb_points), radius=r)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    sub = _subset_cloud_by_indices(cloud, np.asarray(ind, dtype=np.int64), elapsed_ms)
    return PointCloudFilterResult(
        cloud=sub,
        method="radius_outlier_removal",
        parameters={"nb_points": nb_points, "radius": round(r, 5)},
        input_points=cloud.num_points,
        output_points=sub.num_points,
        removed_points=cloud.num_points - sub.num_points,
        retention_ratio=float(sub.num_points) / float(cloud.num_points),
        processing_time_ms=elapsed_ms,
    )


def voxel_downsample_cloud(
    cloud: PointCloudData,
    voxel_size: float = 0.025,
) -> PointCloudFilterResult:
    """Downsample a point cloud on a regular 3D voxel grid of size `voxel_size`.

    Parameters:
        cloud: Input PointCloudData.
        voxel_size: Edge length of each cubic 3D voxel in camera space units (default: 0.025).
    """
    t0 = time.perf_counter()
    if voxel_size <= 0.0:
        raise GeometryProcessingError(f"voxel_size must be > 0, got {voxel_size}")

    pcd = cloud.to_open3d()
    down_pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    pts = np.asarray(down_pcd.points, dtype=np.float32)
    if pts.shape[0] == 0:
        raise GeometryProcessingError("Voxel downsampling produced 0 points.")
    cols = np.asarray(down_pcd.colors, dtype=np.float32)
    norms = (
        np.asarray(down_pcd.normals, dtype=np.float32)
        if down_pcd.has_normals()
        else None
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    new_cloud = PointCloudData(
        points=pts,
        colors=cols,
        pixel_coords=np.zeros((pts.shape[0], 2), dtype=np.int32),
        valid_mask=cloud.valid_mask.copy(),
        generation_time_ms=cloud.generation_time_ms + elapsed_ms,
        normals=norms,
    )
    return PointCloudFilterResult(
        cloud=new_cloud,
        method="voxel_downsample",
        parameters={"voxel_size": voxel_size},
        input_points=cloud.num_points,
        output_points=new_cloud.num_points,
        removed_points=cloud.num_points - new_cloud.num_points,
        retention_ratio=float(new_cloud.num_points) / float(cloud.num_points),
        processing_time_ms=elapsed_ms,
    )


def normal_grazing_angle_filter(
    cloud: PointCloudData,
    max_grazing_angle_deg: float = 82.0,
    normal_max_nn: int = 30,
) -> PointCloudFilterResult:
    """Remove 'flying veil' points along depth discontinuities whose surface normal is nearly
    orthogonal to the camera viewing ray.

    Parameters:
        cloud: Input PointCloudData (normals will be estimated if not present).
        max_grazing_angle_deg: Maximum allowed angle in degrees between the outward surface
            normal n_i and the camera viewing vector -p_i / ||p_i|| (default: 82.0 deg).
        normal_max_nn: k-NN neighborhood size if normals need to be computed.
    """
    t0 = time.perf_counter()
    if not (10.0 <= max_grazing_angle_deg < 90.0):
        raise GeometryProcessingError(
            f"max_grazing_angle_deg must be in [10, 90), got {max_grazing_angle_deg}"
        )

    c_norm = (
        cloud
        if cloud.normals is not None
        else estimate_and_orient_normals(cloud, max_nn=normal_max_nn)
    )
    normals = c_norm.normals
    assert normals is not None

    # Viewing ray from point p_i toward camera origin (0,0,0) is v_i = -p_i / ||p_i||
    view_vecs = -c_norm.points
    view_norms = np.linalg.norm(view_vecs, axis=1, keepdims=True)
    view_unit = view_vecs / np.maximum(view_norms, 1e-8)

    cos_theta = np.sum(normals * view_unit, axis=1)
    min_cos = math.cos(math.radians(float(max_grazing_angle_deg)))
    keep_idx = np.where(cos_theta >= min_cos)[0]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    sub = _subset_cloud_by_indices(c_norm, keep_idx, elapsed_ms)
    return PointCloudFilterResult(
        cloud=sub,
        method="normal_grazing_angle_filter",
        parameters={
            "max_grazing_angle_deg": max_grazing_angle_deg,
            "min_cos_theta": round(min_cos, 4),
        },
        input_points=cloud.num_points,
        output_points=sub.num_points,
        removed_points=cloud.num_points - sub.num_points,
        retention_ratio=float(sub.num_points) / float(cloud.num_points),
        processing_time_ms=elapsed_ms,
    )


def smooth_point_cloud_tangent_plane(
    cloud: PointCloudData,
    alpha: float = 0.35,
    normal_max_nn: int = 30,
) -> PointCloudFilterResult:
    """Smooth high-frequency depth jitter along local surface normal directions using vectorized
    2D grid neighborhood averaging (for structured clouds) or KD-tree neighbor projection.

    Parameters:
        cloud: Input PointCloudData.
        alpha: Damping step size in (0, 1] for normal-direction projection toward local centroid.
        normal_max_nn: Neighborhood count for normal estimation if normals are absent.
    """
    t0 = time.perf_counter()
    if not (0.0 < alpha <= 1.0):
        raise GeometryProcessingError(f"alpha must be in (0, 1], got {alpha}")

    c_norm = (
        cloud
        if cloud.normals is not None
        else estimate_and_orient_normals(cloud, max_nn=normal_max_nn)
    )
    pts = c_norm.points.copy()
    normals = c_norm.normals
    assert normals is not None

    # Use SciPy cKDTree for fast multi-threaded k-NN centroid query
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    k = min(9, pts.shape[0])
    _, nn_idx = tree.query(pts, k=k, workers=-1)
    if k > 1:
        local_centroids = np.mean(pts[nn_idx], axis=1)
        diff = local_centroids - pts
        # Project displacement onto surface normal so points only move to flatten normal noise
        proj_mag = np.sum(diff * normals, axis=1, keepdims=True)
        pts = (pts + float(alpha) * proj_mag * normals).astype(np.float32)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    smoothed_cloud = PointCloudData(
        points=pts,
        colors=c_norm.colors.copy(),
        pixel_coords=c_norm.pixel_coords.copy(),
        valid_mask=c_norm.valid_mask.copy(),
        generation_time_ms=c_norm.generation_time_ms + elapsed_ms,
        normals=normals.copy(),
    )
    # Re-estimate normals on smoothed geometry
    smoothed_cloud = estimate_and_orient_normals(smoothed_cloud, max_nn=normal_max_nn)

    return PointCloudFilterResult(
        cloud=smoothed_cloud,
        method="tangent_plane_normal_smoothing",
        parameters={"alpha": alpha, "k_neighbors": k},
        input_points=cloud.num_points,
        output_points=smoothed_cloud.num_points,
        removed_points=0,
        retention_ratio=1.0,
        processing_time_ms=elapsed_ms,
    )
