"""Phase 7c: Sparse-aligned monocular depth, multi-view consistency filtering, and sequential TSDF fusion.

Designed for 7.7 GB RAM / 2 GB VRAM:
- Processes views sequentially (one view at a time), caching intermediate per-view depth maps to
  temporary disk storage and freeing arrays between views (`gc.collect()`).
- Fits affine inverse-depth scale & shift (1/Z = s * d_mono + t) to SfM sparse 3D points via RANSAC.
  Falls back to scale-only (t = 0) if inlier ratio is low or shift produces non-positive disparity,
  and rejects views that still fail quality checks.
- Masks high depth-discontinuity pixels before fusion.
- Filters pixels via configurable multi-view geometric & photometric consistency checks across
  neighboring views, reporting exact rejection counts.
- Fuses into Open3D ScalableTSDFVolume, extracts mesh + point cloud, and applies view-angle-weighted
  multi-image color blending.
"""

from dataclasses import dataclass, field
import gc
import math
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import psutil

from app.core.exceptions import GeometryProcessingError
from app.core.logging import get_logger
from app.geometry.mesh import MeshData
from app.geometry.point_cloud import PointCloudData
from app.vision.camera import CameraIntrinsics
from app.vision.depth_estimation import DepthModel
from app.vision.sfm import SfmCameraPose, SfmReconstructionResult
from app.vision.uncertainty import compute_uncertainty_proxy

logger = get_logger(__name__)


@dataclass
class MultiViewFusionConfig:
    """Configurable thresholds for Phase 7c sparse alignment, consistency filtering, and TSDF fusion."""

    depth_max_dim: int = 640
    min_sparse_points: int = 10
    ransac_iterations: int = 300
    ransac_rel_depth_thresh: float = 0.15
    min_inlier_ratio: float = 0.25
    min_fallback_inlier_ratio: float = 0.15
    num_neighbor_views: int = 4
    min_consistent_views: int = 1
    consistency_rel_depth_thresh: float = 0.10
    consistency_rgb_thresh: float = 0.18  # in [0, 1] (~46 / 255)
    discontinuity_threshold: float = 0.08  # physical ||grad(log Z)||_2 per-pixel jump cutoff (~8%)
    tsdf_voxel_length: Optional[float] = None  # auto-scaled from scene extent if None
    tsdf_sdf_trunc_mult: float = 4.0
    max_depth_percentile_clip: float = 98.0
    seed: int = 42


@dataclass
class ViewFusionStats:
    """Per-view alignment and consistency filtering telemetry."""

    image_name: str
    fit_mode: str  # "scale_shift_ransac", "scale_only_fallback", "rejected"
    rejected: bool
    rejection_reason: Optional[str]
    scale: float
    shift: float
    num_sparse_points: int
    inlier_ratio: float
    sparse_abs_rel: float
    sparse_rmse: float
    total_pixels: int
    rejected_discontinuity_pixels: int
    rejected_consistency_pixels: int
    valid_fused_pixels: int
    heldout_abs_rel: Optional[float] = None
    heldout_inlier_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_name": self.image_name,
            "fit_mode": self.fit_mode,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
            "scale": round(float(self.scale), 6),
            "shift": round(float(self.shift), 6),
            "num_sparse_points": self.num_sparse_points,
            "inlier_ratio": round(float(self.inlier_ratio), 4),
            "in_sample_abs_rel": round(float(self.sparse_abs_rel), 4),
            "sparse_abs_rel": round(float(self.sparse_abs_rel), 4),
            "sparse_rmse": round(float(self.sparse_rmse), 4),
            "heldout_abs_rel": round(float(self.heldout_abs_rel), 4) if self.heldout_abs_rel is not None else None,
            "heldout_inlier_ratio": round(float(self.heldout_inlier_ratio), 4) if self.heldout_inlier_ratio is not None else None,
            "total_pixels": self.total_pixels,
            "rejected_discontinuity_pixels": self.rejected_discontinuity_pixels,
            "rejected_consistency_pixels": self.rejected_consistency_pixels,
            "valid_fused_pixels": self.valid_fused_pixels,
        }


@dataclass
class MultiViewFusionResult:
    """Complete output of Phase 7c sequential dense multi-view fusion."""

    point_cloud: PointCloudData
    mesh: MeshData
    view_stats: List[ViewFusionStats]
    num_registered_views: int
    num_accepted_views: int
    num_rejected_views: int
    num_scale_only_fallbacks: int
    total_pixels_evaluated: int
    total_rejected_discontinuity_pixels: int
    total_rejected_consistency_pixels: int
    total_valid_fused_pixels: int
    tsdf_voxel_length: float
    depth_and_align_ms: float
    consistency_and_tsdf_ms: float
    mesh_and_blend_ms: float
    total_runtime_ms: float
    peak_ram_mb: float
    peak_ram_above_baseline_mb: float
    config: Dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "num_registered_views": self.num_registered_views,
            "num_accepted_views": self.num_accepted_views,
            "num_rejected_views": self.num_rejected_views,
            "num_scale_only_fallbacks": self.num_scale_only_fallbacks,
            "total_pixels_evaluated": self.total_pixels_evaluated,
            "total_rejected_discontinuity_pixels": self.total_rejected_discontinuity_pixels,
            "total_rejected_consistency_pixels": self.total_rejected_consistency_pixels,
            "total_valid_fused_pixels": self.total_valid_fused_pixels,
            "rejected_discontinuity_pct": round(
                100.0 * self.total_rejected_discontinuity_pixels / max(self.total_pixels_evaluated, 1), 2
            ),
            "rejected_consistency_pct": round(
                100.0 * self.total_rejected_consistency_pixels / max(self.total_pixels_evaluated, 1), 2
            ),
            "valid_fused_pct": round(
                100.0 * self.total_valid_fused_pixels / max(self.total_pixels_evaluated, 1), 2
            ),
            "tsdf_voxel_length": round(self.tsdf_voxel_length, 5),
            "fused_points": self.point_cloud.num_points,
            "mesh_vertices": self.mesh.num_vertices,
            "mesh_faces": self.mesh.num_faces,
            "depth_and_align_ms": round(self.depth_and_align_ms, 2),
            "consistency_and_tsdf_ms": round(self.consistency_and_tsdf_ms, 2),
            "mesh_and_blend_ms": round(self.mesh_and_blend_ms, 2),
            "total_runtime_ms": round(self.total_runtime_ms, 2),
            "peak_ram_mb": round(self.peak_ram_mb, 2),
            "peak_ram_above_baseline_mb": round(self.peak_ram_above_baseline_mb, 2),
            "config": self.config,
            "per_view_stats": [v.to_dict() for v in self.view_stats],
        }


def _scale_intrinsics_to_size(intr: CameraIntrinsics, target_w: int, target_h: int) -> CameraIntrinsics:
    """Scale pinhole intrinsics from SfM image size to depth-inference image size."""
    sx = float(target_w) / float(intr.width)
    sy = float(target_h) / float(intr.height)
    fx = intr.fx * sx
    fy = intr.fy * sy
    cx = intr.cx * sx
    cy = intr.cy * sy
    fov_x = math.degrees(2.0 * math.atan(target_w / (2.0 * max(fx, 1e-6))))
    fov_y = math.degrees(2.0 * math.atan(target_h / (2.0 * max(fy, 1e-6))))
    return CameraIntrinsics(
        width=target_w,
        height=target_h,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        source=f"{intr.source}_scaled",
        fov_x_deg=fov_x,
        fov_y_deg=fov_y,
    )


def _fit_inverse_depth_ransac(
    d_obs: np.ndarray,
    z_obs: np.ndarray,
    full_disparity_map: Optional[np.ndarray],
    cfg: MultiViewFusionConfig,
) -> Tuple[Optional[float], float, float, str, Optional[str]]:
    """Core RANSAC 2-point affine inverse-depth fit (1/Z = s * d + t) with scale-only fallback."""
    n = len(z_obs)
    if n < 5:
        return None, 0.0, 0.0, "rejected", "too_few_sparse_points"
    inv_z_obs = 1.0 / z_obs
    rng = np.random.default_rng(cfg.seed)
    best_inliers = np.zeros(n, dtype=bool)
    best_cnt = 0

    for _ in range(cfg.ransac_iterations):
        idx = rng.choice(n, size=2, replace=False)
        d0, d1 = d_obs[idx[0]], d_obs[idx[1]]
        if abs(d1 - d0) < 1e-6:
            continue
        s_cand = (inv_z_obs[idx[1]] - inv_z_obs[idx[0]]) / (d1 - d0)
        if s_cand <= 1e-6:
            continue
        t_cand = inv_z_obs[idx[0]] - s_cand * d0
        inv_pred = s_cand * d_obs + t_cand
        if np.any(inv_pred <= 1e-5):
            continue
        z_pred = 1.0 / inv_pred
        rel_err = np.abs(z_pred - z_obs) / z_obs
        inl = rel_err <= cfg.ransac_rel_depth_thresh
        cnt = int(np.sum(inl))
        if cnt > best_cnt:
            best_cnt = cnt
            best_inliers = inl

    inlier_ratio = float(best_cnt) / float(n)
    use_scale_only = False
    s_fit, t_fit = 0.0, 0.0
    if best_cnt >= 4 and inlier_ratio >= cfg.min_inlier_ratio:
        A = np.column_stack([d_obs[best_inliers], np.ones(best_cnt, dtype=np.float64)])
        sol, _, _, _ = np.linalg.lstsq(A, inv_z_obs[best_inliers], rcond=None)
        s_fit, t_fit = float(sol[0]), float(sol[1])
        min_disp = (
            float(np.min(s_fit * full_disparity_map.astype(np.float64) + t_fit))
            if full_disparity_map is not None
            else float(np.min(s_fit * d_obs + t_fit))
        )
        if s_fit <= 1e-6 or min_disp <= 1e-4:
            use_scale_only = True
        else:
            inv_pred = s_fit * d_obs + t_fit
            z_pred = 1.0 / np.maximum(inv_pred, 1e-6)
            inlier_ratio = float(np.mean(np.abs(z_pred - z_obs) / z_obs <= cfg.ransac_rel_depth_thresh))
            if inlier_ratio < cfg.min_inlier_ratio:
                use_scale_only = True
    else:
        use_scale_only = True

    fit_mode = "scale_shift_ransac"
    if use_scale_only:
        pos = d_obs > 1e-3
        if int(np.sum(pos)) < 4:
            return None, 0.0, inlier_ratio, "rejected", "non_positive_disparity_samples"
        s_fit = float(np.median(inv_z_obs[pos] / d_obs[pos]))
        t_fit = 0.0
        if s_fit <= 1e-6:
            return None, 0.0, 0.0, "rejected", "negative_scale_only_fit"
        inv_pred = np.maximum(s_fit * d_obs, 1e-6)
        z_pred = 1.0 / inv_pred
        inlier_ratio = float(np.mean(np.abs(z_pred - z_obs) / z_obs <= cfg.ransac_rel_depth_thresh))
        if inlier_ratio < cfg.min_fallback_inlier_ratio:
            return None, 0.0, inlier_ratio, "rejected", "low_inlier_ratio_after_scale_only_fallback"
        fit_mode = "scale_only_fallback"

    return s_fit, t_fit, inlier_ratio, fit_mode, None


def evaluate_heldout_sparse_alignment(
    disparity_map: np.ndarray,
    pose: SfmCameraPose,
    cfg: MultiViewFusionConfig,
    train_frac: float = 0.8,
) -> Tuple[Optional[float], Optional[float]]:
    """Fit (scale, shift) on 80% of sparse points and evaluate Abs Rel and inlier ratio on the held-out 20%."""
    h, w = disparity_map.shape[:2]
    if pose.observed_xyz_w.shape[0] < 15:
        return None, None
    pts_cam = (pose.R_w2c @ pose.observed_xyz_w.T).T + pose.t_w2c[None, :]
    z_sfm = pts_cam[:, 2]
    sx = float(w) / float(pose.intrinsics.width)
    sy = float(h) / float(pose.intrinsics.height)
    u_d = np.round(pose.observed_uv[:, 0] * sx).astype(np.int64)
    v_d = np.round(pose.observed_uv[:, 1] * sy).astype(np.int64)
    valid = (z_sfm > 1e-3) & (u_d >= 0) & (u_d < w) & (v_d >= 0) & (v_d < h)
    if int(np.sum(valid)) < 15:
        return None, None

    z_obs = z_sfm[valid]
    d_obs = disparity_map[v_d[valid], u_d[valid]].astype(np.float64)
    rng = np.random.default_rng(cfg.seed + 991)
    perm = rng.permutation(len(z_obs))
    n_train = max(8, int(round(train_frac * len(z_obs))))
    if n_train >= len(z_obs) - 2:
        return None, None

    tr_idx, te_idx = perm[:n_train], perm[n_train:]
    s_fit, t_fit, _, _, rej = _fit_inverse_depth_ransac(d_obs[tr_idx], z_obs[tr_idx], disparity_map, cfg)
    if s_fit is None or rej is not None:
        return None, None

    inv_pred_te = np.maximum(s_fit * d_obs[te_idx] + t_fit, 1e-4)
    z_pred_te = 1.0 / inv_pred_te
    rel_err_te = np.abs(z_pred_te - z_obs[te_idx]) / z_obs[te_idx]
    return float(np.mean(rel_err_te)), float(np.mean(rel_err_te <= cfg.ransac_rel_depth_thresh))


def align_monocular_disparity_to_sparse_depth(
    disparity_map: np.ndarray,
    pose: SfmCameraPose,
    depth_intrinsics: CameraIntrinsics,
    cfg: MultiViewFusionConfig,
) -> Tuple[Optional[np.ndarray], str, float, float, int, float, float, float, Optional[str]]:
    """Align monocular disparity `d_mono` (larger = closer) to SfM sparse 3D points in inverse-depth space.

    Model: 1 / Z = s * d_mono + t (with s > 0).
    Falls back to scale-only (t = 0) if RANSAC inlier ratio < `cfg.min_inlier_ratio` OR if
    `min(s * d_mono + t) <= 1e-4` (non-positive disparity). Rejects view if scale-only also fails.

    Returns:
        (aligned_depth_map, fit_mode, scale, shift, num_pts, inlier_ratio, in_sample_abs_rel, rmse, rejection_reason)
    """
    h, w = disparity_map.shape[:2]
    if pose.observed_xyz_w.shape[0] < cfg.min_sparse_points:
        return None, "rejected", 0.0, 0.0, int(pose.observed_xyz_w.shape[0]), 0.0, float("inf"), float("inf"), "too_few_sparse_points"

    pts_cam = (pose.R_w2c @ pose.observed_xyz_w.T).T + pose.t_w2c[None, :]
    z_sfm = pts_cam[:, 2]
    sx = float(w) / float(pose.intrinsics.width)
    sy = float(h) / float(pose.intrinsics.height)
    u_d = np.round(pose.observed_uv[:, 0] * sx).astype(np.int64)
    v_d = np.round(pose.observed_uv[:, 1] * sy).astype(np.int64)

    valid = (z_sfm > 1e-3) & (u_d >= 0) & (u_d < w) & (v_d >= 0) & (v_d < h)
    if int(np.sum(valid)) < cfg.min_sparse_points:
        return None, "rejected", 0.0, 0.0, int(np.sum(valid)), 0.0, float("inf"), float("inf"), "too_few_in_frame_sparse_points"

    z_obs = z_sfm[valid]
    d_obs = disparity_map[v_d[valid], u_d[valid]].astype(np.float64)
    n = len(z_obs)

    s_fit, t_fit, inlier_ratio, fit_mode, rej_reason = _fit_inverse_depth_ransac(
        d_obs, z_obs, disparity_map, cfg
    )
    if s_fit is None or rej_reason is not None:
        return None, "rejected", 0.0, 0.0, n, inlier_ratio, float("inf"), float("inf"), rej_reason

    inv_z_map = np.maximum(s_fit * disparity_map.astype(np.float64) + t_fit, 1e-4)
    aligned_depth = (1.0 / inv_z_map).astype(np.float32)

    z_pred_obs = aligned_depth[v_d[valid], u_d[valid]].astype(np.float64)
    abs_rel = float(np.mean(np.abs(z_pred_obs - z_obs) / z_obs))
    rmse = float(np.sqrt(np.mean((z_pred_obs - z_obs) ** 2)))
    return aligned_depth, fit_mode, s_fit, t_fit, n, inlier_ratio, abs_rel, rmse, None


def _select_neighbor_views(
    target_name: str,
    accepted_names: List[str],
    poses: Dict[str, SfmCameraPose],
    max_neighbors: int,
) -> List[str]:
    """Rank neighbor views by shared sparse 3D point IDs (with camera-center distance tie-breaker)."""
    target_ids = set(poses[target_name].observed_point3d_ids.tolist())
    c_target = poses[target_name].camera_center_w
    scored = []
    for other in accepted_names:
        if other == target_name:
            continue
        other_ids = set(poses[other].observed_point3d_ids.tolist())
        shared = len(target_ids & other_ids)
        dist = float(np.linalg.norm(poses[other].camera_center_w - c_target))
        scored.append((shared, -dist, other))
    scored.sort(reverse=True)
    return [x[2] for x in scored[:max_neighbors]]


def filter_depth_by_discontinuity_and_consistency(
    view_name: str,
    rgb_u8: np.ndarray,
    aligned_depth: np.ndarray,
    intr_depth: CameraIntrinsics,
    pose: SfmCameraPose,
    neighbor_data: List[Tuple[SfmCameraPose, CameraIntrinsics, np.ndarray, np.ndarray]],
    cfg: MultiViewFusionConfig,
) -> Tuple[np.ndarray, int, int, int, int]:
    """Mask out high depth-discontinuity pixels and multi-view inconsistent pixels.

    Returns:
        (filtered_depth_f32, total_pixels, rejected_discontinuity_px, rejected_consistency_px, valid_px)
    """
    h, w = aligned_depth.shape[:2]
    total_px = h * w

    # 1. Mask high depth-discontinuity pixels using physical log-depth gradient ||grad(log Z)||_2
    log_z = np.log(np.maximum(aligned_depth, 1e-4)).astype(np.float32)
    gx = cv2.Sobel(log_z, cv2.CV_32F, 1, 0, ksize=3) * 0.25
    gy = cv2.Sobel(log_z, cv2.CV_32F, 0, 1, ksize=3) * 0.25
    grad_log_z = np.sqrt(gx * gx + gy * gy)
    valid_mask = (grad_log_z <= cfg.discontinuity_threshold) & np.isfinite(aligned_depth) & (aligned_depth > 1e-3)
    # Clip extreme far tail to avoid TSDF stray background rays
    z_max = float(np.percentile(aligned_depth[valid_mask], cfg.max_depth_percentile_clip)) if np.any(valid_mask) else 10.0
    valid_mask &= aligned_depth <= max(z_max, 0.1)
    rejected_disc_px = int(total_px - int(np.sum(valid_mask)))

    # 2. Multi-view geometric + photometric consistency check across neighbor views
    rejected_cons_px = 0
    if neighbor_data and int(np.sum(valid_mask)) > 0 and cfg.min_consistent_views > 0:
        vs, us = np.where(valid_mask)
        zs = aligned_depth[vs, us].astype(np.float64)
        # Back-project valid pixels to world coordinates
        x_cam = (us.astype(np.float64) - intr_depth.cx) * zs / intr_depth.fx
        y_cam = (vs.astype(np.float64) - intr_depth.cy) * zs / intr_depth.fy
        pts_cam = np.column_stack([x_cam, y_cam, zs])
        pts_world = (pose.R_w2c.T @ (pts_cam - pose.t_w2c[None, :]).T).T

        src_rgb = rgb_u8[vs, us].astype(np.float32) / 255.0
        consistent_votes = np.zeros(len(zs), dtype=np.int32)

        for nbr_pose, nbr_intr, nbr_depth, nbr_rgb_u8 in neighbor_data:
            nh, nw = nbr_depth.shape[:2]
            pts_nbr_cam = (nbr_pose.R_w2c @ pts_world.T).T + nbr_pose.t_w2c[None, :]
            z_proj = pts_nbr_cam[:, 2]
            in_front = z_proj > 1e-3
            u_nbr = np.round(nbr_intr.fx * (pts_nbr_cam[:, 0] / np.maximum(z_proj, 1e-8)) + nbr_intr.cx).astype(np.int64)
            v_nbr = np.round(nbr_intr.fy * (pts_nbr_cam[:, 1] / np.maximum(z_proj, 1e-8)) + nbr_intr.cy).astype(np.int64)
            in_bounds = in_front & (u_nbr >= 0) & (u_nbr < nw) & (v_nbr >= 0) & (v_nbr < nh)
            if not np.any(in_bounds):
                continue

            idx_b = np.where(in_bounds)[0]
            z_n = nbr_depth[v_nbr[idx_b], u_nbr[idx_b]].astype(np.float64)
            z_p = z_proj[idx_b]
            rel_diff = np.abs(z_p - z_n) / np.maximum(np.maximum(z_p, z_n), 1e-6)
            geo_ok = (z_n > 1e-3) & (rel_diff <= cfg.consistency_rel_depth_thresh)

            nbr_rgb = nbr_rgb_u8[v_nbr[idx_b], u_nbr[idx_b]].astype(np.float32) / 255.0
            rgb_diff = np.mean(np.abs(src_rgb[idx_b] - nbr_rgb), axis=1)
            photo_ok = rgb_diff <= cfg.consistency_rgb_thresh

            consistent_votes[idx_b[geo_ok & photo_ok]] += 1

        req_votes = min(cfg.min_consistent_views, len(neighbor_data))
        keep_idx = consistent_votes >= req_votes
        rejected_cons_px = int(len(zs) - int(np.sum(keep_idx)))
        new_mask = np.zeros_like(valid_mask)
        new_mask[vs[keep_idx], us[keep_idx]] = True
        valid_mask = new_mask

    filtered_depth = np.where(valid_mask, aligned_depth, 0.0).astype(np.float32)
    valid_px = int(np.sum(valid_mask))
    return filtered_depth, total_px, rejected_disc_px, rejected_cons_px, valid_px


def _blend_multiview_vertex_colors(
    vertices: np.ndarray,
    normals: np.ndarray,
    accepted_names: List[str],
    poses: Dict[str, SfmCameraPose],
    depth_intrinsics_map: Dict[str, CameraIntrinsics],
    cache_dir: Path,
    power: float = 2.0,
) -> np.ndarray:
    """Apply view-angle-weighted multi-image color blending to fused mesh/cloud vertices."""
    n_verts = len(vertices)
    if n_verts == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    color_accum = np.zeros((n_verts, 3), dtype=np.float64)
    weight_accum = np.zeros(n_verts, dtype=np.float64)

    for name in accepted_names:
        pose = poses[name]
        intr = depth_intrinsics_map[name]
        rgb = np.load(cache_dir / f"{name}_rgb.npy")
        depth = np.load(cache_dir / f"{name}_depth.npy")
        h, w = depth.shape[:2]

        pts_c = (pose.R_w2c @ vertices.T).T + pose.t_w2c[None, :]
        z = pts_c[:, 2]
        u = np.round(intr.fx * (pts_c[:, 0] / np.maximum(z, 1e-8)) + intr.cx).astype(np.int64)
        v = np.round(intr.fy * (pts_c[:, 1] / np.maximum(z, 1e-8)) + intr.cy).astype(np.int64)
        in_view = (z > 1e-3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(in_view):
            del rgb, depth
            continue

        idx = np.where(in_view)[0]
        z_map = depth[v[idx], u[idx]].astype(np.float64)
        # Check visibility against aligned depth map (within 8% tolerance)
        vis_ok = (z_map > 1e-3) & (np.abs(z[idx] - z_map) / np.maximum(z_map, 1e-6) <= 0.12)
        idx = idx[vis_ok]
        if len(idx) == 0:
            del rgb, depth
            continue

        # View-angle cosine weight: max(0, dot(normal, dir_to_camera))^power
        cam_c = pose.camera_center_w
        view_dirs = cam_c[None, :] - vertices[idx]
        view_dirs /= np.maximum(np.linalg.norm(view_dirs, axis=1, keepdims=True), 1e-8)
        cos_ang = np.sum(normals[idx] * view_dirs, axis=1)
        weights = np.power(np.maximum(cos_ang, 0.05), power)

        sampled_rgb = rgb[v[idx], u[idx]].astype(np.float64) / 255.0
        color_accum[idx] += sampled_rgb * weights[:, None]
        weight_accum[idx] += weights
        del rgb, depth

    out_rgb = np.full((n_verts, 3), 0.7, dtype=np.float64)
    has_w = weight_accum > 1e-8
    out_rgb[has_w] = color_accum[has_w] / weight_accum[has_w, None]
    return np.clip(np.round(out_rgb * 255.0), 0, 255).astype(np.uint8)


def fuse_multiview_tsdf_and_mesh(
    sfm_result: SfmReconstructionResult,
    depth_estimator: DepthModel,
    config: Optional[MultiViewFusionConfig] = None,
) -> MultiViewFusionResult:
    """Run sequential sparse-aligned monocular depth + consistency filtering + Open3D TSDF fusion."""
    cfg = config or MultiViewFusionConfig()
    if sfm_result.num_registered_images < 2:
        raise GeometryProcessingError(
            f"Multi-view fusion requires at least 2 registered views, got {sfm_result.num_registered_images}"
        )

    proc = psutil.Process(os.getpid())
    baseline_ram = proc.memory_info().rss
    peak_ram = baseline_ram
    stop_evt = threading.Event()

    def _sampler() -> None:
        nonlocal peak_ram
        while not stop_evt.is_set():
            curr = proc.memory_info().rss
            if curr > peak_ram:
                peak_ram = curr
            stop_evt.wait(0.02)

    sampler = threading.Thread(target=_sampler, daemon=True)
    sampler.start()

    t_start = time.perf_counter()
    cache_dir = Path(tempfile.mkdtemp(prefix="hologen_mv_cache_"))

    try:
        poses = sfm_result.registered_poses
        view_names = sorted(poses.keys())

        # Stage 1: Sequential monocular depth inference & RANSAC sparse alignment (one view at a time)
        t_s1 = time.perf_counter()
        depth_intrinsics_map: Dict[str, CameraIntrinsics] = {}
        accepted_names: List[str] = []
        partial_stats: Dict[str, Dict[str, Any]] = {}
        num_fallbacks = 0
        num_rejected = 0

        for name in view_names:
            pose = poses[name]
            bgr = cv2.imread(str(pose.image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                num_rejected += 1
                partial_stats[name] = {
                    "fit_mode": "rejected", "rejected": True, "rejection_reason": "image_read_failed",
                    "scale": 0.0, "shift": 0.0, "num_sparse_points": 0, "inlier_ratio": 0.0,
                    "abs_rel": float("inf"), "rmse": float("inf"),
                }
                continue

            rgb_full = bgr[:, :, ::-1]
            h0, w0 = rgb_full.shape[:2]
            max_dim = max(h0, w0)
            if cfg.depth_max_dim > 0 and max_dim > cfg.depth_max_dim:
                sc = float(cfg.depth_max_dim) / float(max_dim)
                rgb_resized = cv2.resize(rgb_full, (max(1, int(round(w0 * sc))), max(1, int(round(h0 * sc)))), interpolation=cv2.INTER_AREA)
            else:
                rgb_resized = rgb_full

            rh, rw = rgb_resized.shape[:2]
            intr_d = _scale_intrinsics_to_size(pose.intrinsics, rw, rh)
            depth_pred = depth_estimator.predict(rgb_resized)

            aligned_d, fit_mode, s_fit, t_fit, n_pts, inl_ratio, abs_rel, rmse, rej_reason = (
                align_monocular_disparity_to_sparse_depth(depth_pred.disparity_map, pose, intr_d, cfg)
            )
            ho_abs_rel, ho_inl_ratio = evaluate_heldout_sparse_alignment(depth_pred.disparity_map, pose, cfg)
            if aligned_d is None:
                num_rejected += 1
                partial_stats[name] = {
                    "fit_mode": "rejected", "rejected": True, "rejection_reason": rej_reason,
                    "scale": s_fit, "shift": t_fit, "num_sparse_points": n_pts, "inlier_ratio": inl_ratio,
                    "abs_rel": abs_rel, "rmse": rmse,
                    "heldout_abs_rel": ho_abs_rel, "heldout_inlier_ratio": ho_inl_ratio,
                }
            else:
                if fit_mode == "scale_only_fallback":
                    num_fallbacks += 1
                accepted_names.append(name)
                depth_intrinsics_map[name] = intr_d
                np.save(cache_dir / f"{name}_depth.npy", aligned_d)
                np.save(cache_dir / f"{name}_rgb.npy", rgb_resized)
                partial_stats[name] = {
                    "fit_mode": fit_mode, "rejected": False, "rejection_reason": None,
                    "scale": s_fit, "shift": t_fit, "num_sparse_points": n_pts, "inlier_ratio": inl_ratio,
                    "abs_rel": abs_rel, "rmse": rmse,
                    "heldout_abs_rel": ho_abs_rel, "heldout_inlier_ratio": ho_inl_ratio,
                }

            del bgr, rgb_full, rgb_resized, depth_pred, aligned_d
            gc.collect()

        depth_and_align_ms = (time.perf_counter() - t_s1) * 1000.0
        if not accepted_names:
            raise GeometryProcessingError("All views were rejected during sparse-to-monocular depth alignment.")

        # Auto-determine TSDF voxel length from sparse scene bounding box
        if cfg.tsdf_voxel_length is not None:
            voxel_len = float(cfg.tsdf_voxel_length)
        else:
            pts_ref = sfm_result.sparse_points_xyz
            if len(pts_ref) >= 10:
                q_low, q_high = np.percentile(pts_ref, [5.0, 95.0], axis=0)
                diag = float(np.linalg.norm(q_high - q_low))
                voxel_len = max(diag / 220.0, 0.005)
            else:
                voxel_len = 0.02

        sdf_trunc = voxel_len * cfg.tsdf_sdf_trunc_mult
        tsdf_vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_len,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        # Stage 2: Sequential consistency filtering + TSDF integration
        t_s2 = time.perf_counter()
        view_stats: List[ViewFusionStats] = []
        tot_eval = tot_disc_rej = tot_cons_rej = tot_fused = 0

        for name in view_names:
            ps = partial_stats[name]
            if ps["rejected"]:
                view_stats.append(
                    ViewFusionStats(
                        image_name=name, fit_mode="rejected", rejected=True,
                        rejection_reason=ps["rejection_reason"], scale=ps["scale"], shift=ps["shift"],
                        num_sparse_points=ps["num_sparse_points"], inlier_ratio=ps["inlier_ratio"],
                        sparse_abs_rel=ps["abs_rel"], sparse_rmse=ps["rmse"],
                        total_pixels=0, rejected_discontinuity_pixels=0,
                        rejected_consistency_pixels=0, valid_fused_pixels=0,
                        heldout_abs_rel=ps.get("heldout_abs_rel"),
                        heldout_inlier_ratio=ps.get("heldout_inlier_ratio"),
                    )
                )
                continue

            pose = poses[name]
            intr_d = depth_intrinsics_map[name]
            aligned_d = np.load(cache_dir / f"{name}_depth.npy")
            rgb_u8 = np.load(cache_dir / f"{name}_rgb.npy")

            nbr_names = _select_neighbor_views(name, accepted_names, poses, cfg.num_neighbor_views)
            nbr_data = []
            for nn in nbr_names:
                nbr_data.append((
                    poses[nn],
                    depth_intrinsics_map[nn],
                    np.load(cache_dir / f"{nn}_depth.npy"),
                    np.load(cache_dir / f"{nn}_rgb.npy"),
                ))

            filt_d, t_px, r_disc, r_cons, v_px = filter_depth_by_discontinuity_and_consistency(
                name, rgb_u8, aligned_d, intr_d, pose, nbr_data, cfg
            )
            tot_eval += t_px
            tot_disc_rej += r_disc
            tot_cons_rej += r_cons
            tot_fused += v_px

            view_stats.append(
                ViewFusionStats(
                    image_name=name, fit_mode=ps["fit_mode"], rejected=False, rejection_reason=None,
                    scale=ps["scale"], shift=ps["shift"], num_sparse_points=ps["num_sparse_points"],
                    inlier_ratio=ps["inlier_ratio"], sparse_abs_rel=ps["abs_rel"], sparse_rmse=ps["rmse"],
                    total_pixels=t_px, rejected_discontinuity_pixels=r_disc,
                    rejected_consistency_pixels=r_cons, valid_fused_pixels=v_px,
                    heldout_abs_rel=ps.get("heldout_abs_rel"),
                    heldout_inlier_ratio=ps.get("heldout_inlier_ratio"),
                )
            )

            if v_px > 0:
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(rgb_u8)),
                    o3d.geometry.Image(np.ascontiguousarray(filt_d)),
                    depth_scale=1.0,
                    depth_trunc=float(np.max(filt_d)) + 0.5,
                    convert_rgb_to_intensity=False,
                )
                o3d_intr = o3d.camera.PinholeCameraIntrinsic(
                    intr_d.width, intr_d.height, intr_d.fx, intr_d.fy, intr_d.cx, intr_d.cy
                )
                extr_w2c = np.eye(4, dtype=np.float64)
                extr_w2c[:3, :3] = pose.R_w2c
                extr_w2c[:3, 3] = pose.t_w2c
                tsdf_vol.integrate(rgbd, o3d_intr, extr_w2c)

            del aligned_d, rgb_u8, nbr_data, filt_d
            gc.collect()

        consistency_and_tsdf_ms = (time.perf_counter() - t_s2) * 1000.0

        # Stage 3: Mesh & PointCloud extraction + View-angle-weighted multi-image color blending
        t_s3 = time.perf_counter()
        o3d_mesh = tsdf_vol.extract_triangle_mesh()
        o3d_mesh.remove_duplicated_vertices()
        o3d_mesh.remove_duplicated_triangles()
        o3d_mesh.remove_degenerate_triangles()
        o3d_mesh.compute_vertex_normals()

        verts = np.asarray(o3d_mesh.vertices, dtype=np.float64)
        norms = np.asarray(o3d_mesh.vertex_normals, dtype=np.float64)
        faces = np.asarray(o3d_mesh.triangles, dtype=np.int32)

        blended_rgb = _blend_multiview_vertex_colors(
            verts, norms, accepted_names, poses, depth_intrinsics_map, cache_dir
        )
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(blended_rgb.astype(np.float64) / 255.0)

        o3d_pcd = tsdf_vol.extract_point_cloud()
        pcd_pts = np.asarray(o3d_pcd.points, dtype=np.float64)
        pcd_rgb_01 = np.clip(np.asarray(o3d_pcd.colors, dtype=np.float64), 0.0, 1.0).astype(np.float32)
        pcd_norms = np.asarray(o3d_pcd.normals, dtype=np.float64) if o3d_pcd.has_normals() else None

        mesh_and_blend_ms = (time.perf_counter() - t_s3) * 1000.0
        total_ms = (time.perf_counter() - t_start) * 1000.0

        stop_evt.set()
        sampler.join(timeout=1.0)
        peak_ram_mb = float(peak_ram) / (1024.0 * 1024.0)
        peak_ram_delta_mb = float(peak_ram - baseline_ram) / (1024.0 * 1024.0)

        pc_res = PointCloudData(
            points=pcd_pts.astype(np.float32),
            colors=pcd_rgb_01,
            pixel_coords=np.zeros((len(pcd_pts), 2), dtype=np.int32),
            valid_mask=np.ones((1, 1), dtype=bool),
            generation_time_ms=consistency_and_tsdf_ms,
            normals=pcd_norms.astype(np.float32) if pcd_norms is not None else None,
        )
        mesh_res = MeshData(
            vertices=verts.astype(np.float32),
            faces=faces,
            vertex_colors=(blended_rgb.astype(np.float32) / 255.0),
            vertex_normals=norms.astype(np.float32),
            method="multiview_tsdf_view_weighted",
            generation_time_ms=mesh_and_blend_ms,
            parameters={"tsdf_voxel_length": voxel_len},
        )

        logger.info(
            "7c Fusion complete: %d/%d views accepted (%d fallbacks, %d rejected), "
            "%d faces, %.1f%% pixels fused, runtime %.2f s, peak RAM %.1f MB",
            len(accepted_names),
            len(view_names),
            num_fallbacks,
            num_rejected,
            mesh_res.num_faces,
            100.0 * tot_fused / max(tot_eval, 1),
            total_ms / 1000.0,
            peak_ram_mb,
        )

        return MultiViewFusionResult(
            point_cloud=pc_res,
            mesh=mesh_res,
            view_stats=view_stats,
            num_registered_views=len(view_names),
            num_accepted_views=len(accepted_names),
            num_rejected_views=num_rejected,
            num_scale_only_fallbacks=num_fallbacks,
            total_pixels_evaluated=tot_eval,
            total_rejected_discontinuity_pixels=tot_disc_rej,
            total_rejected_consistency_pixels=tot_cons_rej,
            total_valid_fused_pixels=tot_fused,
            tsdf_voxel_length=voxel_len,
            depth_and_align_ms=depth_and_align_ms,
            consistency_and_tsdf_ms=consistency_and_tsdf_ms,
            mesh_and_blend_ms=mesh_and_blend_ms,
            total_runtime_ms=total_ms,
            peak_ram_mb=peak_ram_mb,
            peak_ram_above_baseline_mb=peak_ram_delta_mb,
            config={
                "depth_max_dim": cfg.depth_max_dim,
                "min_inlier_ratio": cfg.min_inlier_ratio,
                "discontinuity_threshold": cfg.discontinuity_threshold,
                "consistency_rel_depth_thresh": cfg.consistency_rel_depth_thresh,
                "consistency_rgb_thresh": cfg.consistency_rgb_thresh,
                "min_consistent_views": cfg.min_consistent_views,
                "tsdf_voxel_length": voxel_len,
            },
        )
    finally:
        stop_evt.set()
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
