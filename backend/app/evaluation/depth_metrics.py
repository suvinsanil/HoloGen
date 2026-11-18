"""Depth estimation metrics, scale-shift alignment, edge-preservation metrics, and RGB-D loader."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
from PIL import Image


@dataclass
class DepthEvalMetrics:
    """Standard monocular depth benchmark metrics against metric ground truth."""

    abs_rel: float
    sq_rel: float
    rmse: float
    rmse_log: float
    delta_1: float  # % of pixels with max(z/z*, z*/z) < 1.25
    delta_2: float  # % of pixels with max(z/z*, z*/z) < 1.25^2
    delta_3: float  # % of pixels with max(z/z*, z*/z) < 1.25^3
    edge_f1: float  # F1 score of depth discontinuity edges vs GT depth edges
    gradient_correlation: float  # Pearson correlation of log-depth gradient magnitude vs GT
    num_valid_pixels: int
    alignment_mode: str
    scale_factor: float
    shift_factor: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "abs_rel": round(self.abs_rel, 4),
            "sq_rel": round(self.sq_rel, 4),
            "rmse": round(self.rmse, 4),
            "rmse_log": round(self.rmse_log, 4),
            "delta_1": round(self.delta_1, 4),
            "delta_2": round(self.delta_2, 4),
            "delta_3": round(self.delta_3, 4),
            "edge_f1": round(self.edge_f1, 4),
            "gradient_correlation": round(self.gradient_correlation, 4),
            "num_valid_pixels": self.num_valid_pixels,
            "alignment_mode": self.alignment_mode,
            "scale_factor": round(self.scale_factor, 6),
            "shift_factor": round(self.shift_factor, 6),
        }


def align_depth_least_squares(
    pred: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
) -> Tuple[np.ndarray, float, float]:
    """Align predicted depth to ground-truth depth via closed-form linear least squares:
    min_{s, t} sum_{i in valid} (s * pred_i + t - gt_i)^2.
    """
    mask = valid_mask.astype(bool) & np.isfinite(pred) & np.isfinite(gt_depth)
    p_valid = pred[mask].astype(np.float64)
    g_valid = gt_depth[mask].astype(np.float64)
    if p_valid.size < 4:
        return np.clip(pred, min_depth, max_depth).astype(np.float32), 1.0, 0.0

    A = np.vstack([p_valid, np.ones_like(p_valid)]).T
    sol, _, _, _ = np.linalg.lstsq(A, g_valid, rcond=None)
    s, t = float(sol[0]), float(sol[1])
    aligned = np.clip(s * pred.astype(np.float64) + t, min_depth, max_depth).astype(np.float32)
    return aligned, s, t


def align_disparity_least_squares(
    pred_disparity: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
) -> Tuple[np.ndarray, float, float]:
    """Align predicted affine-invariant disparity d to ground-truth inverse depth (1 / Z_gt)
    via closed-form linear least squares (Ranftl et al., MiDaS TPAMI 2022):
    min_{s, t} sum_{i in valid} (s * d_i + t - (1 / gt_i))^2, then Z_aligned = 1 / (s * d + t).
    """
    mask = valid_mask.astype(bool) & np.isfinite(pred_disparity) & (gt_depth > min_depth)
    d_valid = pred_disparity[mask].astype(np.float64)
    inv_gt_valid = 1.0 / gt_depth[mask].astype(np.float64)
    if d_valid.size < 4:
        return np.clip(pred_disparity, min_depth, max_depth).astype(np.float32), 1.0, 0.0

    A = np.vstack([d_valid, np.ones_like(d_valid)]).T
    sol, _, _, _ = np.linalg.lstsq(A, inv_gt_valid, rcond=None)
    s, t = float(sol[0]), float(sol[1])
    inv_aligned = s * pred_disparity.astype(np.float64) + t
    inv_aligned = np.clip(inv_aligned, 1.0 / max_depth, 1.0 / min_depth)
    depth_aligned = (1.0 / inv_aligned).astype(np.float32)
    return depth_aligned, s, t


def compute_edge_preservation_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    edge_percentile: float = 85.0,
) -> Tuple[float, float]:
    """Compute edge F1 score and gradient magnitude Pearson correlation between predicted and GT depth."""
    p = np.maximum(np.asarray(pred_depth, dtype=np.float32), 1e-4)
    g = np.maximum(np.asarray(gt_depth, dtype=np.float32), 1e-4)

    log_p = np.log(p)
    log_g = np.log(g)

    px = cv2.Sobel(log_p, cv2.CV_32F, 1, 0, ksize=3)
    py = cv2.Sobel(log_p, cv2.CV_32F, 0, 1, ksize=3)
    grad_p = np.sqrt(px * px + py * py)

    gx = cv2.Sobel(log_g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(log_g, cv2.CV_32F, 0, 1, ksize=3)
    grad_g = np.sqrt(gx * gx + gy * gy)

    if valid_mask is not None:
        # Erode valid_mask by 2 pixels so missing-sensor-depth borders don't create artificial step edges
        kernel = np.ones((5, 5), dtype=np.uint8)
        interior_mask = cv2.erode(valid_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    else:
        interior_mask = np.ones_like(p, dtype=bool)

    gp_v = grad_p[interior_mask]
    gg_v = grad_g[interior_mask]
    if gp_v.size < 16 or np.std(gp_v) < 1e-8 or np.std(gg_v) < 1e-8:
        return 0.0, 0.0

    corr_matrix = np.corrcoef(gp_v, gg_v)
    grad_corr = float(corr_matrix[0, 1]) if np.isfinite(corr_matrix[0, 1]) else 0.0

    thresh_p = float(np.percentile(gp_v, edge_percentile))
    thresh_g = float(np.percentile(gg_v, edge_percentile))

    edges_p = (grad_p > max(thresh_p, 1e-5)) & interior_mask
    edges_g = (grad_g > max(thresh_g, 1e-5)) & interior_mask

    # Allow 1-pixel morphological dilation tolerance for boundary localization
    dilate_k = np.ones((3, 3), dtype=np.uint8)
    edges_p_dil = cv2.dilate(edges_p.astype(np.uint8), dilate_k, iterations=1).astype(bool)
    edges_g_dil = cv2.dilate(edges_g.astype(np.uint8), dilate_k, iterations=1).astype(bool)

    tp_prec = int(np.sum(edges_p & edges_g_dil))
    tp_rec = int(np.sum(edges_g & edges_p_dil))
    num_p = int(np.sum(edges_p))
    num_g = int(np.sum(edges_g))

    if num_p == 0 or num_g == 0:
        return 0.0, grad_corr

    precision = float(tp_prec) / float(num_p)
    recall = float(tp_rec) / float(num_g)
    if precision + recall < 1e-8:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return float(f1), float(grad_corr)


def compute_depth_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    alignment_mode: str = "depth_scale_shift",
    raw_disparity: Optional[np.ndarray] = None,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
) -> DepthEvalMetrics:
    """Compute Abs Rel, Sq Rel, RMSE, RMSE log, delta thresholds, and edge metrics.

    Args:
        pred_depth: Predicted depth map (H, W) in positive depth units.
        gt_depth: Ground-truth metric depth map (H, W) in meters.
        valid_mask: Optional boolean mask of valid ground-truth pixels.
        alignment_mode: One of:
            - "none": Direct metric evaluation without scale-shift alignment.
            - "depth_scale_shift": Least-squares scale-shift alignment in depth space.
            - "disparity_scale_shift": Least-squares scale-shift alignment in inverse-depth space.
        raw_disparity: Optional raw disparity map (used when alignment_mode == "disparity_scale_shift").
    """
    pred = np.asarray(pred_depth, dtype=np.float32)
    gt = np.asarray(gt_depth, dtype=np.float32)
    if pred.shape != gt.shape:
        raise ValueError(f"pred_depth shape {pred.shape} != gt_depth shape {gt.shape}")

    mask = np.isfinite(gt) & (gt >= float(min_depth)) & (gt <= float(max_depth))
    if valid_mask is not None:
        mask = mask & valid_mask.astype(bool)

    if alignment_mode == "none":
        aligned = np.clip(pred, min_depth, max_depth)
        scale_f, shift_f = 1.0, 0.0
    elif alignment_mode == "disparity_scale_shift" and raw_disparity is not None:
        aligned, scale_f, shift_f = align_disparity_least_squares(
            raw_disparity, gt, mask, min_depth=min_depth, max_depth=max_depth
        )
    elif alignment_mode in ("depth_scale_shift", "disparity_scale_shift"):
        aligned, scale_f, shift_f = align_depth_least_squares(
            pred, gt, mask, min_depth=min_depth, max_depth=max_depth
        )
    else:
        raise ValueError(f"Unsupported alignment_mode: {alignment_mode}")

    p = aligned[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    p = np.maximum(p, 1e-4)
    n_valid = int(p.size)
    if n_valid == 0:
        raise ValueError("No valid ground-truth pixels available for depth metric computation.")

    diff = p - g
    abs_rel = float(np.mean(np.abs(diff) / g))
    sq_rel = float(np.mean((diff * diff) / g))
    rmse = float(np.sqrt(np.mean(diff * diff)))

    log_diff = np.log(p) - np.log(g)
    rmse_log = float(np.sqrt(np.mean(log_diff * log_diff)))

    ratio = np.maximum(p / g, g / p)
    delta_1 = float(np.mean(ratio < 1.25))
    delta_2 = float(np.mean(ratio < (1.25 ** 2)))
    delta_3 = float(np.mean(ratio < (1.25 ** 3)))

    edge_f1, grad_corr = compute_edge_preservation_metrics(aligned, gt, valid_mask=mask)

    return DepthEvalMetrics(
        abs_rel=abs_rel,
        sq_rel=sq_rel,
        rmse=rmse,
        rmse_log=rmse_log,
        delta_1=delta_1,
        delta_2=delta_2,
        delta_3=delta_3,
        edge_f1=edge_f1,
        gradient_correlation=grad_corr,
        num_valid_pixels=n_valid,
        alignment_mode=alignment_mode,
        scale_factor=scale_f,
        shift_factor=shift_f,
    )


@dataclass
class RGBDBenchmarkFrame:
    """Single calibrated RGB-D benchmark scene with ground-truth depth in meters."""

    scene_id: str
    dataset_name: str
    rgb_uint8: np.ndarray  # (H, W, 3) uint8
    gt_depth_meters: np.ndarray  # (H, W) float32 in meters
    valid_mask: np.ndarray  # (H, W) bool


def load_rgbd_benchmark_samples(data_root: Path) -> List[RGBDBenchmarkFrame]:
    """Load the 8 calibrated indoor RGB-D benchmark frames (NYU Depth V2, SUN, TUM, Redwood 0..4)."""
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    nyu = o3d.data.SampleNYURGBDImage(data_root=str(data_root))
    sun = o3d.data.SampleSUNRGBDImage(data_root=str(data_root))
    tum = o3d.data.SampleTUMRGBDImage(data_root=str(data_root))
    red = o3d.data.SampleRedwoodRGBDImages(data_root=str(data_root))

    frames: List[RGBDBenchmarkFrame] = []

    # 1. NYU Depth V2 frame
    nyu_rgb = np.asarray(Image.open(nyu.color_path).convert("RGB"), dtype=np.uint8)
    nyu_d_raw = cv2.imread(nyu.depth_path, cv2.IMREAD_UNCHANGED).byteswap().astype(np.float32)
    nyu_valid = (nyu_d_raw > 0) & (nyu_d_raw < 1090)
    nyu_meters = np.where(nyu_valid, 351.3 / np.maximum(1092.5 - nyu_d_raw, 1.0), 0.0).astype(np.float32)
    nyu_valid = nyu_valid & (nyu_meters >= 0.4) & (nyu_meters <= 10.0)
    frames.append(
        RGBDBenchmarkFrame(
            scene_id="nyu_depth_v2_office",
            dataset_name="NYU_Depth_V2",
            rgb_uint8=nyu_rgb,
            gt_depth_meters=nyu_meters,
            valid_mask=nyu_valid,
        )
    )

    # 2. SUN RGB-D frame
    sun_rgb = np.asarray(Image.open(sun.color_path).convert("RGB"), dtype=np.uint8)
    sun_d_raw = cv2.imread(sun.depth_path, cv2.IMREAD_UNCHANGED)
    sun_shifted = (((sun_d_raw >> 3) | (sun_d_raw << 13)) & 0xFFFF).astype(np.float32)
    sun_meters = sun_shifted / 1000.0
    sun_valid = (sun_meters >= 0.4) & (sun_meters <= 10.0)
    frames.append(
        RGBDBenchmarkFrame(
            scene_id="sun_rgbd_bedroom",
            dataset_name="SUN_RGBD",
            rgb_uint8=sun_rgb,
            gt_depth_meters=sun_meters,
            valid_mask=sun_valid,
        )
    )

    # 3. TUM RGB-D frame
    tum_rgb = np.asarray(Image.open(tum.color_path).convert("RGB"), dtype=np.uint8)
    tum_d_raw = cv2.imread(tum.depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    tum_meters = tum_d_raw / 5000.0
    tum_valid = (tum_meters >= 0.4) & (tum_meters <= 10.0)
    frames.append(
        RGBDBenchmarkFrame(
            scene_id="tum_rgbd_desk",
            dataset_name="TUM_RGBD",
            rgb_uint8=tum_rgb,
            gt_depth_meters=tum_meters,
            valid_mask=tum_valid,
        )
    )

    # 4. Redwood Indoor RGB-D (5 frames)
    for idx, (c_path, d_path) in enumerate(zip(red.color_paths, red.depth_paths)):
        red_rgb = np.asarray(Image.open(c_path).convert("RGB"), dtype=np.uint8)
        red_d_raw = cv2.imread(d_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        red_meters = red_d_raw / 1000.0
        red_valid = (red_meters >= 0.4) & (red_meters <= 10.0)
        frames.append(
            RGBDBenchmarkFrame(
                scene_id=f"redwood_livingroom_{idx:02d}",
                dataset_name="Redwood_Indoor_RGBD",
                rgb_uint8=red_rgb,
                gt_depth_meters=red_meters,
                valid_mask=red_valid,
            )
        )

    return frames
