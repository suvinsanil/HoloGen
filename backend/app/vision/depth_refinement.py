"""Edge-preserving depth map refinement with discontinuity protection."""

from dataclasses import dataclass
import time
from typing import Any, Dict, Literal

import cv2
import numpy as np

from app.core.exceptions import DepthEstimationError


@dataclass
class DepthRefinementResult:
    """Structured output from edge-preserving depth refinement."""

    refined_depth: np.ndarray  # float32 (H, W)
    discontinuity_mask: np.ndarray  # bool (H, W) pixels located on sharp depth jumps
    method: str
    parameters: Dict[str, Any]
    processing_time_ms: float
    discontinuity_fraction: float

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "parameters": self.parameters,
            "processing_time_ms": round(self.processing_time_ms, 3),
            "discontinuity_fraction": round(self.discontinuity_fraction, 4),
        }


def detect_depth_discontinuities(
    depth_map: np.ndarray,
    relative_jump_threshold: float = 0.08,
) -> np.ndarray:
    """Detect pixels lying on sharp relative depth discontinuities (occlusion boundaries).

    A pixel is flagged as a discontinuity if its maximum 4-neighbor depth difference
    exceeds `relative_jump_threshold * depth(u, v)`.
    """
    d = np.asarray(depth_map, dtype=np.float32)
    if d.ndim != 2:
        raise DepthEstimationError(f"Expected 2D depth map, got shape {d.shape}")

    pad = np.pad(d, 1, mode="edge")
    diff_up = np.abs(d - pad[:-2, 1:-1])
    diff_down = np.abs(d - pad[2:, 1:-1])
    diff_left = np.abs(d - pad[1:-1, :-2])
    diff_right = np.abs(d - pad[1:-1, 2:])
    max_diff = np.maximum(np.maximum(diff_up, diff_down), np.maximum(diff_left, diff_right))

    rel_diff = max_diff / np.maximum(d, 1e-4)
    return rel_diff > float(relative_jump_threshold)


def guided_filter_gray(
    guidance: np.ndarray,
    target: np.ndarray,
    radius: int = 4,
    eps: float = 1e-3,
) -> np.ndarray:
    """Fast O(N) edge-preserving guided filter (He et al., TPAMI 2013) using box filters.

    Locally models output q_i = a_k * I_i + b_k in window w_k of radius r around pixel k,
    ensuring output edges only appear where guidance image I has edges.
    """
    I = np.asarray(guidance, dtype=np.float32)
    p = np.asarray(target, dtype=np.float32)
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, ksize)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + float(eps))
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    q = mean_a * I + mean_b
    return q.astype(np.float32)


def refine_depth_map(
    depth_map: np.ndarray,
    rgb_uint8: np.ndarray,
    method: Literal["none", "bilateral", "guided", "guided_bilateral"] = "guided",
    bilateral_d: int = 7,
    bilateral_sigma_color: float = 0.05,
    bilateral_sigma_space: float = 7.0,
    guided_radius: int = 4,
    guided_eps: float = 1e-3,
    discontinuity_threshold: float = 0.08,
) -> DepthRefinementResult:
    """Refine a monocular depth map while preserving sharp occlusion boundaries.

     Smoothing is performed in normalized inverse-depth space (1/Z) so relative precision is
    preserved in both near and far regions, and pixels flagged on sharp depth discontinuities
    retain their unblurred boundary values to prevent 'flying pixel' ramps between foreground
    and background surfaces.
    """
    t0 = time.perf_counter()
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim != 2:
        raise DepthEstimationError(f"Expected 2D depth map, got shape {depth.shape}")
    if rgb_uint8.shape[:2] != depth.shape:
        raise DepthEstimationError(
            f"RGB shape {rgb_uint8.shape[:2]} does not match depth shape {depth.shape}"
        )

    disc_mask = detect_depth_discontinuities(depth, relative_jump_threshold=discontinuity_threshold)
    disc_frac = float(np.mean(disc_mask))

    if method == "none":
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return DepthRefinementResult(
            refined_depth=depth.copy(),
            discontinuity_mask=disc_mask,
            method="none",
            parameters={},
            processing_time_ms=elapsed_ms,
            discontinuity_fraction=disc_frac,
        )

    # Work in normalized inverse depth [0, 1] for scale-consistent filtering
    safe_depth = np.maximum(depth, 1e-4)
    inv_d = 1.0 / safe_depth
    inv_min = float(np.min(inv_d))
    inv_max = float(np.max(inv_d))
    span = max(inv_max - inv_min, 1e-6)
    norm_inv = ((inv_d - inv_min) / span).astype(np.float32)

    gray_guidance = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    if method == "bilateral":
        filtered_norm = cv2.bilateralFilter(
            norm_inv,
            d=int(bilateral_d),
            sigmaColor=float(bilateral_sigma_color),
            sigmaSpace=float(bilateral_sigma_space),
        )
        params = {
            "bilateral_d": bilateral_d,
            "bilateral_sigma_color": bilateral_sigma_color,
            "bilateral_sigma_space": bilateral_sigma_space,
            "discontinuity_threshold": discontinuity_threshold,
        }
    elif method == "guided":
        filtered_norm = guided_filter_gray(
            guidance=gray_guidance,
            target=norm_inv,
            radius=int(guided_radius),
            eps=float(guided_eps),
        )
        params = {
            "guided_radius": guided_radius,
            "guided_eps": guided_eps,
            "discontinuity_threshold": discontinuity_threshold,
        }
    elif method == "guided_bilateral":
        step1 = cv2.bilateralFilter(
            norm_inv,
            d=int(bilateral_d),
            sigmaColor=float(bilateral_sigma_color),
            sigmaSpace=float(bilateral_sigma_space),
        )
        filtered_norm = guided_filter_gray(
            guidance=gray_guidance,
            target=step1,
            radius=int(guided_radius),
            eps=float(guided_eps),
        )
        params = {
            "bilateral_d": bilateral_d,
            "bilateral_sigma_color": bilateral_sigma_color,
            "guided_radius": guided_radius,
            "guided_eps": guided_eps,
            "discontinuity_threshold": discontinuity_threshold,
        }
    else:
        raise DepthEstimationError(f"Unsupported depth refinement method: {method}")

    filtered_norm = np.clip(filtered_norm, 0.0, 1.0)
    refined_inv = inv_min + filtered_norm * span
    refined_depth = (1.0 / np.maximum(refined_inv, 1e-6)).astype(np.float32)

    # Discontinuity preservation guard: do not interpolate across sharp occlusion jumps
    refined_depth[disc_mask] = depth[disc_mask]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return DepthRefinementResult(
        refined_depth=refined_depth,
        discontinuity_mask=disc_mask,
        method=method,
        parameters=params,
        processing_time_ms=elapsed_ms,
        discontinuity_fraction=disc_frac,
    )
