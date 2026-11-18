"""Monocular depth uncertainty proxy estimation.

IMPORTANT: Pretrained deterministic monocular depth networks do not emit calibrated
statistical confidence intervals. All maps and statistics produced here are explicitly
named and treated as an UNCERTAINTY PROXY (combining spatial depth-gradient discontinuity,
horizontal-flip consistency, and optional inter-model disagreement).
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

from app.core.exceptions import DepthEstimationError


@dataclass
class UncertaintyProxyResult:
    """Container for monocular depth uncertainty proxy maps and statistics."""

    uncertainty_proxy_map: np.ndarray  # float32 (H, W) in [0, 1] (0 = low uncertainty, 1 = high)
    gradient_proxy_map: np.ndarray  # float32 (H, W) in [0, 1]
    flip_proxy_map: Optional[np.ndarray]  # float32 (H, W) in [0, 1] if computed
    inter_model_proxy_map: Optional[np.ndarray]  # float32 (H, W) in [0, 1] if computed
    reliable_mask: np.ndarray  # bool (H, W) where uncertainty_proxy_map <= threshold
    threshold: float
    components_used: list[str]
    processing_time_ms: float

    @property
    def mean_uncertainty_proxy(self) -> float:
        return float(np.mean(self.uncertainty_proxy_map))

    @property
    def median_uncertainty_proxy(self) -> float:
        return float(np.median(self.uncertainty_proxy_map))

    @property
    def p90_uncertainty_proxy(self) -> float:
        return float(np.percentile(self.uncertainty_proxy_map, 90.0))

    @property
    def p95_uncertainty_proxy(self) -> float:
        return float(np.percentile(self.uncertainty_proxy_map, 95.0))

    @property
    def p99_uncertainty_proxy(self) -> float:
        return float(np.percentile(self.uncertainty_proxy_map, 99.0))

    @property
    def reliable_pixel_fraction(self) -> float:
        return float(np.mean(self.reliable_mask))

    def histogram_summary(self) -> Dict[str, float]:
        """Compute percentage of pixels falling into interpretable uncertainty proxy bins."""
        u = self.uncertainty_proxy_map.ravel()
        n = max(float(u.size), 1.0)
        edges = [0.0, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0001]
        labels = [
            "[0.00, 0.02)",
            "[0.02, 0.05)",
            "[0.05, 0.10)",
            "[0.10, 0.25)",
            "[0.25, 0.50)",
            "[0.50, 0.75)",
            "[0.75, 1.00]",
        ]
        counts, _ = np.histogram(u, bins=edges)
        return {label: round(float(c) * 100.0 / n, 2) for label, c in zip(labels, counts)}

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "is_calibrated_probability": False,
            "semantics": "monocular_uncertainty_proxy",
            "components_used": self.components_used,
            "threshold": round(self.threshold, 4),
            "mean_uncertainty_proxy": round(self.mean_uncertainty_proxy, 4),
            "median_uncertainty_proxy": round(self.median_uncertainty_proxy, 4),
            "p90_uncertainty_proxy": round(self.p90_uncertainty_proxy, 4),
            "p95_uncertainty_proxy": round(self.p95_uncertainty_proxy, 4),
            "p99_uncertainty_proxy": round(self.p99_uncertainty_proxy, 4),
            "reliable_pixel_fraction": round(self.reliable_pixel_fraction, 4),
            "histogram_pct": self.histogram_summary(),
            "processing_time_ms": round(self.processing_time_ms, 3),
        }


def _normalize_robust_01(arr: np.ndarray, percentile_high: float = 98.0) -> np.ndarray:
    """Normalize a non-negative 2D array into [0, 1] using robust upper percentile."""
    a = np.maximum(np.asarray(arr, dtype=np.float32), 0.0)
    hi = float(np.percentile(a, percentile_high))
    if hi < 1e-6:
        hi = float(np.max(a))
    if hi < 1e-6:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip(a / hi, 0.0, 1.0).astype(np.float32)


def compute_gradient_uncertainty_proxy(depth_map: np.ndarray) -> np.ndarray:
    """Compute normalized spatial gradient magnitude of log-depth (scale-invariant boundary cue)."""
    d = np.maximum(np.asarray(depth_map, dtype=np.float32), 1e-4)
    log_d = np.log(d)
    gx = cv2.Sobel(log_d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(log_d, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return _normalize_robust_01(mag, percentile_high=98.0)


def compute_uncertainty_proxy(
    depth_map: np.ndarray,
    flipped_depth_map: Optional[np.ndarray] = None,
    secondary_depth_map: Optional[np.ndarray] = None,
    uncertainty_threshold: float = 0.65,
) -> UncertaintyProxyResult:
    """Compute a composite monocular uncertainty proxy map in [0, 1].

    Args:
        depth_map: Primary depth map (H, W).
        flipped_depth_map: Optional horizontally unflipped depth prediction from flipped input RGB.
        secondary_depth_map: Optional depth map from a second model for inter-model disagreement.
        uncertainty_threshold: Cutoff in [0, 1] above which pixels are marked unreliable.
    """
    t0 = time.perf_counter()
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim != 2:
        raise DepthEstimationError(f"Expected 2D depth map, got shape {depth.shape}")

    grad_proxy = compute_gradient_uncertainty_proxy(depth)
    components = ["depth_gradient_discontinuity"]
    weights = [0.5]
    maps = [grad_proxy]

    flip_proxy: Optional[np.ndarray] = None
    if flipped_depth_map is not None:
        fd = np.asarray(flipped_depth_map, dtype=np.float32)
        if fd.shape != depth.shape:
            raise DepthEstimationError(
                f"flipped_depth_map shape {fd.shape} != depth_map shape {depth.shape}"
            )
        rel_diff = np.abs(depth - fd) / np.maximum(depth, 1e-4)
        flip_proxy = _normalize_robust_01(rel_diff, percentile_high=98.0)
        components.append("horizontal_flip_consistency")
        weights.append(0.3)
        maps.append(flip_proxy)

    inter_proxy: Optional[np.ndarray] = None
    if secondary_depth_map is not None:
        sd = np.asarray(secondary_depth_map, dtype=np.float32)
        if sd.shape != depth.shape:
            raise DepthEstimationError(
                f"secondary_depth_map shape {sd.shape} != depth_map shape {depth.shape}"
            )
        # Align secondary depth to primary via least-squares scale-shift before measuring disagreement
        A = np.vstack([sd.ravel(), np.ones(sd.size, dtype=np.float32)]).T
        sol, _, _, _ = np.linalg.lstsq(A, depth.ravel(), rcond=None)
        sd_aligned = sol[0] * sd + sol[1]
        rel_disagreement = np.abs(depth - sd_aligned) / np.maximum(depth, 1e-4)
        inter_proxy = _normalize_robust_01(rel_disagreement, percentile_high=98.0)
        components.append("inter_model_disagreement")
        weights.append(0.2)
        maps.append(inter_proxy)

    w_arr = np.asarray(weights, dtype=np.float32)
    w_arr = w_arr / np.sum(w_arr)

    combined = np.zeros_like(depth, dtype=np.float32)
    for w_i, m_i in zip(w_arr, maps):
        combined += float(w_i) * m_i
    combined = np.clip(combined, 0.0, 1.0)

    reliable_mask = combined <= float(uncertainty_threshold)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return UncertaintyProxyResult(
        uncertainty_proxy_map=combined,
        gradient_proxy_map=grad_proxy,
        flip_proxy_map=flip_proxy,
        inter_model_proxy_map=inter_proxy,
        reliable_mask=reliable_mask,
        threshold=float(uncertainty_threshold),
        components_used=components,
        processing_time_ms=elapsed_ms,
    )
