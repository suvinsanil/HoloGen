"""From-scratch two-view epipolar geometry and triangulation (Phase 7a), plus OpenCV & pycolmap baselines.

All core geometry primitives in this module are implemented from scratch using pure NumPy SVD
and linear algebra without calling OpenCV's geometry solvers:
- `normalize_points_2d`: Hartley isotropic point normalization
- `eight_point_fundamental_matrix`: Normalized 8-point algorithm with SVD rank-2 projection
- `compute_sampson_distance`: First-order Sampson epipolar error
- `ransac_fundamental_matrix`: Adaptive RANSAC loop with Sampson error and inlier refit
- `essential_from_fundamental`: Essential matrix from F and intrinsics K1, K2 with diag(1,1,0) projection
- `decompose_essential_matrix`: Exact 4-hypothesis (R, t) SVD decomposition on SO(3)
- `triangulate_points_dlt`: Linear DLT homogeneous SVD triangulation
- `recover_pose_cheirality`: Cheirality disambiguation (Z1 > 0 and Z2 > 0) and reprojection error

OpenCV (`cv2.findEssentialMat` / `cv2.recoverPose`) and `pycolmap` (`estimate_calibrated_two_view_geometry`)
are included strictly in dedicated baseline comparison functions at the bottom of this file.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.exceptions import GeometryProcessingError
from app.vision.camera import CameraIntrinsics


@dataclass
class TwoViewGeometryResult:
    """Complete output of two-view relative pose and sparse 3D triangulation."""

    method: str  # "from_scratch_8pt_sampson", "opencv_5pt_baseline", "pycolmap_baseline"
    F: np.ndarray  # float64 (3, 3) Fundamental matrix
    E: np.ndarray  # float64 (3, 3) Essential matrix
    R: np.ndarray  # float64 (3, 3) Relative rotation (Camera 1 -> Camera 2: X_cam2 = R @ X_cam1 + t)
    t: np.ndarray  # float64 (3,) Unit relative translation vector (||t|| = 1)
    inlier_mask: np.ndarray  # bool (N,) RANSAC inliers
    cheirality_mask: np.ndarray  # bool (N,) inliers with positive depth in both views
    points_3d: np.ndarray  # float64 (M, 3) triangulated 3D points in Camera 1 frame for cheirality_mask
    reprojection_errors_px: np.ndarray  # float64 (M,) symmetric reprojection error in pixels
    mean_reprojection_error_px: float
    median_reprojection_error_px: float
    rmse_reprojection_error_px: float
    num_correspondences: int
    num_inliers: int
    num_cheirality_valid: int
    ransac_iterations: int
    runtime_ms: float

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "num_correspondences": self.num_correspondences,
            "num_inliers": self.num_inliers,
            "inlier_ratio": round(
                float(self.num_inliers) / max(float(self.num_correspondences), 1.0), 4
            ),
            "num_cheirality_valid": self.num_cheirality_valid,
            "mean_reprojection_error_px": round(self.mean_reprojection_error_px, 4),
            "median_reprojection_error_px": round(self.median_reprojection_error_px, 4),
            "rmse_reprojection_error_px": round(self.rmse_reprojection_error_px, 4),
            "ransac_iterations": self.ransac_iterations,
            "runtime_ms": round(self.runtime_ms, 3),
        }


def normalize_points_2d(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley isotropic normalization of 2D point coordinates (Hartley, TPAMI 1997).

    Translates points so their centroid is at (0, 0) and scales isotropically so the average
    Euclidean distance from the origin equals sqrt(2).

    Returns:
        (normalized_pts_hom, T) where normalized_pts_hom is (N, 3) and T is (3, 3).
    """
    p = np.asarray(pts, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 1:
        raise GeometryProcessingError(f"Expected Nx2 point array, got shape {p.shape}")

    centroid = np.mean(p, axis=0)
    shifted = p - centroid
    dists = np.linalg.norm(shifted, axis=1)
    mean_dist = float(np.mean(dists))
    scale = math.sqrt(2.0) / max(mean_dist, 1e-12)

    T = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ones = np.ones((p.shape[0], 1), dtype=np.float64)
    p_hom = np.hstack([p, ones])
    norm_hom = (T @ p_hom.T).T
    return norm_hom, T


def eight_point_fundamental_matrix(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """From-scratch Normalized 8-Point Algorithm for Fundamental Matrix estimation.

    Solves x2_i^T F x1_i = 0 via SVD on Hartley-normalized coordinates, enforces the rank-2
    internal constraint det(F) = 0 by zeroing the smallest singular value, and denormalizes.
    """
    p1 = np.asarray(pts1, dtype=np.float64)
    p2 = np.asarray(pts2, dtype=np.float64)
    if p1.shape != p2.shape or p1.ndim != 2 or p1.shape[1] != 2 or p1.shape[0] < 8:
        raise GeometryProcessingError(
            f"Normalized 8-point algorithm requires at least 8 Nx2 correspondences, got {p1.shape}"
        )

    x1, T1 = normalize_points_2d(p1)
    x2, T2 = normalize_points_2d(p2)

    u1, v1 = x1[:, 0], x1[:, 1]
    u2, v2 = x2[:, 0], x2[:, 1]

    # Construct N x 9 design matrix A where each row satisfies x2^T F_tilde x1 = 0
    A = np.column_stack(
        [
            u2 * u1,
            u2 * v1,
            u2,
            v2 * u1,
            v2 * v1,
            v2,
            u1,
            v1,
            np.ones_like(u1),
        ]
    )

    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    F_tilde = Vt[-1].reshape(3, 3)

    # Enforce internal rank-2 constraint (det(F) = 0) via SVD projection
    Uf, Sf, Vtf = np.linalg.svd(F_tilde)
    Sf[2] = 0.0
    F_rank2 = Uf @ np.diag(Sf) @ Vtf

    # Denormalize: F = T2^T F_rank2 T1
    F = T2.T @ F_rank2 @ T1
    norm_f = float(np.linalg.norm(F))
    if norm_f > 1e-15:
        F = F / norm_f
    return F


def compute_sampson_distance(
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: np.ndarray,
) -> np.ndarray:
    """Compute first-order Sampson epipolar distance (in pixels) for correspondences (pts1, pts2).

    Sampson squared error:
        d_S^2 = (x2^T F x1)^2 / ((F x1)_0^2 + (F x1)_1^2 + (F^T x2)_0^2 + (F^T x2)_1^2)
    Returns:
        1D array of non-negative Sampson distances sqrt(d_S^2) in pixels.
    """
    p1 = np.asarray(pts1, dtype=np.float64)
    p2 = np.asarray(pts2, dtype=np.float64)
    n = p1.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)
    x1 = np.hstack([p1, ones])  # (N, 3)
    x2 = np.hstack([p2, ones])  # (N, 3)

    Fx1 = (F @ x1.T).T  # (N, 3)
    Ftx2 = (F.T @ x2.T).T  # (N, 3)
    alg_err = np.sum(x2 * Fx1, axis=1)  # x2^T F x1

    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    sampson_sq = (alg_err ** 2) / np.maximum(denom, 1e-15)
    return np.sqrt(np.maximum(sampson_sq, 0.0))


def essential_from_fundamental(
    F: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
) -> np.ndarray:
    """Compute Essential Matrix E = K2^T F K1 and project onto the Essential manifold diag(1, 1, 0)."""
    E_raw = K2.T @ F @ K1
    U, _, Vt = np.linalg.svd(E_raw)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1.0
    E = U @ np.diag([1.0, 1.0, 0.0]) @ Vt
    return E


def ransac_fundamental_matrix(
    pts1: np.ndarray,
    pts2: np.ndarray,
    sampson_threshold_px: float = 1.5,
    max_iterations: int = 1200,
    confidence: float = 0.999,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """From-scratch RANSAC loop estimating Fundamental Matrix F using the normalized 8-point
    algorithm and Sampson epipolar error, followed by an inlier least-squares refit.

    Returns:
        (F_best, inlier_mask, iterations_executed)
    """
    p1 = np.asarray(pts1, dtype=np.float64)
    p2 = np.asarray(pts2, dtype=np.float64)
    n = p1.shape[0]
    if n < 8:
        raise GeometryProcessingError(f"RANSAC requires at least 8 correspondences, got {n}")

    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    best_score = float("inf")
    best_F = np.eye(3, dtype=np.float64)

    target_iters = int(max_iterations)
    it = 0
    thresh = float(sampson_threshold_px)

    while it < target_iters and it < max_iterations:
        it += 1
        sample_idx = rng.choice(n, size=8, replace=False)
        try:
            F_cand = eight_point_fundamental_matrix(p1[sample_idx], p2[sample_idx])
        except Exception:
            continue

        dists = compute_sampson_distance(p1, p2, F_cand)
        inliers = dists <= thresh
        count = int(np.sum(inliers))
        if count < 8:
            continue
        # MSAC (M-estimator Sample Consensus, Torr & Zisserman 2000) truncated squared loss:
        # avoids picking a noisy candidate that accidentally grazes an outlier near `thresh`.
        msac_score = float(np.sum(np.minimum(dists ** 2, thresh ** 2)))

        if msac_score < best_score:
            best_count = count
            best_score = msac_score
            best_inliers = inliers
            best_F = F_cand

            inlier_ratio = float(count) / float(n)
            if inlier_ratio > 0.05:
                prob_all_inliers = inlier_ratio ** 8
                if 1e-12 < prob_all_inliers < 1.0:
                    adaptive = math.log(1.0 - confidence) / math.log(1.0 - prob_all_inliers)
                    target_iters = min(max_iterations, max(80, int(math.ceil(adaptive))))

    # Two-stage trimmed least-squares refit on consensus inliers (rejecting boundary grazers)
    if best_count >= 8:
        for trim_factor in (1.0, 0.75):
            dists_curr = compute_sampson_distance(p1, p2, best_F)
            core_mask = dists_curr <= (thresh * trim_factor)
            if int(np.sum(core_mask)) < 8:
                core_mask = best_inliers
            F_refit = eight_point_fundamental_matrix(p1[core_mask], p2[core_mask])
            refit_dists = compute_sampson_distance(p1, p2, F_refit)
            refit_score = float(np.sum(np.minimum(refit_dists ** 2, thresh ** 2)))
            if refit_score <= best_score:
                best_score = refit_score
                best_F = F_refit
                best_inliers = refit_dists <= thresh
                best_count = int(np.sum(best_inliers))

    return best_F, best_inliers, it


def decompose_essential_matrix(
    E: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Decompose an Essential Matrix E into the 4 candidate (R, t) pairs in SO(3) x S^2."""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1.0

    W = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t1 = U[:, 2].copy()
    t2 = -U[:, 2].copy()

    return [
        (R1, t1),
        (R1, t2),
        (R2, t1),
        (R2, t2),
    ]


def triangulate_points_dlt(
    pts1: np.ndarray,
    pts2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> np.ndarray:
    """From-scratch Linear DLT (Direct Linear Transform) triangulation via SVD.

    For each correspondence (u1, v1) <-> (u2, v2) and 3x4 projection matrices P1, P2,
    solves the 4x4 homogeneous system A_i X_i = 0 with row-norm preconditioning.

    Returns:
        Nx3 array of 3D coordinates (X, Y, Z) in the world/Camera-1 frame.
    """
    p1 = np.asarray(pts1, dtype=np.float64)
    p2 = np.asarray(pts2, dtype=np.float64)
    n = p1.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    pts_3d = np.zeros((n, 3), dtype=np.float64)
    p1_r0, p1_r1, p1_r2 = P1[0], P1[1], P1[2]
    p2_r0, p2_r1, p2_r2 = P2[0], P2[1], P2[2]

    for i in range(n):
        u1, v1 = p1[i, 0], p1[i, 1]
        u2, v2 = p2[i, 0], p2[i, 1]
        A = np.vstack(
            [
                u1 * p1_r2 - p1_r0,
                v1 * p1_r2 - p1_r1,
                u2 * p2_r2 - p2_r0,
                v2 * p2_r2 - p2_r1,
            ]
        )
        # Precondition rows to unit L2 norm for numerical stability
        row_norms = np.linalg.norm(A, axis=1, keepdims=True)
        A = A / np.maximum(row_norms, 1e-12)
        _, _, Vt = np.linalg.svd(A)
        X_hom = Vt[-1]
        w = X_hom[3]
        if abs(w) < 1e-12:
            pts_3d[i] = [0.0, 0.0, -1.0]
        else:
            pts_3d[i] = X_hom[:3] / w

    return pts_3d


def compute_reprojection_errors(
    pts_3d: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Compute symmetric RMS reprojection error in pixels across Camera 1 [I|0] and Camera 2 [R|t]."""
    if pts_3d.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)

    # Project into Camera 1: x1_proj = K1 @ X
    z1 = np.maximum(pts_3d[:, 2:3], 1e-8)
    uv1_proj = (K1[:2, :2] @ (pts_3d[:, :2] / z1).T).T + K1[:2, 2]
    err1_sq = np.sum((uv1_proj - pts1) ** 2, axis=1)

    # Project into Camera 2: X_cam2 = (R @ X^T)^T + t
    pts_cam2 = (R @ pts_3d.T).T + t[None, :]
    z2 = np.maximum(pts_cam2[:, 2:3], 1e-8)
    uv2_proj = (K2[:2, :2] @ (pts_cam2[:, :2] / z2).T).T + K2[:2, 2]
    err2_sq = np.sum((uv2_proj - pts2) ** 2, axis=1)

    return np.sqrt(0.5 * (err1_sq + err2_sq))


def recover_pose_cheirality(
    E: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
    max_depth_ratio: float = 200.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select the unique physically valid (R, t) from the 4 Essential Matrix decompositions
    using the cheirality check (positive depth Z1 > 0 and Z2 > 0 in both cameras).

    Returns:
        (R_best, t_best, triangulated_3d_all, cheirality_valid_mask)
    """
    p1 = np.asarray(pts1, dtype=np.float64)
    p2 = np.asarray(pts2, dtype=np.float64)
    n = p1.shape[0]
    mask = np.ones(n, dtype=bool) if inlier_mask is None else inlier_mask.astype(bool)

    P1 = K1 @ np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)])
    candidates = decompose_essential_matrix(E)

    best_R = candidates[0][0]
    best_t = candidates[0][1]
    best_X = np.zeros((n, 3), dtype=np.float64)
    best_cheirality = np.zeros(n, dtype=bool)
    best_valid_count = -1

    for R_cand, t_cand in candidates:
        P2 = K2 @ np.hstack([R_cand, t_cand.reshape(3, 1)])
        X_all = triangulate_points_dlt(p1, p2, P1, P2)
        X_cam2 = (R_cand @ X_all.T).T + t_cand[None, :]

        z1 = X_all[:, 2]
        z2 = X_cam2[:, 2]
        valid = (
            mask
            & np.all(np.isfinite(X_all), axis=1)
            & (z1 > 1e-3)
            & (z2 > 1e-3)
            & (z1 < max_depth_ratio)
            & (z2 < max_depth_ratio)
        )
        count = int(np.sum(valid))
        if count > best_valid_count:
            best_valid_count = count
            best_R = R_cand
            best_t = t_cand
            best_X = X_all
            best_cheirality = valid

    return best_R, best_t, best_X, best_cheirality


def estimate_two_view_geometry_from_scratch(
    pts1: np.ndarray,
    pts2: np.ndarray,
    intrinsics1: CameraIntrinsics,
    intrinsics2: Optional[CameraIntrinsics] = None,
    sampson_threshold_px: float = 1.5,
    max_ransac_iterations: int = 1200,
    seed: int = 42,
) -> TwoViewGeometryResult:
    """Full from-scratch two-view geometry pipeline (normalized 8-pt + Sampson RANSAC + E + Cheirality + DLT)."""
    t0 = time.perf_counter()
    intr2 = intrinsics2 or intrinsics1
    K1 = intrinsics1.K
    K2 = intr2.K

    F, inliers, iters = ransac_fundamental_matrix(
        pts1=pts1,
        pts2=pts2,
        sampson_threshold_px=sampson_threshold_px,
        max_iterations=max_ransac_iterations,
        seed=seed,
    )
    E = essential_from_fundamental(F, K1, K2)
    R, t, X_all, cheirality_mask = recover_pose_cheirality(
        E=E,
        pts1=pts1,
        pts2=pts2,
        K1=K1,
        K2=K2,
        inlier_mask=inliers,
    )

    valid_X = X_all[cheirality_mask]
    reproj_err = compute_reprojection_errors(
        pts_3d=valid_X,
        pts1=pts1[cheirality_mask],
        pts2=pts2[cheirality_mask],
        K1=K1,
        K2=K2,
        R=R,
        t=t,
    )

    if reproj_err.size > 0:
        mean_err = float(np.mean(reproj_err))
        med_err = float(np.median(reproj_err))
        rmse_err = float(np.sqrt(np.mean(reproj_err ** 2)))
    else:
        mean_err = med_err = rmse_err = float("inf")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return TwoViewGeometryResult(
        method="from_scratch_8pt_sampson",
        F=F,
        E=E,
        R=R,
        t=t,
        inlier_mask=inliers,
        cheirality_mask=cheirality_mask,
        points_3d=valid_X,
        reprojection_errors_px=reproj_err,
        mean_reprojection_error_px=mean_err,
        median_reprojection_error_px=med_err,
        rmse_reprojection_error_px=rmse_err,
        num_correspondences=int(len(pts1)),
        num_inliers=int(np.sum(inliers)),
        num_cheirality_valid=int(np.sum(cheirality_mask)),
        ransac_iterations=iters,
        runtime_ms=elapsed_ms,
    )


def compute_pose_errors_deg(
    R_est: np.ndarray,
    t_est: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
) -> Dict[str, float]:
    """Compute relative rotation angular error (deg) and translation direction angular error (deg)."""
    R_diff = R_gt.T @ R_est
    trace_val = float(np.trace(R_diff))
    cos_r = np.clip((trace_val - 1.0) * 0.5, -1.0, 1.0)
    rot_err_deg = math.degrees(math.acos(cos_r))

    t1 = np.asarray(t_est, dtype=np.float64).ravel()
    t2 = np.asarray(t_gt, dtype=np.float64).ravel()
    n1 = float(np.linalg.norm(t1))
    n2 = float(np.linalg.norm(t2))
    if n1 < 1e-12 or n2 < 1e-12:
        trans_err_deg = 180.0
    else:
        cos_t = float(np.clip(np.dot(t1 / n1, t2 / n2), -1.0, 1.0))
        trans_err_deg = math.degrees(math.acos(cos_t))

    return {
        "rotation_error_deg": round(rot_err_deg, 4),
        "translation_direction_error_deg": round(trans_err_deg, 4),
        "max_pose_error_deg": round(max(rot_err_deg, trans_err_deg), 4),
    }


def render_epipolar_lines(
    rgb1: np.ndarray,
    rgb2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: np.ndarray,
    output_path: Optional[Path] = None,
    num_lines: int = 25,
    seed: int = 42,
) -> np.ndarray:
    """Draw corresponding points and epipolar lines l2 = F x1 (in Image 2) and l1 = F^T x2 (in Image 1)."""
    img1 = rgb1[:, :, ::-1].copy()
    img2 = rgb2[:, :, ::-1].copy()
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    n = len(pts1)
    rng = np.random.default_rng(seed)
    indices = np.arange(n) if n <= num_lines else rng.choice(n, size=num_lines, replace=False)

    for idx in indices:
        color = tuple(int(c) for c in rng.integers(60, 255, size=3))
        u1, v1 = float(pts1[idx, 0]), float(pts1[idx, 1])
        u2, v2 = float(pts2[idx, 0]), float(pts2[idx, 1])

        # Epipolar line in Image 2: l2 = F @ [u1, v1, 1]^T = (a, b, c) -> a*u + b*v + c = 0
        l2 = F @ np.array([u1, v1, 1.0], dtype=np.float64)
        if abs(l2[1]) > 1e-8:
            y0 = int(round(-l2[2] / l2[1]))
            y1 = int(round(-(l2[2] + l2[0] * w2) / l2[1]))
            cv2.line(img2, (0, y0), (w2, y1), color, 1, lineType=cv2.LINE_AA)

        # Epipolar line in Image 1: l1 = F^T @ [u2, v2, 1]^T
        l1 = F.T @ np.array([u2, v2, 1.0], dtype=np.float64)
        if abs(l1[1]) > 1e-8:
            y0 = int(round(-l1[2] / l1[1]))
            y1 = int(round(-(l1[2] + l1[0] * w1) / l1[1]))
            cv2.line(img1, (0, y0), (w1, y1), color, 1, lineType=cv2.LINE_AA)

        cv2.circle(img1, (int(round(u1)), int(round(v1))), 4, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img2, (int(round(u2)), int(round(v2))), 4, color, -1, lineType=cv2.LINE_AA)

    canvas_h = max(h1, h2)
    canvas = np.zeros((canvas_h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1 : w1 + w2] = img2

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas[:, :, ::-1]


# ==============================================================================
# COMPARISON BASELINES ONLY (OpenCV 5-point RANSAC and pycolmap Two-View)
# ==============================================================================


def estimate_two_view_opencv_baseline(
    pts1: np.ndarray,
    pts2: np.ndarray,
    intrinsics1: CameraIntrinsics,
    intrinsics2: Optional[CameraIntrinsics] = None,
    ransac_threshold_px: float = 1.5,
) -> TwoViewGeometryResult:
    """Comparison baseline using OpenCV's `cv2.findEssentialMat` and `cv2.recoverPose`."""
    t0 = time.perf_counter()
    intr2 = intrinsics2 or intrinsics1
    K1 = intrinsics1.K
    K2 = intr2.K

    p1 = np.asarray(pts1, dtype=np.float64)
    p2 = np.asarray(pts2, dtype=np.float64)

    E, mask_e = cv2.findEssentialMat(
        p1,
        p2,
        cameraMatrix=K1,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=float(ransac_threshold_px),
    )
    if E is None or E.shape != (3, 3):
        E = E[:3, :3] if E is not None else np.eye(3, dtype=np.float64)
    inliers = (
        mask_e.ravel().astype(bool)
        if mask_e is not None
        else np.ones(len(p1), dtype=bool)
    )

    _, R, t_vec, mask_pose = cv2.recoverPose(E, p1, p2, cameraMatrix=K1, mask=mask_e.copy() if mask_e is not None else None)
    t = t_vec.ravel()
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    nf = float(np.linalg.norm(F))
    if nf > 1e-15:
        F /= nf

    cheirality_mask = (
        (mask_pose.ravel() > 0) & inliers
        if mask_pose is not None
        else inliers
    )
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, t.reshape(3, 1)])
    X_all = triangulate_points_dlt(p1, p2, P1, P2)
    valid_X = X_all[cheirality_mask]
    reproj_err = compute_reprojection_errors(
        valid_X, p1[cheirality_mask], p2[cheirality_mask], K1, K2, R, t
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return TwoViewGeometryResult(
        method="opencv_5pt_baseline",
        F=F,
        E=E,
        R=R,
        t=t,
        inlier_mask=inliers,
        cheirality_mask=cheirality_mask,
        points_3d=valid_X,
        reprojection_errors_px=reproj_err,
        mean_reprojection_error_px=float(np.mean(reproj_err)) if reproj_err.size > 0 else float("inf"),
        median_reprojection_error_px=float(np.median(reproj_err)) if reproj_err.size > 0 else float("inf"),
        rmse_reprojection_error_px=float(np.sqrt(np.mean(reproj_err ** 2))) if reproj_err.size > 0 else float("inf"),
        num_correspondences=len(p1),
        num_inliers=int(np.sum(inliers)),
        num_cheirality_valid=int(np.sum(cheirality_mask)),
        ransac_iterations=0,
        runtime_ms=elapsed_ms,
    )


def estimate_two_view_pycolmap_baseline(
    pts1: np.ndarray,
    pts2: np.ndarray,
    intrinsics1: CameraIntrinsics,
    intrinsics2: Optional[CameraIntrinsics] = None,
    max_error_px: float = 1.5,
) -> TwoViewGeometryResult:
    """Comparison baseline using `pycolmap.estimate_calibrated_two_view_geometry`."""
    import pycolmap

    t0 = time.perf_counter()
    intr2 = intrinsics2 or intrinsics1
    K1 = intrinsics1.K
    K2 = intr2.K
    p1 = np.asarray(pts1, dtype=np.float64)
    p2 = np.asarray(pts2, dtype=np.float64)

    cam1 = pycolmap.Camera(
        model="PINHOLE",
        width=int(intrinsics1.width),
        height=int(intrinsics1.height),
        params=[float(intrinsics1.fx), float(intrinsics1.fy), float(intrinsics1.cx), float(intrinsics1.cy)],
    )
    cam2 = pycolmap.Camera(
        model="PINHOLE",
        width=int(intr2.width),
        height=int(intr2.height),
        params=[float(intr2.fx), float(intr2.fy), float(intr2.cx), float(intr2.cy)],
    )
    matches = np.column_stack([np.arange(len(p1), dtype=np.uint32), np.arange(len(p2), dtype=np.uint32)])
    opts = pycolmap.TwoViewGeometryOptions()
    opts.ransac.max_error = float(max_error_px)
    opts.compute_relative_pose = True

    res = pycolmap.estimate_calibrated_two_view_geometry(cam1, p1, cam2, p2, matches, opts)
    # cam2_from_cam1 is Rigid3d with rotation and translation
    R = np.asarray(res.cam2_from_cam1.rotation.matrix(), dtype=np.float64)
    t_raw = np.asarray(res.cam2_from_cam1.translation, dtype=np.float64).ravel()
    t_norm = float(np.linalg.norm(t_raw))
    t = t_raw / t_norm if t_norm > 1e-12 else t_raw

    E = np.asarray(res.E, dtype=np.float64)
    F = np.asarray(res.F, dtype=np.float64)

    inliers = np.zeros(len(p1), dtype=bool)
    if res.inlier_matches is not None and len(res.inlier_matches) > 0:
        inlier_idx = np.asarray(res.inlier_matches)[:, 0].astype(np.int64)
        inliers[inlier_idx] = True

    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, t.reshape(3, 1)])
    X_all = triangulate_points_dlt(p1, p2, P1, P2)
    X_cam2 = (R @ X_all.T).T + t[None, :]
    cheirality_mask = inliers & (X_all[:, 2] > 1e-3) & (X_cam2[:, 2] > 1e-3)
    valid_X = X_all[cheirality_mask]
    reproj_err = compute_reprojection_errors(
        valid_X, p1[cheirality_mask], p2[cheirality_mask], K1, K2, R, t
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return TwoViewGeometryResult(
        method="pycolmap_baseline",
        F=F,
        E=E,
        R=R,
        t=t,
        inlier_mask=inliers,
        cheirality_mask=cheirality_mask,
        points_3d=valid_X,
        reprojection_errors_px=reproj_err,
        mean_reprojection_error_px=float(np.mean(reproj_err)) if reproj_err.size > 0 else float("inf"),
        median_reprojection_error_px=float(np.median(reproj_err)) if reproj_err.size > 0 else float("inf"),
        rmse_reprojection_error_px=float(np.sqrt(np.mean(reproj_err ** 2))) if reproj_err.size > 0 else float("inf"),
        num_correspondences=len(p1),
        num_inliers=int(np.sum(inliers)),
        num_cheirality_valid=int(np.sum(cheirality_mask)),
        ransac_iterations=0,
        runtime_ms=elapsed_ms,
    )
