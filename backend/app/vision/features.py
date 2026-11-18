"""2D feature extraction (SIFT, ORB) and correspondence matching with Lowe's ratio test and mutual check."""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, Literal, Optional, Tuple

import cv2
import numpy as np

from app.core.exceptions import HoloGenError


@dataclass
class FeatureSet:
    """Extracted 2D keypoints and descriptors for a single image view."""

    keypoints_xy: np.ndarray  # float32 (N, 2) [u, v] in working image coordinates
    descriptors: np.ndarray  # float32 (N, 128) for SIFT or uint8 (N, 32) for ORB
    scores: np.ndarray  # float32 (N,) keypoint response strength
    image_size: Tuple[int, int]  # (width, height) at extraction resolution
    scale_from_original: float
    method: str
    extraction_time_ms: float

    @property
    def num_keypoints(self) -> int:
        return int(self.keypoints_xy.shape[0])


@dataclass
class FeatureMatches:
    """Filtered 2D-2D point correspondences between two views."""

    indices1: np.ndarray  # int32 (M,) indices into FeatureSet 1
    indices2: np.ndarray  # int32 (M,) indices into FeatureSet 2
    pts1_xy: np.ndarray  # float32 (M, 2) matched [u1, v1]
    pts2_xy: np.ndarray  # float32 (M, 2) matched [u2, v2]
    distances: np.ndarray  # float32 (M,) descriptor distances
    ratio_threshold: float
    mutual_check: bool
    matching_time_ms: float

    @property
    def num_matches(self) -> int:
        return int(self.indices1.shape[0])

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "num_matches": self.num_matches,
            "ratio_threshold": self.ratio_threshold,
            "mutual_check": self.mutual_check,
            "matching_time_ms": round(self.matching_time_ms, 3),
        }


def extract_features(
    rgb_uint8: np.ndarray,
    method: Literal["sift", "orb"] = "sift",
    max_keypoints: int = 4096,
    sfm_max_dim: int = 1600,
) -> FeatureSet:
    """Extract 2D keypoints and descriptors using SIFT or ORB at up to `sfm_max_dim` resolution.

    Note: Uses OpenCV's SIFT/ORB image feature detectors; all downstream two-view projective
    geometry (normalized 8-point, Sampson RANSAC, E decomposition, cheirality, DLT) is implemented
    from scratch in `app.geometry.two_view`.
    """
    t0 = time.perf_counter()
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise HoloGenError(f"Expected HxWx3 RGB uint8 image, got {rgb_uint8.shape}")

    orig_h, orig_w = int(rgb_uint8.shape[0]), int(rgb_uint8.shape[1])
    max_side = max(orig_w, orig_h)
    if sfm_max_dim > 0 and max_side > sfm_max_dim:
        scale = float(sfm_max_dim) / float(max_side)
        new_w = max(16, int(round(orig_w * scale)))
        new_h = max(16, int(round(orig_h * scale)))
        work_rgb = cv2.resize(rgb_uint8, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        new_w, new_h = orig_w, orig_h
        work_rgb = rgb_uint8

    gray = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2GRAY)

    if method == "sift":
        detector = cv2.SIFT_create(nfeatures=int(max_keypoints))
    elif method == "orb":
        detector = cv2.ORB_create(nfeatures=int(max_keypoints))
    else:
        raise HoloGenError(f"Unsupported feature extraction method: {method}")

    kps, desc = detector.detectAndCompute(gray, None)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if not kps or desc is None or len(kps) == 0:
        return FeatureSet(
            keypoints_xy=np.zeros((0, 2), dtype=np.float32),
            descriptors=np.zeros((0, 128 if method == "sift" else 32), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            image_size=(new_w, new_h),
            scale_from_original=scale,
            method=method,
            extraction_time_ms=elapsed_ms,
        )

    pts = np.array([kp.pt for kp in kps], dtype=np.float32)
    scores = np.array([kp.response for kp in kps], dtype=np.float32)
    if method == "sift":
        # RootSIFT normalization (Arandjelovic & Zisserman, CVPR 2012) improves L2 matching accuracy
        d_f32 = desc.astype(np.float32)
        l1_norm = np.sum(np.abs(d_f32), axis=1, keepdims=True)
        d_f32 = np.sqrt(d_f32 / np.maximum(l1_norm, 1e-8))
        desc_out = d_f32
    else:
        desc_out = desc

    return FeatureSet(
        keypoints_xy=pts,
        descriptors=desc_out,
        scores=scores,
        image_size=(new_w, new_h),
        scale_from_original=scale,
        method=method,
        extraction_time_ms=elapsed_ms,
    )


def match_features(
    feats1: FeatureSet,
    feats2: FeatureSet,
    matcher_type: Literal["flann", "bf"] = "bf",
    ratio_threshold: float = 0.75,
    mutual_check: bool = True,
) -> FeatureMatches:
    """Match descriptors between two views using Lowe's ratio test and optional mutual nearest check."""
    t0 = time.perf_counter()
    if feats1.num_keypoints < 2 or feats2.num_keypoints < 2:
        return FeatureMatches(
            indices1=np.zeros((0,), dtype=np.int32),
            indices2=np.zeros((0,), dtype=np.int32),
            pts1_xy=np.zeros((0, 2), dtype=np.float32),
            pts2_xy=np.zeros((0, 2), dtype=np.float32),
            distances=np.zeros((0,), dtype=np.float32),
            ratio_threshold=ratio_threshold,
            mutual_check=mutual_check,
            matching_time_ms=0.0,
        )

    is_binary = feats1.descriptors.dtype == np.uint8
    if matcher_type == "flann" and not is_binary:
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        norm_type = cv2.NORM_HAMMING if is_binary else cv2.NORM_L2
        matcher = cv2.BFMatcher(norm_type, crossCheck=False)

    knn_12 = matcher.knnMatch(feats1.descriptors, feats2.descriptors, k=2)
    forward_good: Dict[int, Tuple[int, float]] = {}
    for pair in knn_12:
        if len(pair) == 2:
            m, n = pair
            if m.distance < float(ratio_threshold) * n.distance:
                forward_good[int(m.queryIdx)] = (int(m.trainIdx), float(m.distance))

    if mutual_check and len(forward_good) > 0:
        knn_21 = matcher.knnMatch(feats2.descriptors, feats1.descriptors, k=2)
        backward_best: Dict[int, int] = {}
        for pair in knn_21:
            if len(pair) >= 1:
                m = pair[0]
                if len(pair) == 1 or m.distance < float(ratio_threshold) * pair[1].distance:
                    backward_best[int(m.queryIdx)] = int(m.trainIdx)

        idx1_list = []
        idx2_list = []
        dist_list = []
        for i1, (i2, d_val) in forward_good.items():
            if backward_best.get(i2) == i1:
                idx1_list.append(i1)
                idx2_list.append(i2)
                dist_list.append(d_val)
    else:
        idx1_list = list(forward_good.keys())
        idx2_list = [v[0] for v in forward_good.values()]
        dist_list = [v[1] for v in forward_good.values()]

    idx1 = np.asarray(idx1_list, dtype=np.int32)
    idx2 = np.asarray(idx2_list, dtype=np.int32)
    dists = np.asarray(dist_list, dtype=np.float32)

    pts1 = feats1.keypoints_xy[idx1] if idx1.size > 0 else np.zeros((0, 2), dtype=np.float32)
    pts2 = feats2.keypoints_xy[idx2] if idx2.size > 0 else np.zeros((0, 2), dtype=np.float32)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return FeatureMatches(
        indices1=idx1,
        indices2=idx2,
        pts1_xy=pts1,
        pts2_xy=pts2,
        distances=dists,
        ratio_threshold=ratio_threshold,
        mutual_check=mutual_check,
        matching_time_ms=elapsed_ms,
    )


def render_feature_matches(
    rgb1: np.ndarray,
    rgb2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    max_draw: int = 150,
) -> np.ndarray:
    """Render side-by-side feature correspondences (green=inliers, red=outliers) to an RGB image."""
    h1, w1 = rgb1.shape[:2]
    h2, w2 = rgb2.shape[:2]
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = rgb1[:, :, ::-1]  # BGR for cv2 drawing
    canvas[:h2, w1 : w1 + w2] = rgb2[:, :, ::-1]

    n = len(pts1)
    if n > 0:
        step = max(1, n // max_draw)
        for i in range(0, n, step):
            p1 = (int(round(float(pts1[i, 0]))), int(round(float(pts1[i, 1]))))
            p2 = (int(round(float(pts2[i, 0]))) + w1, int(round(float(pts2[i, 1]))))
            is_inlier = True if inlier_mask is None else bool(inlier_mask[i])
            color = (60, 235, 90) if is_inlier else (60, 60, 235)
            cv2.circle(canvas, p1, 3, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, p2, 3, color, -1, lineType=cv2.LINE_AA)
            cv2.line(canvas, p1, p2, color, 1, lineType=cv2.LINE_AA)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    return canvas[:, :, ::-1]
