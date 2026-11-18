"""N-view Incremental Structure-from-Motion (SfM) via pycolmap and Sim(3) Umeyama alignment (Phase 7b).

Key engineering features:
1. Single shared camera model (`CameraMode.SINGLE`) for same-camera image sets.
2. Separate `sfm_max_image_size` (default 1600 px) independent of the 640 px monocular depth cap.
3. CPU-friendly SIFT extraction, exhaustive matching, and incremental bundle adjustment.
4. Full telemetry: registered-image count vs. total images, mean track length, mean reprojection
   error (px), stage-wise and total runtime (ms), and peak process RAM (MB).
5. Pure-NumPy Sim(3) Umeyama alignment (`align_umeyama_sim3`) for evaluating recovered camera
   trajectories and aligning multi-view reconstructions to ground-truth scans.
"""

from dataclasses import dataclass, field
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
import psutil

from app.core.exceptions import GeometryProcessingError
from app.core.logging import get_logger
from app.vision.camera import CameraIntrinsics

logger = get_logger(__name__)


@dataclass
class Sim3AlignmentResult:
    """Result of Sim(3) Umeyama similarity alignment: y_hat = scale * (R @ x) + t."""

    scale: float
    R: np.ndarray  # float64 (3, 3) in SO(3)
    t: np.ndarray  # float64 (3,)
    aligned_points: np.ndarray  # float64 (N, 3)
    residuals: np.ndarray  # float64 (N,) Euclidean distances ||y_hat_i - dst_i||_2
    ate_rmse: float
    ate_mean: float
    ate_median: float

    def transform_points(self, pts: np.ndarray) -> np.ndarray:
        """Apply y = scale * (R @ x) + t to an (M, 3) point array."""
        p = np.asarray(pts, dtype=np.float64)
        if p.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return self.scale * (self.R @ p.T).T + self.t[None, :]

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "scale": round(float(self.scale), 6),
            "ate_rmse": round(float(self.ate_rmse), 5),
            "ate_mean": round(float(self.ate_mean), 5),
            "ate_median": round(float(self.ate_median), 5),
        }


@dataclass
class SfmCameraPose:
    """Recovered camera pose and 2D-3D correspondences for a single registered SfM view."""

    image_id: int
    image_name: str
    image_path: Path
    camera_id: int
    intrinsics: CameraIntrinsics
    R_w2c: np.ndarray  # float64 (3, 3) world-to-camera rotation
    t_w2c: np.ndarray  # float64 (3,) world-to-camera translation
    camera_center_w: np.ndarray  # float64 (3,) -R_w2c^T @ t_w2c in SfM world coordinates
    observed_uv: np.ndarray  # float64 (K, 2) 2D keypoint coordinates in SfM image space
    observed_point3d_ids: np.ndarray  # int64 (K,) corresponding 3D point IDs in the sparse cloud
    observed_xyz_w: np.ndarray  # float64 (K, 3) corresponding 3D world coordinates

    def project_world_points(self, xyz_w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D world points into this camera's 2D pixel coordinates and return (uv, depth_z)."""
        pts_c = (self.R_w2c @ np.asarray(xyz_w, dtype=np.float64).T).T + self.t_w2c[None, :]
        z = pts_c[:, 2]
        z_safe = np.where(np.abs(z) > 1e-8, z, 1e-8)
        u = self.intrinsics.fx * (pts_c[:, 0] / z_safe) + self.intrinsics.cx
        v = self.intrinsics.fy * (pts_c[:, 1] / z_safe) + self.intrinsics.cy
        return np.column_stack([u, v]), z


@dataclass
class SfmReconstructionResult:
    """Complete output of N-view incremental SfM via pycolmap."""

    num_input_images: int
    num_registered_images: int
    registration_ratio: float
    shared_camera_model: str
    single_camera_mode: bool
    sfm_max_image_size: int
    shared_intrinsics: Optional[CameraIntrinsics]
    registered_poses: Dict[str, SfmCameraPose]  # keyed by image filename (e.g. '00000-color.jpg')
    unregistered_images: List[str]
    sparse_points_xyz: np.ndarray  # float64 (M, 3)
    sparse_points_rgb: np.ndarray  # uint8 (M, 3)
    sparse_point_errors_px: np.ndarray  # float64 (M,)
    sparse_point_track_lengths: np.ndarray  # int32 (M,)
    mean_reprojection_error_px: float
    mean_track_length: float
    feature_extraction_ms: float
    feature_matching_ms: float
    incremental_mapping_ms: float
    total_runtime_ms: float
    peak_ram_mb: float
    peak_ram_delta_mb: float = 0.0
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def num_sparse_points(self) -> int:
        return int(self.sparse_points_xyz.shape[0])

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "num_input_images": self.num_input_images,
            "num_registered_images": self.num_registered_images,
            "unregistered_images": self.unregistered_images,
            "registration_ratio": round(self.registration_ratio, 4),
            "shared_camera_model": self.shared_camera_model,
            "single_camera_mode": self.single_camera_mode,
            "sfm_max_image_size": self.sfm_max_image_size,
            "shared_intrinsics": self.shared_intrinsics.to_dict() if self.shared_intrinsics else None,
            "num_sparse_points": self.num_sparse_points,
            "mean_reprojection_error_px": round(self.mean_reprojection_error_px, 4),
            "mean_track_length": round(self.mean_track_length, 3),
            "feature_extraction_ms": round(self.feature_extraction_ms, 2),
            "feature_matching_ms": round(self.feature_matching_ms, 2),
            "incremental_mapping_ms": round(self.incremental_mapping_ms, 2),
            "total_runtime_ms": round(self.total_runtime_ms, 2),
            "peak_ram_mb": round(self.peak_ram_mb, 2),
            "peak_ram_above_baseline_mb": round(self.peak_ram_delta_mb, 2),
            "ram_measurement": "process RSS sampled every 20 ms by a background thread during SfM",
        }


def align_umeyama_sim3(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    with_scale: bool = True,
) -> Sim3AlignmentResult:
    """From-scratch Sim(3) Umeyama least-squares similarity alignment (Umeyama, TPAMI 1991).

    Finds scale s > 0, rotation R in SO(3), and translation t in R^3 minimizing:
        sum_{i=1}^N || dst_i - (s * R @ src_i + t) ||_2^2

    Args:
        src_pts: (N, 3) source 3D points.
        dst_pts: (N, 3) destination (reference/ground-truth) 3D points.
        with_scale: If True, solve for similarity scale s; if False, fix s = 1.0 (SE(3) Kabsch).

    Returns:
        Sim3AlignmentResult with (scale, R, t, aligned_points, residuals, ate_rmse, ate_mean, ate_median).
    """
    x = np.asarray(src_pts, dtype=np.float64)
    y = np.asarray(dst_pts, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or x.shape[0] < 3:
        raise GeometryProcessingError(
            f"Sim(3) Umeyama alignment requires matching Nx3 arrays with N >= 3, got {x.shape} and {y.shape}"
        )

    n = float(x.shape[0])
    mu_x = np.mean(x, axis=0)
    mu_y = np.mean(y, axis=0)

    x_centered = x - mu_x
    y_centered = y - mu_y

    var_x = float(np.sum(x_centered ** 2) / n)
    if var_x < 1e-15:
        raise GeometryProcessingError("Degenerate source point set with near-zero variance in Sim(3) alignment.")

    cov_xy = (y_centered.T @ x_centered) / n
    U, D, Vt = np.linalg.svd(cov_xy)

    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    scale = float(np.sum(D * np.diag(S)) / var_x) if with_scale else 1.0
    t = mu_y - scale * (R @ mu_x)

    aligned = scale * (R @ x.T).T + t[None, :]
    residuals = np.linalg.norm(aligned - y, axis=1)
    ate_rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ate_mean = float(np.mean(residuals))
    ate_median = float(np.median(residuals))

    return Sim3AlignmentResult(
        scale=scale,
        R=R,
        t=t,
        aligned_points=aligned,
        residuals=residuals,
        ate_rmse=ate_rmse,
        ate_mean=ate_mean,
        ate_median=ate_median,
    )


def _prepare_sfm_staging_images(
    image_paths: List[Path],
    staging_img_dir: Path,
    sfm_max_image_size: int,
) -> Dict[str, Tuple[Path, float]]:
    """Copy or downscale input images into `staging_img_dir` so max dimension <= `sfm_max_image_size`."""
    staging_img_dir.mkdir(parents=True, exist_ok=True)
    mapping: Dict[str, Tuple[Path, float]] = {}

    for p in image_paths:
        p = Path(p)
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise GeometryProcessingError(f"Failed to read input image for SfM: {p}")
        h, w = bgr.shape[:2]
        max_dim = max(h, w)
        if sfm_max_image_size > 0 and max_dim > sfm_max_image_size:
            scale = float(sfm_max_image_size) / float(max_dim)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0

        dst_path = staging_img_dir / p.name
        cv2.imwrite(str(dst_path), bgr)
        mapping[p.name] = (p, scale)

    return mapping


def _camera_to_holo_intrinsics(cam: Any) -> CameraIntrinsics:
    """Convert a pycolmap.Camera object into HoloGen's CameraIntrinsics."""
    w = int(cam.width)
    h = int(cam.height)
    fx = float(cam.focal_length_x) if hasattr(cam, "focal_length_x") else float(cam.focal_length)
    fy = float(cam.focal_length_y) if hasattr(cam, "focal_length_y") else float(cam.focal_length)
    cx = float(cam.principal_point_x)
    cy = float(cam.principal_point_y)
    fov_x = math.degrees(2.0 * math.atan(w / (2.0 * max(fx, 1e-6))))
    fov_y = math.degrees(2.0 * math.atan(h / (2.0 * max(fy, 1e-6))))
    return CameraIntrinsics(
        width=w,
        height=h,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        source=f"pycolmap_{cam.model.name if hasattr(cam.model, 'name') else str(cam.model)}",
        fov_x_deg=fov_x,
        fov_y_deg=fov_y,
    )


def run_incremental_sfm(
    image_paths: List[Path],
    sfm_max_image_size: int = 1600,
    camera_model: str = "SIMPLE_RADIAL",
    single_camera: bool = True,
    max_num_features: int = 4096,
    num_threads: int = 4,
    work_dir: Optional[Path] = None,
) -> SfmReconstructionResult:
    """Run N-view incremental Structure-from-Motion using `pycolmap` on CPU.

    Enforces `CameraMode.SINGLE` by default so same-camera image sets share a single intrinsics
    calibration across all views, and logs registered-image count vs. total, runtime, and peak RAM.
    """
    import pycolmap

    if len(image_paths) < 2:
        raise GeometryProcessingError(
            f"Multi-view SfM requires at least 2 images, got {len(image_paths)}"
        )

    proc = psutil.Process(os.getpid())
    baseline_ram_bytes = proc.memory_info().rss
    peak_ram_bytes = baseline_ram_bytes
    _stop_sampler = threading.Event()

    def _sample_ram() -> None:
        nonlocal peak_ram_bytes
        curr = proc.memory_info().rss
        if curr > peak_ram_bytes:
            peak_ram_bytes = curr

    def _sampler_loop() -> None:
        # Background sampler: catches peaks inside pycolmap's C++ calls (stage-boundary sampling misses them).
        while not _stop_sampler.is_set():
            _sample_ram()
            _stop_sampler.wait(0.02)

    sampler = threading.Thread(target=_sampler_loop, daemon=True)
    sampler.start()

    t_start = time.perf_counter()
    cleanup_temp = False
    if work_dir is None:
        tmp_root = Path(tempfile.mkdtemp(prefix="hologen_sfm_"))
        cleanup_temp = True
    else:
        tmp_root = Path(work_dir)
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        tmp_root.mkdir(parents=True, exist_ok=True)

    try:
        staging_img_dir = tmp_root / "images"
        db_path = tmp_root / "database.db"
        sparse_out_dir = tmp_root / "sparse"
        sparse_out_dir.mkdir(parents=True, exist_ok=True)

        orig_map = _prepare_sfm_staging_images(image_paths, staging_img_dir, sfm_max_image_size)
        _sample_ram()

        # 1. Feature extraction (single shared camera model for same-camera sets)
        t_feat0 = time.perf_counter()
        sift_opts = pycolmap.SiftExtractionOptions()
        sift_opts.max_image_size = int(sfm_max_image_size)
        sift_opts.max_num_features = int(max_num_features)
        sift_opts.num_threads = int(num_threads)

        cam_mode = (
            pycolmap.CameraMode.SINGLE if single_camera else pycolmap.CameraMode.PER_IMAGE
        )
        pycolmap.extract_features(
            database_path=str(db_path),
            image_path=str(staging_img_dir),
            camera_mode=cam_mode,
            camera_model=camera_model,
            sift_options=sift_opts,
            device=pycolmap.Device.cpu,
        )
        feat_ms = (time.perf_counter() - t_feat0) * 1000.0
        _sample_ram()

        # 2. Exhaustive matching
        t_match0 = time.perf_counter()
        match_sift_opts = pycolmap.SiftMatchingOptions()
        match_sift_opts.num_threads = int(num_threads)
        pycolmap.match_exhaustive(
            database_path=str(db_path),
            sift_options=match_sift_opts,
            device=pycolmap.Device.cpu,
        )
        match_ms = (time.perf_counter() - t_match0) * 1000.0
        _sample_ram()

        # 3. Incremental mapping (bundle adjustment)
        t_map0 = time.perf_counter()
        mapper_opts = pycolmap.IncrementalPipelineOptions()
        mapper_opts.num_threads = int(num_threads)
        reconstructions = pycolmap.incremental_mapping(
            database_path=str(db_path),
            image_path=str(staging_img_dir),
            output_path=str(sparse_out_dir),
            options=mapper_opts,
        )
        map_ms = (time.perf_counter() - t_map0) * 1000.0
        _sample_ram()

        if not reconstructions:
            raise GeometryProcessingError(
                f"pycolmap incremental SfM failed to register a reconstruction from {len(image_paths)} views."
            )

        # Pick the largest connected component reconstruction (most registered images)
        best_rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())

        # Parse shared camera(s)
        shared_intr: Optional[CameraIntrinsics] = None
        cam_intr_by_id: Dict[int, CameraIntrinsics] = {}
        for cam_id, cam_obj in best_rec.cameras.items():
            intr = _camera_to_holo_intrinsics(cam_obj)
            cam_intr_by_id[int(cam_id)] = intr
            if shared_intr is None:
                shared_intr = intr

        # Parse sparse 3D points
        p3d_ids: List[int] = []
        xyz_list: List[np.ndarray] = []
        rgb_list: List[np.ndarray] = []
        err_list: List[float] = []
        trk_list: List[int] = []
        p3d_xyz_lookup: Dict[int, np.ndarray] = {}

        for pid, pt3d in best_rec.points3D.items():
            pid_int = int(pid)
            xyz = np.asarray(pt3d.xyz, dtype=np.float64)
            rgb = np.asarray(pt3d.color, dtype=np.uint8)
            p3d_ids.append(pid_int)
            xyz_list.append(xyz)
            rgb_list.append(rgb)
            err_list.append(float(pt3d.error))
            trk_list.append(int(pt3d.track.length()))
            p3d_xyz_lookup[pid_int] = xyz

        if xyz_list:
            sparse_xyz = np.vstack(xyz_list)
            sparse_rgb = np.vstack(rgb_list)
            sparse_err = np.asarray(err_list, dtype=np.float64)
            sparse_trk = np.asarray(trk_list, dtype=np.int32)
        else:
            sparse_xyz = np.zeros((0, 3), dtype=np.float64)
            sparse_rgb = np.zeros((0, 3), dtype=np.uint8)
            sparse_err = np.zeros((0,), dtype=np.float64)
            sparse_trk = np.zeros((0,), dtype=np.int32)

        # Parse registered image poses and 2D-3D observations
        registered_poses: Dict[str, SfmCameraPose] = {}
        for img_id, img_obj in best_rec.images.items():
            if not img_obj.has_pose:
                continue
            cfw = img_obj.cam_from_world() if callable(img_obj.cam_from_world) else img_obj.cam_from_world
            R_w2c = np.asarray(cfw.rotation.matrix(), dtype=np.float64)
            t_w2c = np.asarray(cfw.translation, dtype=np.float64).ravel()
            center_w = np.asarray(img_obj.projection_center(), dtype=np.float64).ravel()

            obs_uv: List[np.ndarray] = []
            obs_ids: List[int] = []
            obs_xyz: List[np.ndarray] = []
            for p2d in img_obj.points2D:
                if p2d.has_point3D():
                    pid = int(p2d.point3D_id)
                    if pid in p3d_xyz_lookup:
                        obs_uv.append(np.asarray(p2d.xy, dtype=np.float64))
                        obs_ids.append(pid)
                        obs_xyz.append(p3d_xyz_lookup[pid])

            orig_path, _ = orig_map.get(img_obj.name, (staging_img_dir / img_obj.name, 1.0))
            intr_view = cam_intr_by_id.get(int(img_obj.camera_id), shared_intr)
            assert intr_view is not None

            registered_poses[img_obj.name] = SfmCameraPose(
                image_id=int(img_id),
                image_name=img_obj.name,
                image_path=orig_path,
                camera_id=int(img_obj.camera_id),
                intrinsics=intr_view,
                R_w2c=R_w2c,
                t_w2c=t_w2c,
                camera_center_w=center_w,
                observed_uv=np.vstack(obs_uv) if obs_uv else np.zeros((0, 2), dtype=np.float64),
                observed_point3d_ids=np.asarray(obs_ids, dtype=np.int64),
                observed_xyz_w=np.vstack(obs_xyz) if obs_xyz else np.zeros((0, 3), dtype=np.float64),
            )

        all_names = [Path(p).name for p in image_paths]
        unregistered = [name for name in all_names if name not in registered_poses]
        num_in = len(image_paths)
        num_reg = len(registered_poses)
        reg_ratio = float(num_reg) / max(float(num_in), 1.0)

        mean_reproj = float(best_rec.compute_mean_reprojection_error())
        mean_track = float(best_rec.compute_mean_track_length())
        total_ms = (time.perf_counter() - t_start) * 1000.0
        _stop_sampler.set()
        sampler.join(timeout=1.0)
        _sample_ram()
        peak_ram_mb = float(peak_ram_bytes) / (1024.0 * 1024.0)
        peak_ram_delta_mb = float(peak_ram_bytes - baseline_ram_bytes) / (1024.0 * 1024.0)

        logger.info(
            "SfM finished: registered %d/%d views (%.1f%%), %d sparse 3D points, "
            "mean reproj %.3f px, runtime %.2f s, peak RAM %.1f MB",
            num_reg,
            num_in,
            reg_ratio * 100.0,
            len(sparse_xyz),
            mean_reproj,
            total_ms / 1000.0,
            peak_ram_mb,
        )

        return SfmReconstructionResult(
            num_input_images=num_in,
            num_registered_images=num_reg,
            registration_ratio=reg_ratio,
            shared_camera_model=camera_model,
            single_camera_mode=single_camera,
            sfm_max_image_size=sfm_max_image_size,
            shared_intrinsics=shared_intr,
            registered_poses=registered_poses,
            unregistered_images=unregistered,
            sparse_points_xyz=sparse_xyz,
            sparse_points_rgb=sparse_rgb,
            sparse_point_errors_px=sparse_err,
            sparse_point_track_lengths=sparse_trk,
            mean_reprojection_error_px=mean_reproj,
            mean_track_length=mean_track,
            feature_extraction_ms=feat_ms,
            feature_matching_ms=match_ms,
            incremental_mapping_ms=map_ms,
            total_runtime_ms=total_ms,
            peak_ram_mb=peak_ram_mb,
            peak_ram_delta_mb=peak_ram_delta_mb,
            stage_timings_ms={
                "feature_extraction_ms": round(feat_ms, 2),
                "feature_matching_ms": round(match_ms, 2),
                "incremental_mapping_ms": round(map_ms, 2),
                "total_runtime_ms": round(total_ms, 2),
            },
        )
    finally:
        _stop_sampler.set()
        if cleanup_temp and tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)


def evaluate_sfm_poses_against_reference(
    sfm_result: SfmReconstructionResult,
    ref_views_by_filename: Dict[str, Any],
) -> Tuple[Dict[str, Any], Sim3AlignmentResult]:
    """Align recovered SfM camera centers to reference/ground-truth camera centers via Sim(3) Umeyama
    and compute ATE RMSE, rotation error (deg), and pairwise relative translation direction error (deg).
    """
    matched_names = [
        name
        for name in sorted(sfm_result.registered_poses.keys())
        if name in ref_views_by_filename
    ]
    if len(matched_names) < 3:
        raise GeometryProcessingError(
            f"Need at least 3 registered views matching reference poses for Sim(3) alignment, got {len(matched_names)}"
        )

    sfm_centers = np.vstack([sfm_result.registered_poses[n].camera_center_w for n in matched_names])
    ref_centers = np.vstack(
        [
            getattr(ref_views_by_filename[n], "camera_center_world", getattr(ref_views_by_filename[n], "camera_center_w", None))
            for n in matched_names
        ]
    )

    sim3 = align_umeyama_sim3(sfm_centers, ref_centers, with_scale=True)

    # Rotation error after aligning the SfM world coordinate frame via R_align:
    # A vector v_sfm in SfM world maps to v_ref = R_align @ v_sfm in reference world,
    # so R_w2c_aligned = R_w2c_sfm @ R_align^T.
    rot_errors_deg: List[float] = []
    for name in matched_names:
        R_sfm = sfm_result.registered_poses[name].R_w2c
        R_ref = ref_views_by_filename[name].R_w2c
        R_aligned = R_sfm @ sim3.R.T
        cos_ang = np.clip((float(np.trace(R_ref.T @ R_aligned)) - 1.0) * 0.5, -1.0, 1.0)
        rot_errors_deg.append(math.degrees(math.acos(cos_ang)))

    # Rotation error after the closed-form best global rotation (isolates per-camera rotation
    # error from the conditioning of the centre-based Umeyama rotation).
    R_g = best_global_rotation(
        [sfm_result.registered_poses[n].R_w2c for n in matched_names],
        [ref_views_by_filename[n].R_w2c for n in matched_names],
    )
    rot_opt_errors_deg: List[float] = []
    for name in matched_names:
        R_al = sfm_result.registered_poses[name].R_w2c @ R_g.T
        R_ref = ref_views_by_filename[name].R_w2c
        cos_o = np.clip((float(np.trace(R_ref.T @ R_al)) - 1.0) * 0.5, -1.0, 1.0)
        rot_opt_errors_deg.append(math.degrees(math.acos(cos_o)))

    # Adjacent pairwise relative pose errors (invariant to global world frame)
    rel_rot_errors_deg: List[float] = []
    rel_trans_errors_deg: List[float] = []
    for i in range(len(matched_names) - 1):
        n1, n2 = matched_names[i], matched_names[i + 1]
        p1_s, p2_s = sfm_result.registered_poses[n1], sfm_result.registered_poses[n2]
        p1_r, p2_r = ref_views_by_filename[n1], ref_views_by_filename[n2]

        R_rel_s = p2_s.R_w2c @ p1_s.R_w2c.T
        t_rel_s = p2_s.t_w2c - R_rel_s @ p1_s.t_w2c

        R_rel_r = p2_r.R_w2c @ p1_r.R_w2c.T
        t_rel_r = p2_r.t_w2c - R_rel_r @ p1_r.t_w2c

        cos_r = np.clip((float(np.trace(R_rel_r.T @ R_rel_s)) - 1.0) * 0.5, -1.0, 1.0)
        rel_rot_errors_deg.append(math.degrees(math.acos(cos_r)))

        ns = float(np.linalg.norm(t_rel_s))
        nr = float(np.linalg.norm(t_rel_r))
        if ns > 1e-12 and nr > 1e-12:
            cos_t = np.clip(float(np.dot(t_rel_s / ns, t_rel_r / nr)), -1.0, 1.0)
            rel_trans_errors_deg.append(math.degrees(math.acos(cos_t)))

    metrics = {
        "num_aligned_cameras": len(matched_names),
        "umeyama_scale": round(float(sim3.scale), 5),
        "camera_center_ate_rmse": round(float(sim3.ate_rmse), 5),
        "camera_center_ate_mean": round(float(sim3.ate_mean), 5),
        "camera_center_ate_median": round(float(sim3.ate_median), 5),
        "trajectory_collinearity_sv_ratio": round(trajectory_sv_ratio(ref_centers), 5),
        "aligned_rotation_error_mean_deg": round(float(np.mean(rot_errors_deg)), 4),
        "aligned_rotation_error_median_deg": round(float(np.median(rot_errors_deg)), 4),
        "rotation_error_after_best_global_rotation_mean_deg": round(float(np.mean(rot_opt_errors_deg)), 4),
        "pairwise_relative_rotation_error_mean_deg": round(float(np.mean(rel_rot_errors_deg)), 4),
        "pairwise_relative_translation_dir_error_mean_deg": round(float(np.mean(rel_trans_errors_deg)), 4),
    }
    return metrics, sim3


def trajectory_sv_ratio(centers: np.ndarray) -> float:
    """Ratio sigma_2 / sigma_1 of centred camera centres. Near 0 => nearly collinear trajectory,
    in which case the Umeyama rotation about the trajectory axis is poorly constrained."""
    c = np.asarray(centers, dtype=np.float64)
    s = np.linalg.svd(c - c.mean(axis=0), compute_uv=False)
    return float(s[1] / s[0]) if s[0] > 1e-15 else 0.0


def best_global_rotation(R_sfm_list: List[np.ndarray], R_ref_list: List[np.ndarray]) -> np.ndarray:
    """Closed-form R_g in SO(3) minimizing sum_i ||R_ref_i - R_sfm_i R_g^T||_F^2 (chordal L2).

    With N = sum_i R_ref_i^T R_sfm_i = U S V^T, the optimum is R_g^T = V D U^T, D = diag(1, 1, det(V U^T)).
    """
    N = np.zeros((3, 3), dtype=np.float64)
    for Rs, Rr in zip(R_sfm_list, R_ref_list):
        N += Rr.T @ Rs
    U, _, Vt = np.linalg.svd(N)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    X = Vt.T @ D @ U.T  # = R_g^T
    return X.T


def render_sfm_sparse_reconstruction_visual(
    sfm_result: SfmReconstructionResult,
    sim3_alignment: Optional[Sim3AlignmentResult],
    ref_views_by_filename: Optional[Dict[str, Any]],
    scene_title: str,
    pose_semantics_label: str,
    output_png_path: Path,
) -> None:
    """Render a 3-panel engineering summary of the sparse SfM reconstruction:
    - Left: Aligned 3D Sparse Point Cloud + Recovered Camera Frustums vs Reference/GT Trajectory (XZ view)
    - Middle: Camera Center Trajectory Alignment in 3D (XY / XZ projections)
    - Right: Per-View Track Observations & Mean Reprojection Error summary
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_png_path = Path(output_png_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    names = sorted(sfm_result.registered_poses.keys())
    sfm_centers = np.vstack([sfm_result.registered_poses[n].camera_center_w for n in names])
    pts_xyz = sfm_result.sparse_points_xyz.copy()
    pts_rgb = sfm_result.sparse_points_rgb.astype(np.float32) / 255.0

    if sim3_alignment is not None:
        sfm_centers_plot = sim3_alignment.transform_points(sfm_centers)
        pts_xyz_plot = sim3_alignment.transform_points(pts_xyz)
    else:
        sfm_centers_plot = sfm_centers
        pts_xyz_plot = pts_xyz

    # Filter extreme distant sparse outlier points (>97th percentile distance from camera centroid) for clear plotting
    if len(pts_xyz_plot) > 10:
        cam_mean = np.mean(sfm_centers_plot, axis=0)
        dists = np.linalg.norm(pts_xyz_plot - cam_mean[None, :], axis=1)
        keep = dists <= np.percentile(dists, 96.0)
        pts_xyz_plot = pts_xyz_plot[keep]
        pts_rgb = pts_rgb[keep]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=140)
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#475569")

    # Panel 1: Sparse Point Cloud + Camera Frustums (X-Z Top-Down View)
    ax0 = axes[0]
    if len(pts_xyz_plot) > 0:
        ax0.scatter(
            pts_xyz_plot[:, 0],
            pts_xyz_plot[:, 2],
            c=pts_rgb,
            s=3.5,
            alpha=0.75,
            label=f"Sparse 3D Cloud ({sfm_result.num_sparse_points} pts)",
        )
    ax0.plot(
        sfm_centers_plot[:, 0],
        sfm_centers_plot[:, 2],
        "-o",
        color="#38bdf8",
        markersize=5,
        linewidth=1.6,
        label=f"SfM Cameras ({sfm_result.num_registered_images}/{sfm_result.num_input_images})",
    )
    # Draw camera viewing direction arrows
    scale_arrow = 0.08 * max(float(np.ptp(sfm_centers_plot, axis=0).max()), 0.5)
    for idx, n in enumerate(names):
        pose = sfm_result.registered_poses[n]
        look_sfm = pose.R_w2c.T @ np.array([0.0, 0.0, 1.0])
        look_w = (sim3_alignment.R @ look_sfm) if sim3_alignment is not None else look_sfm
        c = sfm_centers_plot[idx]
        ax0.arrow(
            c[0],
            c[2],
            look_w[0] * scale_arrow,
            look_w[2] * scale_arrow,
            head_width=scale_arrow * 0.28,
            color="#facc15",
            alpha=0.9,
        )

    ax0.set_title("Sparse 3D Cloud & Camera Frustums (X-Z)", color="#f8fafc", fontsize=10, fontweight="bold")
    ax0.set_xlabel("X (aligned m)", color="#cbd5e1", fontsize=9)
    ax0.set_ylabel("Z (aligned m)", color="#cbd5e1", fontsize=9)
    ax0.legend(loc="upper right", fontsize=7.5, facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc")
    ax0.grid(True, color="#334155", linestyle="--", alpha=0.5)

    # Panel 2: Sim(3) Aligned Camera Trajectory vs Reference / Ground Truth
    ax1 = axes[1]
    ax1.plot(
        sfm_centers_plot[:, 0],
        sfm_centers_plot[:, 1],
        "-o",
        color="#38bdf8",
        linewidth=1.8,
        markersize=5.5,
        label="Recovered SfM Trajectory (Sim3)",
    )
    if ref_views_by_filename is not None:
        ref_pts = np.vstack(
            [
                getattr(ref_views_by_filename[n], "camera_center_world", getattr(ref_views_by_filename[n], "camera_center_w", None))
                for n in names
                if n in ref_views_by_filename
            ]
        )
        ax1.plot(
            ref_pts[:, 0],
            ref_pts[:, 1],
            "--s",
            color="#f43f5e",
            linewidth=1.5,
            markersize=5,
            label=pose_semantics_label,
        )
    ax1.set_title("Camera Trajectory Alignment (X-Y)", color="#f8fafc", fontsize=10, fontweight="bold")
    ax1.set_xlabel("X (aligned m)", color="#cbd5e1", fontsize=9)
    ax1.set_ylabel("Y (aligned m)", color="#cbd5e1", fontsize=9)
    ax1.legend(loc="best", fontsize=7.5, facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc")
    ax1.grid(True, color="#334155", linestyle="--", alpha=0.5)

    # Panel 3: Per-view 3D Track Observations
    ax2 = axes[2]
    obs_counts = [len(sfm_result.registered_poses[n].observed_point3d_ids) for n in names]
    short_labels = [n[:10] for n in names]
    bars = ax2.bar(range(len(names)), obs_counts, color="#34d399", edgecolor="#059669", alpha=0.88)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax2.set_title(
        f"3D Points Observed per View (Reproj: {sfm_result.mean_reprojection_error_px:.2f} px)",
        color="#f8fafc",
        fontsize=10,
        fontweight="bold",
    )
    ax2.set_ylabel("Triangulated 3D Observations", color="#cbd5e1", fontsize=9)
    ax2.grid(True, axis="y", color="#334155", linestyle="--", alpha=0.5)
    for b, cnt in zip(bars, obs_counts):
        ax2.text(
            b.get_x() + b.get_width() * 0.5,
            b.get_height() + max(obs_counts) * 0.02,
            str(cnt),
            ha="center",
            va="bottom",
            color="#e2e8f0",
            fontsize=7,
        )

    fig.suptitle(
        f"{scene_title} | Registered {sfm_result.num_registered_images}/{sfm_result.num_input_images} Views "
        f"| Runtime {sfm_result.total_runtime_ms / 1000.0:.2f}s | Peak RAM {sfm_result.peak_ram_mb:.0f} MB",
        color="#f8fafc",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(str(output_png_path), facecolor=fig.get_facecolor())
    plt.close(fig)
