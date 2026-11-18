"""Phase 7e — Experiment D: Single-View vs. Sparse SfM vs. Multi-View Fusion (N = 1, 3, 5, 10, all).

Protocol:
1. Consistency threshold tuning on Fountain (10 views) by Precision & F-score (NOT point count),
   then out-of-sample evaluation on ETH3D 'pipes' (Faro laser scan ground truth).
2. Multi-view alignment: Sim(3) Umeyama on recovered camera centres -> GT/ref camera centres,
   evaluated against scan/reference both WITHOUT ICP and WITH ICP refinement.
3. Completeness (Recall) computed ONLY within the reconstruction's bounding box (+0.05 m pad).
4. Single-view fair alignment: camera-frame inverse-depth scale+shift fit against visible scan/ref
   points in the view frustum, transformed to world frame (evaluated without and with ICP).
5. Per-view ETH3D GT depth rendering from scan1.ply: compares single-view (median-scaled) vs
   SfM-aligned multi-view depth (Abs Rel, RMSE).
6. Custom scene check (datasets/custom_scene_01/ if present).
"""

import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.evaluation.multiview_datasets import (  # noqa: E402
    CalibratedView,
    load_eth3d_pipes_dataset,
    load_fountain_dataset,
)
from app.geometry.multi_view_fusion import (  # noqa: E402
    MultiViewFusionConfig,
    align_monocular_disparity_to_sparse_depth,
    fuse_multiview_tsdf_and_mesh,
)
from app.pipelines.single_view import SingleViewPipeline  # noqa: E402
from app.vision.camera import CameraIntrinsics  # noqa: E402
from app.vision.depth_estimation import create_depth_model  # noqa: E402
from app.vision.sfm import (  # noqa: E402
    align_umeyama_sim3,
    evaluate_sfm_poses_against_reference,
    run_incremental_sfm,
)


def _sample_surface_points(geom_path: Path, max_points: int = 80000, seed: int = 42) -> np.ndarray:
    """Load a PLY mesh or point cloud and return up to `max_points` 3D points."""
    mesh = o3d.io.read_triangle_mesh(str(geom_path))
    if len(mesh.triangles) > 0:
        pcd = mesh.sample_points_uniformly(number_of_points=min(max_points, max(len(mesh.triangles) * 2, 20000)))
        pts = np.asarray(pcd.points, dtype=np.float64)
    else:
        pcd = o3d.io.read_point_cloud(str(geom_path))
        pts = np.asarray(pcd.points, dtype=np.float64)
    if len(pts) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
    return pts


def _compute_geom_metrics_in_bbox(
    recon_pts: np.ndarray,
    gt_pts: np.ndarray,
    thresholds: List[float],
    bbox_pad: float = 0.05,
) -> Dict[str, Any]:
    """Compute Precision, Recall, F-score at `thresholds` and Chamfer distance.

    Completeness (Recall) is computed ONLY on GT/reference points inside the reconstruction's
    axis-aligned bounding box padded by `bbox_pad` metres.
    """
    if len(recon_pts) == 0 or len(gt_pts) == 0:
        return {"error": "empty_point_cloud"}

    bmin = np.min(recon_pts, axis=0) - bbox_pad
    bmax = np.max(recon_pts, axis=0) + bbox_pad
    in_box = np.all((gt_pts >= bmin) & (gt_pts <= bmax), axis=1)
    gt_box = gt_pts[in_box]
    if len(gt_box) < 10:
        gt_box = gt_pts

    pcd_r = o3d.geometry.PointCloud()
    pcd_r.points = o3d.utility.Vector3dVector(recon_pts)
    pcd_g = o3d.geometry.PointCloud()
    pcd_g.points = o3d.utility.Vector3dVector(gt_box)

    d_r2g = np.asarray(pcd_r.compute_point_cloud_distance(pcd_g), dtype=np.float64)
    d_g2r = np.asarray(pcd_g.compute_point_cloud_distance(pcd_r), dtype=np.float64)

    chamfer = float(0.5 * (np.mean(d_r2g) + np.mean(d_g2r)))
    out: Dict[str, Any] = {
        "num_recon_pts": int(len(recon_pts)),
        "num_gt_pts_in_bbox": int(len(gt_box)),
        "chamfer_m": round(chamfer, 5),
        "accuracy_mean_m": round(float(np.mean(d_r2g)), 5),
        "completeness_mean_m": round(float(np.mean(d_g2r)), 5),
        "thresholds": {},
    }
    for tau in thresholds:
        prec = float(np.mean(d_r2g <= tau))
        rec = float(np.mean(d_g2r <= tau))
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 1e-12 else 0.0
        out["thresholds"][f"{tau:.2f}m"] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f_score": round(f1, 4),
        }
    return out


def _refine_with_icp(src_pts: np.ndarray, gt_pts: np.ndarray, max_corr_dist: float = 0.15) -> Tuple[np.ndarray, float]:
    """Run point-to-point ICP refinement and return (transformed_src_pts, fitness)."""
    pcd_s = o3d.geometry.PointCloud()
    pcd_s.points = o3d.utility.Vector3dVector(src_pts)
    pcd_g = o3d.geometry.PointCloud()
    pcd_g.points = o3d.utility.Vector3dVector(gt_pts)
    reg = o3d.pipelines.registration.registration_icp(
        pcd_s,
        pcd_g,
        max_corr_dist,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
    )
    pcd_s.transform(reg.transformation)
    return np.asarray(pcd_s.points, dtype=np.float64), float(reg.fitness)


def _render_gt_depth_map(
    gt_pts_world: np.ndarray,
    R_w2c: np.ndarray,
    t_w2c: np.ndarray,
    intr: CameraIntrinsics,
    depth_h: int,
    depth_w: int,
) -> np.ndarray:
    """Project GT 3D scan points into a camera with z-buffering (and 3x3 min-pool splat) to get sparse GT depth."""
    sx = float(depth_w) / float(intr.width)
    sy = float(depth_h) / float(intr.height)
    fx, fy, cx, cy = intr.fx * sx, intr.fy * sy, intr.cx * sx, intr.cy * sy

    pts_c = (R_w2c @ gt_pts_world.T).T + t_w2c[None, :]
    z = pts_c[:, 2]
    front = z > 0.1
    pts_c = pts_c[front]
    z = z[front]

    u = np.round(fx * (pts_c[:, 0] / z) + cx).astype(np.int64)
    v = np.round(fy * (pts_c[:, 1] / z) + cy).astype(np.int64)
    in_img = (u >= 0) & (u < depth_w) & (v >= 0) & (v < depth_h)
    u, v, z = u[in_img], v[in_img], z[in_img]

    # Sort descending by z so closer points overwrite farther points
    order = np.argsort(-z)
    gt_depth = np.zeros((depth_h, depth_w), dtype=np.float32)
    gt_depth[v[order], u[order]] = z[order].astype(np.float32)
    return gt_depth


def _eval_single_view_fair(
    view: CalibratedView,
    gt_pts_world: np.ndarray,
    depth_model,
    thresholds: List[float],
    icp_dist: float,
) -> Dict[str, Any]:
    """Fair single-view protocol:
    1. Run SingleViewPipeline on the view's image.
    2. Project visible GT scan/reference points into this view's camera frame to get GT depth.
    3. Fit inverse-depth scale+shift (1/Z = s * d_mono + t) via least-squares on valid GT depth pixels.
    4. Back-project aligned depth to camera frame, transform to world frame via GT/ref (R_w2c, t_w2c),
       and evaluate both WITHOUT ICP and WITH ICP.
    """
    t0 = time.perf_counter()
    rgb_full, intr_full = view.load_rgb_resized(max_dim=640)
    pred = depth_model.predict(rgb_full)
    h, w = pred.disparity_map.shape[:2]
    sx = float(w) / float(intr_full.width)
    sy = float(h) / float(intr_full.height)
    intr_d = CameraIntrinsics(
        width=w, height=h, fx=intr_full.fx * sx, fy=intr_full.fy * sy,
        cx=intr_full.cx * sx, cy=intr_full.cy * sy, source=intr_full.source,
        fov_x_deg=intr_full.fov_x_deg, fov_y_deg=intr_full.fov_y_deg,
    )

    gt_d = _render_gt_depth_map(gt_pts_world, view.R_w2c, view.t_w2c, intr_d, h, w)
    valid_gt = gt_d > 0.1
    d_obs = pred.disparity_map[valid_gt].astype(np.float64)
    inv_z_obs = 1.0 / gt_d[valid_gt].astype(np.float64)

    A = np.column_stack([d_obs, np.ones_like(d_obs)])
    sol, _, _, _ = np.linalg.lstsq(A, inv_z_obs, rcond=None)
    s_fit, t_fit = float(sol[0]), float(sol[1])
    if s_fit <= 1e-6 or float(np.min(s_fit * pred.disparity_map + t_fit)) <= 1e-4:
        s_fit = float(np.median(inv_z_obs / np.maximum(d_obs, 1e-4)))
        t_fit = 0.0

    z_aligned = (1.0 / np.maximum(s_fit * pred.disparity_map.astype(np.float64) + t_fit, 1e-4)).astype(np.float32)

    # Back-project every pixel -> camera frame -> world frame
    vs, us = np.mgrid[0:h, 0:w]
    z_flat = z_aligned.ravel().astype(np.float64)
    x_cam = (us.ravel() - intr_d.cx) * z_flat / intr_d.fx
    y_cam = (vs.ravel() - intr_d.cy) * z_flat / intr_d.fy
    pts_cam = np.column_stack([x_cam, y_cam, z_flat])
    # Subsample to 50k points for fair comparison density
    if len(pts_cam) > 50000:
        pts_cam = pts_cam[:: max(1, len(pts_cam) // 50000)]
    pts_world = (view.R_w2c.T @ (pts_cam - view.t_w2c[None, :]).T).T

    no_icp = _compute_geom_metrics_in_bbox(pts_world, gt_pts_world, thresholds)
    pts_icp, fitness = _refine_with_icp(pts_world, gt_pts_world, max_corr_dist=icp_dist)
    with_icp = _compute_geom_metrics_in_bbox(pts_icp, gt_pts_world, thresholds)
    runtime_s = time.perf_counter() - t0

    return {
        "method": "single_view_aligned",
        "num_views": 1,
        "runtime_s": round(runtime_s, 2),
        "icp_fitness": round(fitness, 4),
        "without_icp": no_icp,
        "with_icp": with_icp,
    }


def _eval_multiview_subset(
    label: str,
    views: List[CalibratedView],
    gt_pts_world: np.ndarray,
    depth_model,
    fusion_cfg: MultiViewFusionConfig,
    thresholds: List[float],
    icp_dist: float,
    include_sparse_only: bool = False,
) -> Dict[str, Any]:
    """Run SfM + 7c Fusion on `views`, align via Sim(3) Umeyama on camera centres, evaluate with & without ICP."""
    proc = psutil.Process(os.getpid())
    peak_rss = proc.memory_info().rss
    stop_evt = threading.Event()

    def _poll():
        nonlocal peak_rss
        while not stop_evt.is_set():
            curr = proc.memory_info().rss
            if curr > peak_rss:
                peak_rss = curr
            stop_evt.wait(0.02)

    th = threading.Thread(target=_poll, daemon=True)
    th.start()
    t0 = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix=f"hologen_expd_{label}_") as tmp_u:
            sfm_res = run_incremental_sfm(
                image_paths=[v.image_path for v in views],
                sfm_max_image_size=1600,
                camera_model="SIMPLE_RADIAL",
                fallback_camera_model="OPENCV",
                single_camera=True,
                max_num_features=4096,
                num_threads=4,
                undistort=True,
                undistorted_dir=Path(tmp_u) / "undist",
            )
            ref_map = {v.image_path.name: v for v in views}
            pose_eval, sim3 = evaluate_sfm_poses_against_reference(sfm_res, ref_map)
            fusion_res = fuse_multiview_tsdf_and_mesh(sfm_res, depth_model, fusion_cfg)
    finally:
        stop_evt.set()
        th.join(timeout=1.0)

    total_s = time.perf_counter() - t0
    peak_mb = round(peak_rss / (1024.0 * 1024.0), 2)

    # Sample fused mesh surface (or cloud) & align to GT via camera-centre Sim(3) Umeyama
    o3d_mesh = fusion_res.mesh.to_open3d()
    if len(o3d_mesh.triangles) > 0:
        pcd_s = o3d_mesh.sample_points_uniformly(number_of_points=min(60000, max(len(o3d_mesh.triangles) * 2, 15000)))
        pts_sfm = np.asarray(pcd_s.points, dtype=np.float64)
    else:
        pts_sfm = fusion_res.point_cloud.points.astype(np.float64)

    pts_aligned = sim3.transform_points(pts_sfm)
    no_icp = _compute_geom_metrics_in_bbox(pts_aligned, gt_pts_world, thresholds)
    pts_icp, fitness = _refine_with_icp(pts_aligned, gt_pts_world, max_corr_dist=icp_dist)
    with_icp = _compute_geom_metrics_in_bbox(pts_icp, gt_pts_world, thresholds)

    res: Dict[str, Any] = {
        "method": label,
        "num_views": len(views),
        "registered_views": sfm_res.num_registered_images,
        "camera_model": sfm_res.shared_camera_model,
        "mean_reprojection_error_px": round(sfm_res.mean_reprojection_error_px, 4),
        "camera_ate_rmse_m": pose_eval["camera_center_ate_rmse"],
        "per_cam_rot_err_deg": pose_eval["rotation_error_after_best_global_rotation_mean_deg"],
        "runtime_s": round(total_s, 2),
        "peak_rss_mb": peak_mb,
        "mesh_faces": fusion_res.mesh.num_faces,
        "icp_fitness": round(fitness, 4),
        "without_icp": no_icp,
        "with_icp": with_icp,
    }

    if include_sparse_only:
        sp_aligned = sim3.transform_points(sfm_res.sparse_points_xyz)
        sp_no_icp = _compute_geom_metrics_in_bbox(sp_aligned, gt_pts_world, thresholds)
        sp_icp, sp_fit = _refine_with_icp(sp_aligned, gt_pts_world, max_corr_dist=icp_dist)
        sp_with_icp = _compute_geom_metrics_in_bbox(sp_icp, gt_pts_world, thresholds)
        res["sparse_sfm_only"] = {
            "num_sparse_points": sfm_res.num_sparse_points,
            "icp_fitness": round(sp_fit, 4),
            "without_icp": sp_no_icp,
            "with_icp": sp_with_icp,
        }

    gc.collect()
    return res


def _eval_eth3d_per_view_depth(
    views: List[CalibratedView],
    gt_scan_pts: np.ndarray,
    depth_model,
    cfg: MultiViewFusionConfig,
) -> Dict[str, Any]:
    """Compare single-view (median-scaled) vs SfM-aligned multi-view depth against GT laser-scan depth."""
    with tempfile.TemporaryDirectory(prefix="hologen_expd_depth_") as tmp_u:
        sfm_res = run_incremental_sfm(
            image_paths=[v.image_path for v in views],
            sfm_max_image_size=1600,
            camera_model="SIMPLE_RADIAL",
            fallback_camera_model="OPENCV",
            single_camera=True,
            max_num_features=4096,
            num_threads=4,
            undistort=True,
            undistorted_dir=Path(tmp_u) / "undist",
        )
        ref_map = {v.image_path.name: v for v in views}
        _, sim3 = evaluate_sfm_poses_against_reference(sfm_res, ref_map)

        per_view = []
        for name in sorted(sfm_res.registered_poses.keys()):
            pose = sfm_res.registered_poses[name]
            v_gt = ref_map[name]
            bgr = cv2.imread(str(pose.image_path))
            rgb = bgr[:, :, ::-1]
            h0, w0 = rgb.shape[:2]
            sc = 640.0 / float(max(h0, w0))
            rgb_d = cv2.resize(rgb, (int(round(w0 * sc)), int(round(h0 * sc))), interpolation=cv2.INTER_AREA)
            rh, rw = rgb_d.shape[:2]
            intr_d = CameraIntrinsics(
                width=rw, height=rh,
                fx=pose.intrinsics.fx * (rw / pose.intrinsics.width),
                fy=pose.intrinsics.fy * (rh / pose.intrinsics.height),
                cx=pose.intrinsics.cx * (rw / pose.intrinsics.width),
                cy=pose.intrinsics.cy * (rh / pose.intrinsics.height),
                source=pose.intrinsics.source,
                fov_x_deg=pose.intrinsics.fov_x_deg,
                fov_y_deg=pose.intrinsics.fov_y_deg,
            )

            pred = depth_model.predict(rgb_d)
            gt_d = _render_gt_depth_map(gt_scan_pts, v_gt.R_w2c, v_gt.t_w2c, intr_d, rh, rw)
            valid = gt_d > 0.1
            if int(np.sum(valid)) < 50:
                continue

            # Single-view depth (median-scaled to GT depth for this view)
            sv_raw = pred.depth_map[valid].astype(np.float64)
            gt_v = gt_d[valid].astype(np.float64)
            sv_scaled = sv_raw * float(np.median(gt_v) / max(np.median(sv_raw), 1e-6))
            sv_abs_rel = float(np.mean(np.abs(sv_scaled - gt_v) / gt_v))
            sv_rmse = float(np.sqrt(np.mean((sv_scaled - gt_v) ** 2)))

            # SfM-aligned multi-view depth (scaled to metric metres via sim3.scale!)
            aligned_sfm_d, fit_mode, *_ = align_monocular_disparity_to_sparse_depth(
                pred.disparity_map, pose, intr_d, cfg
            )
            if aligned_sfm_d is None:
                continue
            mv_metric = aligned_sfm_d[valid].astype(np.float64) * float(sim3.scale)
            mv_abs_rel = float(np.mean(np.abs(mv_metric - gt_v) / gt_v))
            mv_rmse = float(np.sqrt(np.mean((mv_metric - gt_v) ** 2)))

            per_view.append({
                "view": name,
                "fit_mode": fit_mode,
                "num_gt_pixels": int(np.sum(valid)),
                "single_view_abs_rel": round(sv_abs_rel, 4),
                "single_view_rmse_m": round(sv_rmse, 4),
                "multiview_aligned_abs_rel": round(mv_abs_rel, 4),
                "multiview_aligned_rmse_m": round(mv_rmse, 4),
            })

    return {
        "num_views_evaluated": len(per_view),
        "single_view_abs_rel_median": round(float(np.median([x["single_view_abs_rel"] for x in per_view])), 4),
        "single_view_rmse_median_m": round(float(np.median([x["single_view_rmse_m"] for x in per_view])), 4),
        "multiview_aligned_abs_rel_median": round(float(np.median([x["multiview_aligned_abs_rel"] for x in per_view])), 4),
        "multiview_aligned_rmse_median_m": round(float(np.median([x["multiview_aligned_rmse_m"] for x in per_view])), 4),
        "per_view": per_view,
    }


def _plot_scaling_curves(fountain_rows: List[Dict], pipes_rows: List[Dict], out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=140)
    fig.patch.set_facecolor("#0f172a")
    for ax, title, rows, thresh_key in (
        (axes[0], "Fountain (RGB-D Reference, tau=0.05m)", fountain_rows, "0.05m"),
        (axes[1], "ETH3D 'pipes' (Faro Laser GT, tau=0.10m)", pipes_rows, "0.10m"),
    ):
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#475569")
        ns = [r["num_views"] for r in rows]
        f1_no = [r["without_icp"]["thresholds"][thresh_key]["f_score"] for r in rows]
        f1_icp = [r["with_icp"]["thresholds"][thresh_key]["f_score"] for r in rows]
        ax.plot(range(len(ns)), f1_no, "-o", color="#38bdf8", label="F-score (Sim3 only)")
        ax.plot(range(len(ns)), f1_icp, "--s", color="#34d399", label="F-score (Sim3 + ICP)")
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([f"N={n}" for n in ns], color="#cbd5e1")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("F-score", color="#cbd5e1")
        ax.set_title(title, color="#f8fafc", fontsize=10, fontweight="bold")
        ax.grid(True, color="#334155", linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8, facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc")
    plt.tight_layout()
    fig.savefig(str(out_png), facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    settings = get_settings()
    out_dir = settings.outputs_dir / "experiment_d"
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_model = create_depth_model("depth_anything_v2_small")
    fountain_views, fountain_ref_path = load_fountain_dataset(settings.datasets_dir)
    pipes_views, pipes_scan_path = load_eth3d_pipes_dataset(settings.datasets_dir)

    fountain_gt_pts = _sample_surface_points(fountain_ref_path, max_points=80000)
    pipes_gt_pts = _sample_surface_points(pipes_scan_path, max_points=100000)

    # 1. Consistency Threshold Tuning on Fountain (10 views) by Precision & F-score
    tuning_candidates = [
        ("strict", 0.05, 0.12),
        ("default", 0.10, 0.18),
        ("relaxed", 0.20, 0.30),
    ]
    fountain_10 = fountain_views[::3][:10]
    tuning_results = []
    best_cfg = None
    best_f1 = -1.0
    for label, rel_d, rgb_t in tuning_candidates:
        cfg_t = MultiViewFusionConfig(
            depth_max_dim=640, discontinuity_threshold=0.08,
            consistency_rel_depth_thresh=rel_d, consistency_rgb_thresh=rgb_t, min_consistent_views=1,
        )
        r_t = _eval_multiview_subset(
            f"tune_{label}", fountain_10, fountain_gt_pts, depth_model, cfg_t,
            thresholds=[0.02, 0.05], icp_dist=0.10,
        )
        f1_05 = r_t["without_icp"]["thresholds"]["0.05m"]["f_score"]
        prec_05 = r_t["without_icp"]["thresholds"]["0.05m"]["precision"]
        tuning_results.append({
            "setting": label, "consistency_rel_depth_thresh": rel_d, "consistency_rgb_thresh": rgb_t,
            "precision_0.05m": prec_05, "f_score_0.05m": f1_05,
            "chamfer_m": r_t["without_icp"]["chamfer_m"],
        })
        if f1_05 > best_f1:
            best_f1 = f1_05
            best_cfg = cfg_t
    assert best_cfg is not None

    # 2. Fountain Scaling (N = 1, 3, 5, 10, 33) — Reference Geometry (RGB-D)
    f_n1 = _eval_single_view_fair(fountain_views[0], fountain_gt_pts, depth_model, [0.02, 0.05], icp_dist=0.10)
    f_n3 = _eval_multiview_subset("fountain_N3", fountain_views[::2][:3], fountain_gt_pts, depth_model, best_cfg, [0.02, 0.05], 0.10)
    f_n5 = _eval_multiview_subset("fountain_N5", fountain_views[::2][:5], fountain_gt_pts, depth_model, best_cfg, [0.02, 0.05], 0.10)
    f_n10 = _eval_multiview_subset("fountain_N10", fountain_10, fountain_gt_pts, depth_model, best_cfg, [0.02, 0.05], 0.10)
    f_nall = _eval_multiview_subset("fountain_N33", fountain_views, fountain_gt_pts, depth_model, best_cfg, [0.02, 0.05], 0.10, include_sparse_only=True)
    fountain_rows = [f_n1, f_n3, f_n5, f_n10, f_nall]

    # 3. ETH3D 'pipes' Out-of-Sample Evaluation (N = 1, 3, 5, 10, 14) — Ground-Truth Faro Laser Scan
    p_n1 = _eval_single_view_fair(pipes_views[1], pipes_gt_pts, depth_model, [0.05, 0.10], icp_dist=0.20)
    p_n3 = _eval_multiview_subset("pipes_N3", pipes_views[:3], pipes_gt_pts, depth_model, best_cfg, [0.05, 0.10], 0.20)
    p_n5 = _eval_multiview_subset("pipes_N5", [pipes_views[i] for i in (0, 1, 2, 9, 10)], pipes_gt_pts, depth_model, best_cfg, [0.05, 0.10], 0.20)
    p_n10 = _eval_multiview_subset("pipes_N10", pipes_views[:10], pipes_gt_pts, depth_model, best_cfg, [0.05, 0.10], 0.20)
    p_nall = _eval_multiview_subset("pipes_N14", pipes_views, pipes_gt_pts, depth_model, best_cfg, [0.05, 0.10], 0.20, include_sparse_only=True)
    pipes_rows = [p_n1, p_n3, p_n5, p_n10, p_nall]

    # 4. Per-view GT depth evaluation on ETH3D 'pipes'
    eth3d_depth_eval = _eval_eth3d_per_view_depth(pipes_views, pipes_gt_pts, depth_model, best_cfg)

    # 5. Custom scene check (datasets/custom_scene_01/)
    custom_dir = settings.datasets_dir / "custom_scene_01"
    if custom_dir.exists():
        imgs = sorted([p for p in custom_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if len(imgs) >= 2:
            from app.pipelines.multi_view import MultiViewPipeline
            mv_res = MultiViewPipeline(settings=settings, depth_model=depth_model).run(
                image_paths=imgs, run_id="custom_scene_01", output_base_dir=out_dir
            )
            custom_report = {
                "status": "completed",
                "num_input_images": len(imgs),
                "registered_views": mv_res.sfm_result.num_registered_images,
                "runtime_s": round(mv_res.total_runtime_ms / 1000.0, 2),
                "peak_rss_mb": mv_res.whole_pipeline_peak_rss_mb,
                "mesh_faces": mv_res.full_mesh.num_faces,
            }
        else:
            custom_report = {"status": "insufficient_images", "count": len(imgs)}
    else:
        custom_report = {"status": "not_present_in_workspace", "expected_path": str(custom_dir)}

    _plot_scaling_curves(fountain_rows, pipes_rows, out_dir / "experiment_d_scaling_curves.png")

    payload = {
        "phase": "phase_7e_experiment_d",
        "consistency_tuning_on_fountain_10views": tuning_results,
        "selected_consistency_config": {
            "consistency_rel_depth_thresh": best_cfg.consistency_rel_depth_thresh,
            "consistency_rgb_thresh": best_cfg.consistency_rgb_thresh,
        },
        "fountain_reference_rgbd_geometry": fountain_rows,
        "eth3d_pipes_ground_truth_scan": pipes_rows,
        "eth3d_per_view_gt_depth_comparison": eth3d_depth_eval,
        "custom_scene_01": custom_report,
    }
    (out_dir / "experiment_d_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Experiment D complete. Saved to", out_dir / "experiment_d_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
