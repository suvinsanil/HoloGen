"""Phase 7e.2 — Experiment D v2 (Fixed Evaluation Protocol).

Key Protocol Fixes vs Experiment D v1:
1. Equalized Alignment Budget (NO ICP in headline):
   - Multi-view (N=3, 5, 10, all & sparse SfM): Sim(3) Umeyama on recovered camera centres vs GT/ref centres.
   - Single-view (N=1): GT/ref camera pose + 1 global scale (median Z_gt / Z_mono), documented as oracle-scale baseline.
   - Supplementary rigid ICP (SE(3), 40 iters, max_corr=0.10m Fountain / 0.20m ETH3D) reported separately.
2. Equalized Evaluation Region:
   - (a) GT Bounding-Box Cropped: every reconstruction cropped to GT/ref scan AABB (+0.10m Fountain, +0.15m ETH3D) before Precision & Recall.
   - (b) Frustum-Restricted: both GT/ref scan and every reconstruction cropped to the N=1 camera's viewing frustum.
3. Greedy Covisibility Subsets from Single Full-Scene SfM:
   - Run SfM once on all views, then greedily select N=3, 5, 10, all subsets by shared 3D point covisibility
     starting from the N=1 anchor view (so registration failures never alter N).
4. Improved Fusion:
   - min_consistent_views=2 (when N>=3), sparse-depth cap (1.35x p98), drop low-support TSDF vertices (<2 views).
   - Consistency thresholds tuned on Fountain (10 views) by Precision & F-score, evaluated out-of-sample on ETH3D.
5. Metrics:
   - Headline F-score at multiple thresholds + Precision/Recall curves + Truncated Chamfer (capped at tau);
     Raw Chamfer reported only as a supplementary column.
6. Sparse-Cloud Investigation (ETH3D):
   - Evaluates raw vs filtered (track_length >= 3, reproj_error <= 1.0 px) sparse cloud in GT AABB.
7. Per-View GT Depth on ETH3D (14 views):
   - Single-view median-scale vs Single-view oracle scale+shift vs Multi-view SfM-aligned (no GT depth):
     Abs Rel, RMSE, delta < 1.25 (Median and IQR across views).
"""

import copy
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
from app.vision.camera import CameraIntrinsics  # noqa: E402
from app.vision.depth_estimation import create_depth_model  # noqa: E402
from app.vision.sfm import (  # noqa: E402
    SfmReconstructionResult,
    evaluate_sfm_poses_against_reference,
    run_incremental_sfm,
)


def _sample_surface_points(geom_path: Path, max_points: int = 100000, seed: int = 42) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(geom_path))
    if len(mesh.triangles) > 0:
        pcd = mesh.sample_points_uniformly(number_of_points=min(max_points, max(len(mesh.triangles) * 2, 30000)))
        pts = np.asarray(pcd.points, dtype=np.float64)
    else:
        pcd = o3d.io.read_point_cloud(str(geom_path))
        pts = np.asarray(pcd.points, dtype=np.float64)
    if len(pts) > max_points:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), size=max_points, replace=False)]
    return pts


def _crop_to_aabb(pts: np.ndarray, ref_pts: np.ndarray, margin: float) -> np.ndarray:
    if len(pts) == 0 or len(ref_pts) == 0:
        return pts
    bmin = np.min(ref_pts, axis=0) - margin
    bmax = np.max(ref_pts, axis=0) + margin
    mask = np.all((pts >= bmin) & (pts <= bmax), axis=1)
    return pts[mask]


def _filter_to_camera_frustum(
    pts_world: np.ndarray,
    R_w2c: np.ndarray,
    t_w2c: np.ndarray,
    intr: CameraIntrinsics,
    max_depth: Optional[float] = None,
) -> np.ndarray:
    if len(pts_world) == 0:
        return pts_world
    pts_c = (R_w2c @ pts_world.T).T + t_w2c[None, :]
    z = pts_c[:, 2]
    ok = z > 0.1
    if max_depth is not None:
        ok &= z <= max_depth
    u = intr.fx * (pts_c[:, 0] / np.maximum(z, 1e-6)) + intr.cx
    v = intr.fy * (pts_c[:, 1] / np.maximum(z, 1e-6)) + intr.cy
    ok &= (u >= 0.0) & (u < float(intr.width)) & (v >= 0.0) & (v < float(intr.height))
    return pts_world[ok]


def _equal_density_sample(pts: np.ndarray, target_n: int, seed: int = 42) -> np.ndarray:
    """Subsample or keep `pts` at a uniform target count `target_n` so density does not bias P/R."""
    if len(pts) <= target_n:
        return pts
    rng = np.random.default_rng(seed)
    return pts[rng.choice(len(pts), size=target_n, replace=False)]


def _compute_metrics_v2(
    recon_pts: np.ndarray,
    gt_pts: np.ndarray,
    thresholds: List[float],
    trunc_tau: float,
    equal_sample_n: Optional[int] = 40000,
) -> Dict[str, Any]:
    if len(recon_pts) < 10 or len(gt_pts) < 10:
        return {"error": f"too_few_points (recon={len(recon_pts)}, gt={len(gt_pts)})"}

    if equal_sample_n is not None:
        n_eq = min(equal_sample_n, len(recon_pts), len(gt_pts))
        if n_eq >= 100:
            recon_pts = _equal_density_sample(recon_pts, n_eq, seed=42)
            gt_pts = _equal_density_sample(gt_pts, n_eq, seed=43)

    pcd_r = o3d.geometry.PointCloud()
    pcd_r.points = o3d.utility.Vector3dVector(recon_pts)
    pcd_g = o3d.geometry.PointCloud()
    pcd_g.points = o3d.utility.Vector3dVector(gt_pts)

    d_r2g = np.asarray(pcd_r.compute_point_cloud_distance(pcd_g), dtype=np.float64)
    d_g2r = np.asarray(pcd_g.compute_point_cloud_distance(pcd_r), dtype=np.float64)

    trunc_chamfer = float(0.5 * (np.mean(np.minimum(d_r2g, trunc_tau)) + np.mean(np.minimum(d_g2r, trunc_tau))))
    raw_chamfer = float(0.5 * (np.mean(d_r2g) + np.mean(d_g2r)))

    out: Dict[str, Any] = {
        "num_recon_pts": int(len(recon_pts)),
        "num_gt_pts": int(len(gt_pts)),
        f"truncated_chamfer_{trunc_tau:.2f}m": round(trunc_chamfer, 5),
        "raw_chamfer_m_supp": round(raw_chamfer, 5),
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


def _run_rigid_icp_supp(src_pts: np.ndarray, gt_pts: np.ndarray, max_corr_dist: float) -> Tuple[np.ndarray, float]:
    pcd_s = o3d.geometry.PointCloud()
    pcd_s.points = o3d.utility.Vector3dVector(src_pts)
    pcd_g = o3d.geometry.PointCloud()
    pcd_g.points = o3d.utility.Vector3dVector(gt_pts)
    reg = o3d.pipelines.registration.registration_icp(
        pcd_s, pcd_g, max_corr_dist, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
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
    sx = float(depth_w) / float(intr.width)
    sy = float(depth_h) / float(intr.height)
    fx, fy, cx, cy = intr.fx * sx, intr.fy * sy, intr.cx * sx, intr.cy * sy

    pts_c = (R_w2c @ gt_pts_world.T).T + t_w2c[None, :]
    z = pts_c[:, 2]
    front = z > 0.1
    pts_c, z = pts_c[front], z[front]

    u = np.round(fx * (pts_c[:, 0] / z) + cx).astype(np.int64)
    v = np.round(fy * (pts_c[:, 1] / z) + cy).astype(np.int64)
    in_img = (u >= 0) & (u < depth_w) & (v >= 0) & (v < depth_h)
    u, v, z = u[in_img], v[in_img], z[in_img]

    order = np.argsort(-z)
    gt_depth = np.zeros((depth_h, depth_w), dtype=np.float32)
    gt_depth[v[order], u[order]] = z[order].astype(np.float32)
    return gt_depth


def _select_covisible_subsets(
    sfm_res: SfmReconstructionResult,
    anchor_name: str,
    target_ns: List[int],
) -> Dict[int, List[str]]:
    """Greedily select N-view subsets by shared 3D point covisibility starting from `anchor_name`."""
    all_names = sorted(sfm_res.registered_poses.keys())
    if anchor_name not in sfm_res.registered_poses:
        anchor_name = all_names[0]

    pt_sets = {n: set(sfm_res.registered_poses[n].observed_point3d_ids.tolist()) for n in all_names}
    ordered = [anchor_name]
    pool_pts = set(pt_sets[anchor_name])
    remaining = set(all_names) - {anchor_name}

    while remaining:
        best_n = max(remaining, key=lambda cand: (len(pt_sets[cand] & pool_pts), len(pt_sets[cand])))
        ordered.append(best_n)
        pool_pts |= pt_sets[best_n]
        remaining.remove(best_n)

    subsets: Dict[int, List[str]] = {}
    for n in target_ns:
        k = min(n, len(ordered))
        subsets[n] = ordered[:k]
    return subsets


def _subset_sfm_result(sfm_full: SfmReconstructionResult, keep_names: List[str]) -> SfmReconstructionResult:
    """Create a shallow view-filtered SfmReconstructionResult without re-running SfM."""
    keep_set = set(keep_names)
    sub_poses = {k: v for k, v in sfm_full.registered_poses.items() if k in keep_set}
    sub = copy.copy(sfm_full)
    sub.registered_poses = sub_poses
    sub.num_input_images = len(sub_poses)
    sub.num_registered_images = len(sub_poses)
    sub.registration_ratio = 1.0
    return sub


def _eval_single_view_oracle(
    anchor_view: CalibratedView,
    gt_pts_bbox: np.ndarray,
    gt_pts_frustum: np.ndarray,
    depth_model,
    thresholds: List[float],
    trunc_tau: float,
    aabb_margin: float,
    icp_dist: float,
    fit_shift: bool = False,
) -> Dict[str, Any]:
    """Single-view (N=1) baseline:
    - fit_shift=False: GT camera pose + 1 global median scale (oracle-scale).
    - fit_shift=True:  GT camera pose + 2-param affine inverse-depth scale+shift fit to GT depth.
    """
    t0 = time.perf_counter()
    rgb_full, intr_full = anchor_view.load_rgb_resized(max_dim=640)
    pred = depth_model.predict(rgb_full)
    h, w = pred.depth_map.shape[:2]
    sx = float(w) / float(intr_full.width)
    sy = float(h) / float(intr_full.height)
    intr_d = CameraIntrinsics(
        width=w, height=h, fx=intr_full.fx * sx, fy=intr_full.fy * sy,
        cx=intr_full.cx * sx, cy=intr_full.cy * sy, source=intr_full.source,
        fov_x_deg=intr_full.fov_x_deg, fov_y_deg=intr_full.fov_y_deg,
    )

    gt_d = _render_gt_depth_map(gt_pts_bbox, anchor_view.R_w2c, anchor_view.t_w2c, intr_d, h, w)
    valid_gt = gt_d > 0.1
    if fit_shift:
        d_obs = pred.disparity_map[valid_gt].astype(np.float64)
        inv_gt = 1.0 / gt_d[valid_gt].astype(np.float64)
        A = np.column_stack([d_obs, np.ones_like(d_obs)])
        sol, _, _, _ = np.linalg.lstsq(A, inv_gt, rcond=None)
        if sol[0] > 1e-6 and float(np.min(sol[0] * pred.disparity_map + sol[1])) > 1e-4:
            z_scaled = 1.0 / np.maximum(sol[0] * pred.disparity_map.astype(np.float64) + sol[1], 1e-4)
        else:
            scale_global = float(np.median(gt_d[valid_gt]) / max(float(np.median(pred.depth_map[valid_gt])), 1e-6))
            z_scaled = pred.depth_map.astype(np.float64) * scale_global
        method_name = "single_view_oracle_scale_shift"
        budget_desc = "GT camera pose + 2-param scale+shift fit to GT depth (oracle)"
    else:
        scale_global = float(np.median(gt_d[valid_gt]) / max(float(np.median(pred.depth_map[valid_gt])), 1e-6))
        z_scaled = pred.depth_map.astype(np.float64) * scale_global
        method_name = "single_view_oracle_pose_plus_scale"
        budget_desc = "GT camera pose + 1 global median scale (oracle)"

    vs, us = np.mgrid[0:h, 0:w]
    zf = z_scaled.ravel()
    pts_cam = np.column_stack([(us.ravel() - intr_d.cx) * zf / intr_d.fx, (vs.ravel() - intr_d.cy) * zf / intr_d.fy, zf])
    if len(pts_cam) > 60000:
        pts_cam = pts_cam[:: max(1, len(pts_cam) // 60000)]
    pts_world = (anchor_view.R_w2c.T @ (pts_cam - anchor_view.t_w2c[None, :]).T).T

    pts_bbox = _crop_to_aabb(pts_world, gt_pts_bbox, margin=aabb_margin)
    pts_frustum = _filter_to_camera_frustum(pts_bbox, anchor_view.R_w2c, anchor_view.t_w2c, anchor_view.intrinsics_full)

    bbox_no_icp = _compute_metrics_v2(pts_bbox, gt_pts_bbox, thresholds, trunc_tau, equal_sample_n=40000)
    frust_no_icp = _compute_metrics_v2(pts_frustum, gt_pts_frustum, thresholds, trunc_tau, equal_sample_n=15000)

    pts_icp, fitness = _run_rigid_icp_supp(pts_bbox, gt_pts_bbox, icp_dist)
    bbox_with_icp = _compute_metrics_v2(pts_icp, gt_pts_bbox, thresholds, trunc_tau, equal_sample_n=40000)

    return {
        "method": method_name,
        "num_views": 1,
        "registered_views": 1,
        "alignment_budget": budget_desc,
        "runtime_s": round(time.perf_counter() - t0, 2),
        "bbox_cropped_headline": bbox_no_icp,
        "frustum_restricted_headline": frust_no_icp,
        "icp_supplementary": {"fitness": round(fitness, 4), "bbox_cropped": bbox_with_icp},
    }


def _eval_multiview_subset_v2(
    label: str,
    sfm_full: SfmReconstructionResult,
    subset_names: List[str],
    sim3_full,
    anchor_view: CalibratedView,
    gt_pts_bbox: np.ndarray,
    gt_pts_frustum: np.ndarray,
    depth_model,
    cfg: MultiViewFusionConfig,
    thresholds: List[float],
    trunc_tau: float,
    aabb_margin: float,
    icp_dist: float,
) -> Dict[str, Any]:
    sub_sfm = _subset_sfm_result(sfm_full, subset_names)
    t0 = time.perf_counter()
    fusion_res = fuse_multiview_tsdf_and_mesh(sub_sfm, depth_model, cfg)
    fusion_s = time.perf_counter() - t0

    o3d_mesh = fusion_res.mesh.to_open3d()
    if len(o3d_mesh.triangles) > 0:
        pcd_s = o3d_mesh.sample_points_uniformly(number_of_points=min(60000, max(len(o3d_mesh.triangles) * 2, 15000)))
        pts_sfm = np.asarray(pcd_s.points, dtype=np.float64)
    else:
        pts_sfm = fusion_res.point_cloud.points.astype(np.float64)

    pts_aligned = sim3_full.transform_points(pts_sfm)
    pts_bbox = _crop_to_aabb(pts_aligned, gt_pts_bbox, margin=aabb_margin)
    pts_frustum = _filter_to_camera_frustum(pts_bbox, anchor_view.R_w2c, anchor_view.t_w2c, anchor_view.intrinsics_full)

    bbox_no_icp = _compute_metrics_v2(pts_bbox, gt_pts_bbox, thresholds, trunc_tau)
    frust_no_icp = _compute_metrics_v2(pts_frustum, gt_pts_frustum, thresholds, trunc_tau)

    pts_icp, fitness = _run_rigid_icp_supp(pts_bbox, gt_pts_bbox, icp_dist)
    bbox_with_icp = _compute_metrics_v2(pts_icp, gt_pts_bbox, thresholds, trunc_tau)

    gc.collect()
    return {
        "method": label,
        "num_views": len(subset_names),
        "registered_views": len(subset_names),
        "alignment_budget": "Sim(3) Umeyama on recovered camera centres vs GT/ref centres (NO ICP)",
        "fusion_runtime_s": round(fusion_s, 2),
        "peak_rss_mb": round(fusion_res.peak_ram_mb, 2),
        "mesh_faces": fusion_res.mesh.num_faces,
        "rejected_discontinuity_pct": fusion_res.to_metadata_dict()["rejected_discontinuity_pct"],
        "rejected_consistency_pct": fusion_res.to_metadata_dict()["rejected_consistency_pct"],
        "bbox_cropped_headline": bbox_no_icp,
        "frustum_restricted_headline": frust_no_icp,
        "icp_supplementary": {"fitness": round(fitness, 4), "bbox_cropped": bbox_with_icp},
    }


def _eval_sparse_cloud(
    sfm_full: SfmReconstructionResult,
    sim3_full,
    anchor_view: CalibratedView,
    gt_pts_bbox: np.ndarray,
    gt_pts_frustum: np.ndarray,
    thresholds: List[float],
    trunc_tau: float,
    aabb_margin: float,
) -> Dict[str, Any]:
    """Evaluate raw sparse SfM cloud vs quality-filtered (track >= 3, reproj <= 1.0 px) in GT AABB."""
    raw_aligned = sim3_full.transform_points(sfm_full.sparse_points_xyz)
    raw_bbox = _crop_to_aabb(raw_aligned, gt_pts_bbox, margin=aabb_margin)

    keep = (sfm_full.sparse_point_track_lengths >= 3) & (sfm_full.sparse_point_errors_px <= 1.0)
    filt_aligned = sim3_full.transform_points(sfm_full.sparse_points_xyz[keep])
    filt_bbox = _crop_to_aabb(filt_aligned, gt_pts_bbox, margin=aabb_margin)
    filt_frust = _filter_to_camera_frustum(filt_bbox, anchor_view.R_w2c, anchor_view.t_w2c, anchor_view.intrinsics_full)

    return {
        "num_raw_points": int(len(raw_aligned)),
        "num_raw_in_gt_bbox": int(len(raw_bbox)),
        "raw_bbox_cropped": _compute_metrics_v2(raw_bbox, gt_pts_bbox, thresholds, trunc_tau),
        "num_filtered_points_trk3_err1px": int(len(filt_aligned)),
        "num_filtered_in_gt_bbox": int(len(filt_bbox)),
        "filtered_bbox_cropped": _compute_metrics_v2(filt_bbox, gt_pts_bbox, thresholds, trunc_tau),
        "filtered_frustum_restricted": _compute_metrics_v2(filt_frust, gt_pts_frustum, thresholds, trunc_tau),
    }


def _eval_eth3d_per_view_depth_v2(
    sfm_pipes: SfmReconstructionResult,
    sim3_pipes,
    pipes_views: List[CalibratedView],
    gt_scan_pts: np.ndarray,
    depth_model,
    cfg: MultiViewFusionConfig,
) -> Dict[str, Any]:
    """Compare 3 depth estimators across all 14 ETH3D views (Median & IQR):
    1. Single-view median-scale (1 GT param: median scale)
    2. Single-view oracle scale+shift (2 GT params: 1/Z = s*d + t fit directly to GT depth)
    3. Multi-view SfM-aligned (0 GT depth params: 1/Z = s*d + t fit to SfM sparse pts, scaled by Sim3)
    """
    ref_map = {v.image_path.name: v for v in pipes_views}
    rows = []
    for name in sorted(sfm_pipes.registered_poses.keys()):
        pose = sfm_pipes.registered_poses[name]
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
            source=pose.intrinsics.source, fov_x_deg=pose.intrinsics.fov_x_deg, fov_y_deg=pose.intrinsics.fov_y_deg,
        )
        pred = depth_model.predict(rgb_d)
        gt_d = _render_gt_depth_map(gt_scan_pts, v_gt.R_w2c, v_gt.t_w2c, intr_d, rh, rw)
        valid = gt_d > 0.1
        if int(np.sum(valid)) < 50:
            continue
        gt_v = gt_d[valid].astype(np.float64)

        def _depth_stats(z_pred: np.ndarray) -> Dict[str, float]:
            abs_rel = float(np.mean(np.abs(z_pred - gt_v) / gt_v))
            rmse = float(np.sqrt(np.mean((z_pred - gt_v) ** 2)))
            ratio = np.maximum(z_pred / gt_v, gt_v / np.maximum(z_pred, 1e-6))
            d1 = float(np.mean(ratio < 1.25))
            return {"abs_rel": abs_rel, "rmse": rmse, "delta1": d1}

        # 1. Single-view median scale
        sv_raw = pred.depth_map[valid].astype(np.float64)
        sv_med = sv_raw * float(np.median(gt_v) / max(np.median(sv_raw), 1e-6))
        s1 = _depth_stats(sv_med)

        # 2. Single-view oracle scale+shift to GT depth
        d_obs = pred.disparity_map[valid].astype(np.float64)
        inv_gt = 1.0 / gt_v
        A = np.column_stack([d_obs, np.ones_like(d_obs)])
        sol, _, _, _ = np.linalg.lstsq(A, inv_gt, rcond=None)
        if sol[0] > 1e-6 and float(np.min(sol[0] * d_obs + sol[1])) > 1e-4:
            sv_oracle = 1.0 / (sol[0] * d_obs + sol[1])
        else:
            sv_oracle = sv_med
        s2 = _depth_stats(sv_oracle)

        # 3. Multi-view SfM-aligned (no GT depth)
        aligned_sfm, *_ = align_monocular_disparity_to_sparse_depth(pred.disparity_map, pose, intr_d, cfg)
        if aligned_sfm is None:
            continue
        mv_metric = aligned_sfm[valid].astype(np.float64) * float(sim3_pipes.scale)
        s3 = _depth_stats(mv_metric)

        rows.append({"view": name, "sv_median": s1, "sv_oracle_ss": s2, "mv_sfm_aligned": s3})

    def _med_iqr(key: str, metric: str) -> Dict[str, float]:
        arr = np.asarray([r[key][metric] for r in rows], dtype=np.float64)
        q25, med, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
        return {"median": round(float(med), 4), "iqr": round(float(q75 - q25), 4)}

    return {
        "num_views": len(rows),
        "single_view_median_scale": {m: _med_iqr("sv_median", m) for m in ("abs_rel", "rmse", "delta1")},
        "single_view_oracle_scale_shift": {m: _med_iqr("sv_oracle_ss", m) for m in ("abs_rel", "rmse", "delta1")},
        "multiview_sfm_aligned": {m: _med_iqr("mv_sfm_aligned", m) for m in ("abs_rel", "rmse", "delta1")},
    }


def _plot_pr_curves(f_rows: List[Dict], p_rows: List[Dict], out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=140)
    fig.patch.set_facecolor("#0f172a")
    configs = [
        (axes[0], "Fountain (GT AABB, tau=0.05m)", f_rows, "0.05m"),
        (axes[1], "ETH3D 'pipes' (GT AABB, tau=0.10m)", p_rows, "0.10m"),
    ]
    for ax, title, rows, tk in configs:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#475569")
        # Plot N=1 (oracle scale), 3, 5, 10, all (skip duplicate N=1 scale+shift row from x-axis)
        plot_rows = [r for r in rows if r["method"] != "single_view_oracle_scale_shift"]
        ns = [r["num_views"] for r in plot_rows]
        precs = [r["bbox_cropped_headline"]["thresholds"][tk]["precision"] for r in plot_rows]
        recs = [r["bbox_cropped_headline"]["thresholds"][tk]["recall"] for r in plot_rows]
        f1s = [r["bbox_cropped_headline"]["thresholds"][tk]["f_score"] for r in plot_rows]
        x = range(len(ns))
        ax.plot(x, precs, "-o", color="#38bdf8", label="Precision (No ICP)")
        ax.plot(x, recs, "-s", color="#facc15", label="Recall (No ICP)")
        ax.plot(x, f1s, "-^", color="#34d399", linewidth=2.0, label="F-score (No ICP)")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"N={n}" for n in ns], color="#cbd5e1")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Score", color="#cbd5e1")
        ax.set_title(title, color="#f8fafc", fontsize=10, fontweight="bold")
        ax.grid(True, color="#334155", linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=8, facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc")
    plt.tight_layout()
    fig.savefig(str(out_png), facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    settings = get_settings()
    out_dir = settings.outputs_dir / "experiment_d_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_model = create_depth_model("depth_anything_v2_small")
    fountain_views, fountain_ref_path = load_fountain_dataset(settings.datasets_dir)
    pipes_views, pipes_scan_path = load_eth3d_pipes_dataset(settings.datasets_dir)

    fountain_gt_pts = _sample_surface_points(fountain_ref_path, max_points=100000)
    pipes_gt_pts = _sample_surface_points(pipes_scan_path, max_points=100000)

    f_anchor = fountain_views[0]
    p_anchor = pipes_views[1]  # DSC_0635.JPG
    f_gt_frust = _filter_to_camera_frustum(fountain_gt_pts, f_anchor.R_w2c, f_anchor.t_w2c, f_anchor.intrinsics_full)
    p_gt_frust = _filter_to_camera_frustum(pipes_gt_pts, p_anchor.R_w2c, p_anchor.t_w2c, p_anchor.intrinsics_full)

    with tempfile.TemporaryDirectory(prefix="hologen_expd_v2_") as tmp_u:
        tmp_path = Path(tmp_u)

        # 1. Run SfM ONCE on all Fountain views and ONCE on all ETH3D views
        sfm_fountain = run_incremental_sfm(
            image_paths=[v.image_path for v in fountain_views],
            sfm_max_image_size=1600, camera_model="SIMPLE_RADIAL", fallback_camera_model="OPENCV",
            single_camera=True, max_num_features=4096, num_threads=4,
            undistort=True, undistorted_dir=tmp_path / "fountain_undist",
        )
        f_pose_eval, sim3_fountain = evaluate_sfm_poses_against_reference(
            sfm_fountain, {v.image_path.name: v for v in fountain_views}
        )

        sfm_pipes = run_incremental_sfm(
            image_paths=[v.image_path for v in pipes_views],
            sfm_max_image_size=1600, camera_model="SIMPLE_RADIAL", fallback_camera_model="OPENCV",
            single_camera=True, max_num_features=4096, num_threads=4,
            undistort=True, undistorted_dir=tmp_path / "pipes_undist",
        )
        p_pose_eval, sim3_pipes = evaluate_sfm_poses_against_reference(
            sfm_pipes, {v.image_path.name: v for v in pipes_views}
        )

        f_subsets = _select_covisible_subsets(sfm_fountain, f_anchor.image_path.name, [3, 5, 10, sfm_fountain.num_registered_images])
        p_subsets = _select_covisible_subsets(sfm_pipes, p_anchor.image_path.name, [3, 5, 10, sfm_pipes.num_registered_images])

        # 2. Tune consistency thresholds on Fountain (N=10 covisible subset) by Precision & F-score at tau=0.05m
        tuning_candidates = [("strict", 0.05, 0.12), ("default", 0.10, 0.18), ("relaxed", 0.20, 0.30)]
        tuning_results = []
        best_cfg = None
        best_f1 = -1.0
        for label, rel_d, rgb_t in tuning_candidates:
            cfg_t = MultiViewFusionConfig(
                depth_max_dim=640, discontinuity_threshold=0.08,
                consistency_rel_depth_thresh=rel_d, consistency_rgb_thresh=rgb_t,
                min_consistent_views=2, sparse_depth_cap_mult=1.35, min_tsdf_view_support=2,
            )
            r_t = _eval_multiview_subset_v2(
                f"tune_{label}", sfm_fountain, f_subsets[10], sim3_fountain, f_anchor,
                fountain_gt_pts, f_gt_frust, depth_model, cfg_t,
                thresholds=[0.02, 0.05, 0.10], trunc_tau=0.05, aabb_margin=0.10, icp_dist=0.10,
            )
            m05 = r_t["bbox_cropped_headline"]["thresholds"]["0.05m"]
            tuning_results.append({
                "setting": label, "consistency_rel_depth_thresh": rel_d, "consistency_rgb_thresh": rgb_t,
                "precision_0.05m": m05["precision"], "recall_0.05m": m05["recall"], "f_score_0.05m": m05["f_score"],
                "truncated_chamfer_0.05m": r_t["bbox_cropped_headline"]["truncated_chamfer_0.05m"],
            })
            if m05["f_score"] > best_f1:
                best_f1 = m05["f_score"]
                best_cfg = cfg_t
        assert best_cfg is not None

        # 3. Fountain N=1 (scale-only + scale+shift), 3, 5, 10, 33 + Sparse Cloud
        f_n1_s = _eval_single_view_oracle(f_anchor, fountain_gt_pts, f_gt_frust, depth_model, [0.02, 0.05, 0.10], 0.05, 0.10, 0.10, fit_shift=False)
        f_n1_ss = _eval_single_view_oracle(f_anchor, fountain_gt_pts, f_gt_frust, depth_model, [0.02, 0.05, 0.10], 0.05, 0.10, 0.10, fit_shift=True)
        f_rows = [f_n1_s, f_n1_ss]
        for n in (3, 5, 10, sfm_fountain.num_registered_images):
            f_rows.append(
                _eval_multiview_subset_v2(
                    f"fountain_N{n}", sfm_fountain, f_subsets[n], sim3_fountain, f_anchor,
                    fountain_gt_pts, f_gt_frust, depth_model, best_cfg,
                    [0.02, 0.05, 0.10], 0.05, 0.10, 0.10,
                )
            )
        f_sparse = _eval_sparse_cloud(sfm_fountain, sim3_fountain, f_anchor, fountain_gt_pts, f_gt_frust, [0.02, 0.05, 0.10], 0.05, 0.10)

        # 4. ETH3D 'pipes' Out-of-Sample N=1 (scale-only + scale+shift), 3, 5, 10, 14 + Sparse Cloud
        p_n1_s = _eval_single_view_oracle(p_anchor, pipes_gt_pts, p_gt_frust, depth_model, [0.05, 0.10, 0.20], 0.10, 0.15, 0.20, fit_shift=False)
        p_n1_ss = _eval_single_view_oracle(p_anchor, pipes_gt_pts, p_gt_frust, depth_model, [0.05, 0.10, 0.20], 0.10, 0.15, 0.20, fit_shift=True)
        p_rows = [p_n1_s, p_n1_ss]
        for n in (3, 5, 10, sfm_pipes.num_registered_images):
            p_rows.append(
                _eval_multiview_subset_v2(
                    f"pipes_N{n}", sfm_pipes, p_subsets[n], sim3_pipes, p_anchor,
                    pipes_gt_pts, p_gt_frust, depth_model, best_cfg,
                    [0.05, 0.10, 0.20], 0.10, 0.15, 0.20,
                )
            )

        # 4b. ETH3D sensitivity row with DEFAULT thresholds (0.10 / 0.18) on N=14
        default_cfg = MultiViewFusionConfig(
            depth_max_dim=640, discontinuity_threshold=0.08,
            consistency_rel_depth_thresh=0.10, consistency_rgb_thresh=0.18,
            min_consistent_views=2, sparse_depth_cap_mult=1.35, min_tsdf_view_support=2,
        )
        p_n14_default_sens = _eval_multiview_subset_v2(
            "pipes_N14_default_thresh_0.10_0.18", sfm_pipes, p_subsets[sfm_pipes.num_registered_images],
            sim3_pipes, p_anchor, pipes_gt_pts, p_gt_frust, depth_model, default_cfg,
            [0.05, 0.10, 0.20], 0.10, 0.15, 0.20,
        )

        # 4c. Alignment-sensitivity analysis on ETH3D N=14:
        # Compare centre-only Sim(3) vs Sim(3) using camera centres + optical-axis points (c + s * fwd)
        from app.vision.sfm import align_umeyama_sim3
        reg_p_names = sorted(sfm_pipes.registered_poses.keys())
        gt_p_map = {v.image_path.name: v for v in pipes_views}
        c_sfm = np.vstack([sfm_pipes.registered_poses[n].camera_center_w for n in reg_p_names])
        c_gt = np.vstack([gt_p_map[n].camera_center_world for n in reg_p_names])
        fwd_sfm = np.vstack([sfm_pipes.registered_poses[n].R_w2c.T @ np.array([0.0, 0.0, 1.0]) for n in reg_p_names])
        fwd_gt = np.vstack([gt_p_map[n].R_w2c.T @ np.array([0.0, 0.0, 1.0]) for n in reg_p_names])
        # Optical-axis tip scaled proportionally to initial Umeyama scale
        aug_sfm = np.vstack([c_sfm, c_sfm + (1.0 / max(sim3_pipes.scale, 1e-4)) * fwd_sfm])
        aug_gt = np.vstack([c_gt, c_gt + 1.0 * fwd_gt])
        sim3_pipes_axes = align_umeyama_sim3(aug_sfm, aug_gt, with_scale=True)
        p_n14_axes = _eval_multiview_subset_v2(
            "pipes_N14_sim3_centres_plus_axes", sfm_pipes, p_subsets[sfm_pipes.num_registered_images],
            sim3_pipes_axes, p_anchor, pipes_gt_pts, p_gt_frust, depth_model, best_cfg,
            [0.05, 0.10, 0.20], 0.10, 0.15, 0.20,
        )

        p_sparse = _eval_sparse_cloud(sfm_pipes, sim3_pipes, p_anchor, pipes_gt_pts, p_gt_frust, [0.05, 0.10, 0.20], 0.10, 0.15)

        # 5. ETH3D Per-View GT Depth (3 estimators)
        eth3d_depth = _eval_eth3d_per_view_depth_v2(sfm_pipes, sim3_pipes, pipes_views, pipes_gt_pts, depth_model, best_cfg)

    custom_dir = settings.datasets_dir / "custom_scene_01"
    custom_report = {"status": "not_present_in_workspace", "expected_path": str(custom_dir)}

    _plot_pr_curves(f_rows, p_rows, out_dir / "experiment_d_v2_pr_curves.png")

    payload = {
        "phase": "phase_7e2_experiment_d_v2",
        "sfm_fountain": {"registered": sfm_fountain.num_registered_images, "total": sfm_fountain.num_input_images, "pose_eval": f_pose_eval, "peak_rss_mb": sfm_fountain.peak_ram_mb},
        "sfm_pipes": {"registered": sfm_pipes.num_registered_images, "total": sfm_pipes.num_input_images, "camera_model": sfm_pipes.shared_camera_model, "fallback_triggered": sfm_pipes.fallback_triggered, "pose_eval": p_pose_eval, "peak_rss_mb": sfm_pipes.peak_ram_mb},
        "consistency_tuning_fountain_N10": tuning_results,
        "selected_consistency_config": {"consistency_rel_depth_thresh": best_cfg.consistency_rel_depth_thresh, "consistency_rgb_thresh": best_cfg.consistency_rgb_thresh, "min_consistent_views": best_cfg.min_consistent_views, "sparse_depth_cap_mult": best_cfg.sparse_depth_cap_mult, "min_tsdf_view_support": best_cfg.min_tsdf_view_support},
        "fountain_reference_rgbd_geometry": f_rows,
        "fountain_sparse_cloud": f_sparse,
        "eth3d_pipes_ground_truth_scan": p_rows,
        "eth3d_pipes_default_thresh_sensitivity": p_n14_default_sens,
        "eth3d_pipes_alignment_sensitivity_centres_plus_axes": p_n14_axes,
        "eth3d_pipes_sparse_cloud": p_sparse,
        "eth3d_per_view_gt_depth": eth3d_depth,
        "custom_scene_01": custom_report,
    }
    (out_dir / "experiment_d_v2_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Experiment D v2 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
