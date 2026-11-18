"""Phase 6.5 Hardening Script:
1. Validate monocular uncertainty proxy on the 8 calibrated RGB-D scenes by computing pixelwise
   Spearman rank correlation against true absolute depth error (after Experiment A's scale-shift alignment).
2. Verify that the viewer-decimated GLB (120k faces) for sample_01 preserves accurate projective UVs
   and embedded texture compared to the full-resolution GLB (545k faces), rendering before/after
   screenshots into outputs/baseline_sample_01_building/.
"""

import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.stats import spearmanr
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.core.logging import get_logger
from app.evaluation.depth_metrics import (
    align_disparity_least_squares,
    load_rgbd_benchmark_samples,
)
from app.vision.depth_estimation import create_depth_model
from app.vision.uncertainty import compute_uncertainty_proxy

logger = get_logger("scripts.validate_uncertainty_and_glb_uvs")


def render_glb_uv_textured_orthographic(
    glb_path: Path,
    canvas_w: int = 640,
    canvas_h: int = 442,
) -> tuple[np.ndarray, dict]:
    """Load a GLB via trimesh, sample colors strictly from its embedded baseColorTexture using
    its stored per-vertex UV coordinates (s, t), and rasterize the textured triangles.
    """
    scene = trimesh.load(str(glb_path), force="scene")
    geom = list(scene.geometry.values())[0]

    verts = np.asarray(geom.vertices, dtype=np.float32)
    faces = np.asarray(geom.faces, dtype=np.int32)
    uvs = np.asarray(geom.visual.uv, dtype=np.float32)
    pil_tex = geom.visual.material.baseColorTexture
    tex_rgb = np.asarray(pil_tex.convert("RGB"), dtype=np.uint8)
    th, tw = tex_rgb.shape[:2]

    # Sample texture color at each vertex's UV (s, t)
    u_tex = np.clip(np.round(uvs[:, 0] * float(tw - 1)).astype(np.int32), 0, tw - 1)
    v_tex = np.clip(np.round((1.0 - uvs[:, 1]) * float(th - 1)).astype(np.int32), 0, th - 1)
    vert_bgr = tex_rgb[v_tex, u_tex, ::-1]

    # Project vertices to image canvas using UV coordinates (which correspond to camera projection)
    # and also verify 3D glTF coordinates (X_gltf, Y_gltf, Z_gltf) where -Z_gltf is depth
    z_cam = -verts[:, 2]
    tri_z = np.mean(z_cam[faces], axis=1)
    tri_order = np.argsort(-tri_z)  # Back-to-front painter's algorithm

    px_u = np.clip(np.round(uvs[:, 0] * float(canvas_w - 1)).astype(np.int32), 0, canvas_w - 1)
    px_v = np.clip(
        np.round((1.0 - uvs[:, 1]) * float(canvas_h - 1)).astype(np.int32), 0, canvas_h - 1
    )
    pts_2d = np.stack([px_u, px_v], axis=-1)

    canvas = np.full((canvas_h, canvas_w, 3), 18, dtype=np.uint8)
    for tid in tri_order:
        f = faces[tid]
        poly = pts_2d[f]
        mean_bgr = np.mean(vert_bgr[f], axis=0)
        color = (int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2]))
        cv2.fillConvexPoly(canvas, poly, color)

    stats = {
        "glb_file": glb_path.name,
        "file_size_mb": round(glb_path.stat().st_size / (1024.0 * 1024.0), 3),
        "num_vertices": int(len(verts)),
        "num_faces": int(len(faces)),
        "uv_min": [round(float(x), 4) for x in np.min(uvs, axis=0)],
        "uv_max": [round(float(x), 4) for x in np.max(uvs, axis=0)],
        "texture_resolution": [int(tw), int(th)],
    }
    return canvas, stats


def main() -> int:
    settings = get_settings()
    device = settings.resolve_device()

    # 1. Uncertainty-proxy Spearman correlation on the 8 RGB-D scenes
    frames = load_rgbd_benchmark_samples(settings.datasets_dir / "nyu_v2")
    model = create_depth_model("depth_anything_v2_small", device=device)
    model.load()

    per_scene_corr = []
    for frame in frames:
        pred = model.predict(frame.rgb_uint8)
        mask = (
            frame.valid_mask
            & np.isfinite(frame.gt_depth_meters)
            & (frame.gt_depth_meters >= 0.4)
            & (frame.gt_depth_meters <= 10.0)
        )
        aligned_depth, s, t = align_disparity_least_squares(
            pred.raw_prediction,
            frame.gt_depth_meters,
            mask,
            min_depth=0.4,
            max_depth=10.0,
        )
        unc_res = compute_uncertainty_proxy(aligned_depth)

        # Erode mask by 2 pixels to avoid missing-depth sensor borders
        kernel = np.ones((5, 5), dtype=np.uint8)
        eval_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        u_vals = unc_res.uncertainty_proxy_map[eval_mask].astype(np.float64)
        abs_err = np.abs(aligned_depth[eval_mask] - frame.gt_depth_meters[eval_mask]).astype(
            np.float64
        )
        rel_err = abs_err / np.maximum(frame.gt_depth_meters[eval_mask].astype(np.float64), 0.4)

        rho_abs, pval_abs = spearmanr(u_vals, abs_err)
        rho_rel, pval_rel = spearmanr(u_vals, rel_err)

        per_scene_corr.append(
            {
                "scene_id": frame.scene_id,
                "dataset": frame.dataset_name,
                "valid_pixels": int(np.sum(eval_mask)),
                "spearman_rho_abs_err": round(float(rho_abs), 4),
                "p_value_abs_err": float(f"{pval_abs:.3e}"),
                "spearman_rho_rel_err": round(float(rho_rel), 4),
                "p_value_rel_err": float(f"{pval_rel:.3e}"),
                "mean_abs_err_m": round(float(np.mean(abs_err)), 4),
                "mean_uncertainty_proxy": round(float(np.mean(u_vals)), 4),
            }
        )

    mean_rho_abs = round(
        float(np.mean([r["spearman_rho_abs_err"] for r in per_scene_corr])), 4
    )
    mean_rho_rel = round(
        float(np.mean([r["spearman_rho_rel_err"] for r in per_scene_corr])), 4
    )

    # 2. Verify decimated GLB vs Full-Res GLB UV & Texture on sample_01_building
    s01_dir = settings.outputs_dir / "baseline_sample_01_building"
    full_glb = s01_dir / "scene_full.glb"
    viewer_glb = s01_dir / "scene.glb"

    img_full, stats_full = render_glb_uv_textured_orthographic(full_glb, 640, 442)
    img_viewer, stats_viewer = render_glb_uv_textured_orthographic(viewer_glb, 640, 442)

    before_path = s01_dir / "uv_texture_before_full.png"
    after_path = s01_dir / "uv_texture_after_viewer.png"
    comp_path = s01_dir / "uv_texture_full_vs_viewer.png"

    cv2.imwrite(str(before_path), img_full)
    cv2.imwrite(str(after_path), img_viewer)

    # Side-by-side comparison image with labels
    banner_h = 34
    comp = np.full((442 + banner_h, 640 * 2 + 12, 3), 24, dtype=np.uint8)
    comp[banner_h:, :640] = img_full
    comp[banner_h:, 640 + 12 :] = img_viewer
    cv2.putText(
        comp,
        f"Before: Full-Res GLB ({stats_full['num_faces']} faces, {stats_full['file_size_mb']} MB)",
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        comp,
        f"After: Viewer Decimated GLB ({stats_viewer['num_faces']} faces, {stats_viewer['file_size_mb']} MB)",
        (652 + 12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (80, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(comp_path), comp)

    mse = float(np.mean((img_full.astype(np.float64) - img_viewer.astype(np.float64)) ** 2))
    psnr_db = float(10.0 * np.log10((255.0 ** 2) / max(mse, 1e-8)))
    mean_l1_diff = float(np.mean(np.abs(img_full.astype(np.float64) - img_viewer.astype(np.float64))))

    uv_verification = {
        "sample": "baseline_sample_01_building",
        "full_glb": stats_full,
        "viewer_glb": stats_viewer,
        "rendered_psnr_db": round(psnr_db, 2),
        "rendered_mean_rgb_diff_0_255": round(mean_l1_diff, 3),
        "screenshots": {
            "before_full_png": str(before_path),
            "after_viewer_png": str(after_path),
            "side_by_side_png": str(comp_path),
        },
    }

    payload = {
        "uncertainty_proxy_spearman_validation": {
            "num_scenes": len(per_scene_corr),
            "mean_spearman_rho_abs_err": mean_rho_abs,
            "mean_spearman_rho_rel_err": mean_rho_rel,
            "per_scene": per_scene_corr,
        },
        "glb_uv_decimation_verification": uv_verification,
    }

    out_json = settings.outputs_dir / "experiment_a" / "uncertainty_and_uv_validation.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
