"""Experiment A Runner: Benchmark pretrained monocular depth models & edge-preserving refinement
on the 8-scene calibrated RGB-D benchmark subset (NYU Depth V2, SUN RGB-D, TUM RGB-D, Redwood Indoor).
"""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.core.logging import get_logger
from app.evaluation.depth_metrics import (
    compute_depth_metrics,
    load_rgbd_benchmark_samples,
)
from app.geometry.export import save_depth_visualization_png
from app.vision.depth_estimation import create_depth_model
from app.vision.depth_refinement import refine_depth_map
from app.vision.uncertainty import compute_uncertainty_proxy

logger = get_logger("scripts.run_experiment_a")


def get_git_commit_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        )
        return out.strip()
    except Exception:
        return "unknown"


def main() -> int:
    settings = get_settings()
    device = settings.resolve_device()
    commit_hash = get_git_commit_short()

    rgbd_root = settings.datasets_dir / "nyu_v2"
    frames = load_rgbd_benchmark_samples(rgbd_root)
    logger.info("Loaded %d calibrated RGB-D benchmark frames.", len(frames))

    exp_dir = settings.outputs_dir / "experiment_a"
    exp_dir.mkdir(parents=True, exist_ok=True)

    model_configs = [
        {
            "model_name": "depth_anything_v2_small",
            "alignment_mode": "disparity_scale_shift",
            "eval_note": "Per-image least-squares scale-shift in disparity space",
        },
        {
            "model_name": "dpt_hybrid_midas",
            "alignment_mode": "disparity_scale_shift",
            "eval_note": "Per-image least-squares scale-shift in disparity space",
        },
        {
            "model_name": "depth_anything_v2_metric_indoor_small",
            "alignment_mode": "none",
            "eval_note": "Direct unaligned metric prediction (meters)",
        },
        {
            "model_name": "depth_anything_v2_metric_indoor_small",
            "alignment_mode": "depth_scale_shift",
            "eval_note": "Metric prediction + per-image scale-shift alignment",
        },
    ]

    model_summaries = []
    # Cache predictions of the primary model for the refinement ablation
    primary_preds = {}

    for mcfg in model_configs:
        m_name = mcfg["model_name"]
        align_mode = mcfg["alignment_mode"]
        model = create_depth_model(m_name, device=device)
        model.load()

        # Warmup pass on first frame
        _ = model.predict(frames[0].rgb_uint8)

        per_scene = []
        inf_times = []
        vram_peaks = []
        uncertainty_means = []

        for frame in frames:
            pred = model.predict(frame.rgb_uint8)
            inf_times.append(pred.inference_time_ms)
            vram_peaks.append(pred.vram_allocated_mb)

            unc = compute_uncertainty_proxy(pred.depth_map)
            uncertainty_means.append(unc.mean_uncertainty_proxy)

            if m_name == "depth_anything_v2_small" and align_mode == "disparity_scale_shift":
                primary_preds[frame.scene_id] = pred

            metrics = compute_depth_metrics(
                pred_depth=pred.depth_map,
                gt_depth=frame.gt_depth_meters,
                valid_mask=frame.valid_mask,
                alignment_mode=align_mode,
                raw_disparity=pred.raw_prediction if not pred.is_metric else None,
                min_depth=0.4,
                max_depth=10.0,
            )
            per_scene.append(
                {
                    "scene_id": frame.scene_id,
                    "dataset": frame.dataset_name,
                    "inference_ms": round(pred.inference_time_ms, 2),
                    "vram_mb": round(pred.vram_allocated_mb, 2),
                    **metrics.to_dict(),
                }
            )

            # Save visualizations for the NYU Depth V2 frame
            if frame.scene_id == "nyu_depth_v2_office" and align_mode != "depth_scale_shift":
                vis_base = exp_dir / f"{m_name}_{frame.scene_id}"
                save_depth_visualization_png(
                    pred.disparity_map, vis_base.with_name(f"{m_name}_heatmap.png"), mode="heatmap"
                )
                save_depth_visualization_png(
                    pred.disparity_map,
                    vis_base.with_name(f"{m_name}_grayscale.png"),
                    mode="grayscale",
                )
                save_depth_visualization_png(
                    pred.disparity_map,
                    vis_base.with_name(f"{m_name}_inverted.png"),
                    mode="inverted",
                )
                save_depth_visualization_png(
                    unc.uncertainty_proxy_map,
                    vis_base.with_name(f"{m_name}_uncertainty_proxy.png"),
                    mode="heatmap",
                )

        summary = {
            "model_name": m_name,
            "model_id": model.model_id,
            "license": model.license,
            "is_metric": model.is_metric,
            "device": model.device,
            "alignment_mode": align_mode,
            "eval_note": mcfg["eval_note"],
            "num_scenes": len(frames),
            "mean_inference_ms": round(float(np.mean(inf_times)), 2),
            "peak_vram_mb": round(float(np.max(vram_peaks)), 2),
            "abs_rel": round(float(np.mean([s["abs_rel"] for s in per_scene])), 4),
            "sq_rel": round(float(np.mean([s["sq_rel"] for s in per_scene])), 4),
            "rmse": round(float(np.mean([s["rmse"] for s in per_scene])), 4),
            "rmse_log": round(float(np.mean([s["rmse_log"] for s in per_scene])), 4),
            "delta_1": round(float(np.mean([s["delta_1"] for s in per_scene])), 4),
            "delta_2": round(float(np.mean([s["delta_2"] for s in per_scene])), 4),
            "delta_3": round(float(np.mean([s["delta_3"] for s in per_scene])), 4),
            "edge_f1": round(float(np.mean([s["edge_f1"] for s in per_scene])), 4),
            "gradient_correlation": round(
                float(np.mean([s["gradient_correlation"] for s in per_scene])), 4
            ),
            "mean_uncertainty_proxy": round(float(np.mean(uncertainty_means)), 4),
            "per_scene": per_scene,
        }
        model_summaries.append(summary)
        model.unload()

    # Refinement ablation on primary model (depth_anything_v2_small)
    refinement_methods = ["none", "bilateral", "guided", "guided_bilateral"]
    refinement_summaries = []
    for ref_method in refinement_methods:
        ref_times = []
        ref_metrics_list = []
        for frame in frames:
            pred = primary_preds[frame.scene_id]
            # First align raw disparity to metric depth
            base_m = compute_depth_metrics(
                pred_depth=pred.depth_map,
                gt_depth=frame.gt_depth_meters,
                valid_mask=frame.valid_mask,
                alignment_mode="disparity_scale_shift",
                raw_disparity=pred.raw_prediction,
                min_depth=0.4,
                max_depth=10.0,
            )
            inv_aligned = np.clip(
                base_m.scale_factor * pred.raw_prediction + base_m.shift_factor,
                1.0 / 10.0,
                1.0 / 0.4,
            )
            aligned_depth = (1.0 / inv_aligned).astype(np.float32)

            ref_res = refine_depth_map(
                depth_map=aligned_depth,
                rgb_uint8=frame.rgb_uint8,
                method=ref_method,
            )
            ref_times.append(ref_res.processing_time_ms)
            m_ref = compute_depth_metrics(
                pred_depth=ref_res.refined_depth,
                gt_depth=frame.gt_depth_meters,
                valid_mask=frame.valid_mask,
                alignment_mode="none",
                min_depth=0.4,
                max_depth=10.0,
            )
            ref_metrics_list.append(m_ref.to_dict())

        refinement_summaries.append(
            {
                "refinement_method": ref_method,
                "mean_refinement_ms": round(float(np.mean(ref_times)), 3),
                "abs_rel": round(float(np.mean([m["abs_rel"] for m in ref_metrics_list])), 4),
                "rmse": round(float(np.mean([m["rmse"] for m in ref_metrics_list])), 4),
                "delta_1": round(float(np.mean([m["delta_1"] for m in ref_metrics_list])), 4),
                "edge_f1": round(float(np.mean([m["edge_f1"] for m in ref_metrics_list])), 4),
                "gradient_correlation": round(
                    float(np.mean([m["gradient_correlation"] for m in ref_metrics_list])), 4
                ),
            }
        )

    payload = {
        "experiment": "Experiment A: Depth Model & Edge-Preserving Refinement Benchmark",
        "git_commit": commit_hash,
        "device": device,
        "dataset": "8-Scene Calibrated Indoor RGB-D Suite (NYU Depth V2, SUN RGB-D, TUM RGB-D, Redwood Indoor 00-04)",
        "models": model_summaries,
        "refinement_ablation": refinement_summaries,
    }

    out_json = exp_dir / "experiment_a_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved Experiment A results to %s", out_json)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
