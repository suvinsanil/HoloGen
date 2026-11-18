"""Phase 7b: N-view incremental SfM via pycolmap on Fountain (reference RGB-D trajectory) and
ETH3D pipes (ground-truth calibration). Logs registered/total, runtime, peak RAM, and Sim(3)
Umeyama-aligned camera pose errors. All numbers written to outputs/phase_7b/phase_7b_results.json.
"""

import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.evaluation.multiview_datasets import load_eth3d_pipes_dataset, load_fountain_dataset  # noqa: E402
from app.vision.sfm import (  # noqa: E402
    evaluate_sfm_poses_against_reference,
    render_sfm_sparse_reconstruction_visual,
    run_incremental_sfm,
)


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_case(name, views, semantics_label, out_dir, **sfm_kwargs):
    ref_map = {v.image_path.name: v for v in views}
    res = run_incremental_sfm(image_paths=[v.image_path for v in views], **sfm_kwargs)
    metrics, sim3 = evaluate_sfm_poses_against_reference(res, ref_map)
    render_sfm_sparse_reconstruction_visual(
        res, sim3, ref_map, scene_title=name, pose_semantics_label=semantics_label,
        output_png_path=out_dir / f"{name}_sparse_sfm.png",
    )
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sim3.transform_points(res.sparse_points_xyz))
        pcd.colors = o3d.utility.Vector3dVector(res.sparse_points_rgb.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(str(out_dir / f"{name}_sparse_aligned.ply"), pcd)
    except Exception as exc:  # visual artefact only; never block metrics
        print(f"[warn] PLY export failed for {name}: {exc}")
    out = {
        "case": name,
        "pose_semantics": semantics_label,
        "sfm_config": {k: v for k, v in sfm_kwargs.items()},
        "sfm": res.to_metadata_dict(),
        "pose_eval": metrics,
        "sparse_point_error_px_median": round(float(np.median(res.sparse_point_errors_px)), 4)
        if res.num_sparse_points else None,
        "sparse_track_length_median": int(np.median(res.sparse_point_track_lengths))
        if res.num_sparse_points else None,
    }
    print(json.dumps(out, indent=2), flush=True)
    return out


def main() -> int:
    settings = get_settings()
    out_dir = settings.outputs_dir / "phase_7b"
    out_dir.mkdir(parents=True, exist_ok=True)

    fountain, _ = load_fountain_dataset(settings.datasets_dir)
    pipes, _ = load_eth3d_pipes_dataset(settings.datasets_dir)
    fountain_label = "Reference RGB-D trajectory (key.log)"
    pipes_label = "ETH3D ground-truth calibration"

    cases = []
    cases.append(run_case(
        "fountain_8views", fountain[::4][:8], fountain_label, out_dir,
        sfm_max_image_size=1600, camera_model="SIMPLE_RADIAL", single_camera=True,
        max_num_features=4096, num_threads=4,
    ))
    cases.append(run_case(
        "fountain_all33", fountain, fountain_label, out_dir,
        sfm_max_image_size=1600, camera_model="SIMPLE_RADIAL", single_camera=True,
        max_num_features=4096, num_threads=4,
    ))
    for model in ("SIMPLE_RADIAL", "OPENCV"):
        cases.append(run_case(
            f"eth3d_pipes_all14_{model.lower()}", pipes, pipes_label, out_dir,
            sfm_max_image_size=1600, camera_model=model, single_camera=True,
            max_num_features=4096, num_threads=4,
        ))

    import psutil
    payload = {
        "phase": "phase_7b_nview_sfm",
        "commit": _git_hash(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "total_ram_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2),
            "sfm_device": "cpu",
        },
        "pycolmap_version": __import__("pycolmap").__version__,
        "cases": cases,
    }
    (out_dir / "phase_7b_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
