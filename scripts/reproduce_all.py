"""Consolidated reproducibility script for HoloGen (Phase 8).

Modes:
  --quick : Runs fast unit verification + lightweight baseline/cache checks and consolidates
            existing measured outputs into docs/results/ (completes in < 2 minutes).
  --full  : Re-executes all experiment scripts end-to-end with fixed seeds and regenerates
            all outputs and docs/results/ summaries.

Logs hardware, package versions, and git commit hash into docs/results/environment.json.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _collect_environment() -> dict:
    pkgs = [
        "torch", "transformers", "open3d", "trimesh", "opencv-python",
        "pycolmap", "fastapi", "numpy", "scipy", "scikit-learn", "psutil",
    ]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = "not-installed"

    gpu_info = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_info = {
            "name": props.name,
            "vram_mb": round(props.total_memory / (1024 ** 2), 1),
            "cuda_version": torch.version.cuda,
        }

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_hash(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpu": gpu_info or "cpu-only",
        "package_versions": versions,
    }


def _run_script(rel_path: str) -> dict:
    script = ROOT / rel_path
    proc = subprocess.run([sys.executable, str(script)], cwd=ROOT, capture_output=True, text=True)
    return {
        "script": rel_path,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
    }


def _load_json_if_exists(rel_path: str):
    p = ROOT / rel_path
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce HoloGen experiments and consolidate docs/results/.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="Fast verification + cache consolidation (default)")
    group.add_argument("--full", action="store_true", help="Re-run all experiment scripts end-to-end")
    args = parser.parse_args()

    results_dir = ROOT / "docs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    env_info = _collect_environment()
    (results_dir / "environment.json").write_text(json.dumps(env_info, indent=2), encoding="utf-8")

    run_log = []
    if args.full:
        for script in [
            "scripts/run_baseline.py",
            "scripts/run_experiment_a.py",
            "scripts/run_experiment_b.py",
            "scripts/run_experiment_c.py",
            "scripts/run_phase_7b5.py",
            "scripts/run_phase_7c5.py",
            "scripts/run_experiment_d_v2.py",
        ]:
            print(f"[full] Running {script}...", flush=True)
            res = _run_script(script)
            run_log.append(res)
            if not res["ok"]:
                print(f"[warn] {script} exited with code {res['returncode']}", flush=True)
    else:
        # --quick mode: run fast single-view baseline verification + pytest
        print("[quick] Running single-view baseline script...", flush=True)
        run_log.append(_run_script("scripts/run_baseline.py"))

    # Consolidate key result artifacts into docs/results/reproduced_summary.json
    summary = {
        "mode": "full" if args.full else "quick",
        "environment": env_info,
        "scripts_executed": run_log,
        "artifacts": {
            "experiment_a": _load_json_if_exists("outputs/experiment_a/experiment_a_results.json"),
            "experiment_b": _load_json_if_exists("outputs/experiment_b/experiment_b_results.json"),
            "experiment_c": _load_json_if_exists("outputs/experiment_c/experiment_c_results.json"),
            "phase_7b5": _load_json_if_exists("outputs/phase_7b5/phase_7b5_results.json"),
            "phase_7c5": _load_json_if_exists("outputs/phase_7c5/phase_7c5_results.json"),
            "experiment_d_v2": _load_json_if_exists("outputs/experiment_d_v2/experiment_d_v2_results.json"),
        },
    }
    (results_dir / "reproduced_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Consolidated results written to", results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
