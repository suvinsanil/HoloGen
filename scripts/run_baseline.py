"""CLI entry point for single-view 3D reconstruction (supports viewer and full-res presets)."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.core.logging import get_logger
from app.pipelines.single_view import SingleViewPipeline

logger = get_logger("scripts.run_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HoloGen Single-View Reconstruction CLI."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an input image. If omitted, runs on all images in datasets/samples/.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional custom run_id directory name under outputs/.",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=640,
        help="Maximum image dimension for aspect-preserving resize (default: 640).",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=60.0,
        help="Fallback horizontal field of view in degrees when EXIF focal length is absent (default: 60.0).",
    )
    parser.add_argument(
        "--focal-px",
        type=float,
        default=None,
        help="Explicit focal length override in pixels.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["viewer", "full"],
        default="viewer",
        help="Mesh quality preset: 'viewer' (~120k faces, GLB < 5MB) or 'full' (default: viewer).",
    )
    parser.add_argument(
        "--target-faces",
        type=int,
        default=120000,
        help="Target triangle face count for the 'viewer' preset (default: 120000).",
    )
    parser.add_argument(
        "--refinement",
        type=str,
        choices=["none", "bilateral", "guided", "guided_bilateral"],
        default="none",
        help="Depth refinement filter (default: none, per Experiment A findings).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device preference (default: auto).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    settings.device_preference = args.device

    pipeline = SingleViewPipeline(settings=settings)

    if args.image is not None:
        image_paths = [Path(args.image)]
    else:
        samples_dir = settings.datasets_dir / "samples"
        image_paths = sorted(
            [
                p
                for p in samples_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]
        )
        if not image_paths:
            logger.error("No sample images found in %s", samples_dir)
            return 1

    summaries = []
    for img_path in image_paths:
        run_id = args.run_id if (args.run_id and len(image_paths) == 1) else f"baseline_{img_path.stem}"
        result = pipeline.run(
            image_input=img_path,
            run_id=run_id,
            max_dimension=args.max_dim,
            fov_deg=args.fov,
            focal_length_px=args.focal_px,
            refinement_method=args.refinement,
            quality_preset=args.preset,
            viewer_target_faces=args.target_faces,
            export_both_presets=True,
        )
        exports = result.report_data["export_availability"]
        mesh_info = result.report_data["mesh"]
        summaries.append(
            {
                "run_id": result.run_id,
                "image": img_path.name,
                "processed_size": f"{result.preprocessing.width}x{result.preprocessing.height}",
                "device": result.depth_prediction.device,
                "refinement": args.refinement,
                "inference_ms": round(result.depth_prediction.inference_time_ms, 2),
                "num_points": result.point_cloud.num_points,
                "viewer_vertices": mesh_info["num_vertices"],
                "viewer_faces": mesh_info["num_faces"],
                "viewer_glb_mb": exports["glb"]["size_mb"],
                "full_vertices": mesh_info["full_res_vertices"],
                "full_faces": mesh_info["full_res_faces"],
                "full_glb_mb": exports["glb_full"]["size_mb"],
                "view_frustum_coverage_pct": result.report_data["estimated_coverage"][
                    "view_frustum_coverage_pct"
                ],
                "total_ms": round(result.total_time_ms, 2),
                "output_dir": str(result.output_dir),
            }
        )

    print(json.dumps({"runs": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
