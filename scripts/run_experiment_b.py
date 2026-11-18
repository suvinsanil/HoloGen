"""Experiment B Runner: Surface Mesh Reconstruction Comparison on Identical Inputs.

Compares 6 surface meshing methods on identical preprocessed depth & point cloud input:
1. depth_grid_baseline (Depth-grid triangulation with discontinuity edge cutting at 8% jump)
2. depth_grid_cleaned_repaired (Depth-grid + island removal + Taubin smoothing + hole repair)
3. poisson_depth8_raw (Screened Poisson octree depth 8 without density trimming — shows hidden bubble invention)
4. poisson_depth8_trimmed (Screened Poisson octree depth 8 with 8% low-density quantile trimming)
5. ball_pivoting (Ball Pivoting Algorithm with multi-scale radii [1.5, 2.5, 4.0] * d_nn)
6. alpha_shape (3D Alpha Shapes with alpha = 3.2 * d_nn)
"""

import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.core.logging import get_logger
from app.evaluation.geometry_metrics import evaluate_mesh_reconstruction
from app.geometry.filtering import (
    normal_grazing_angle_filter,
    statistical_outlier_removal,
)
from app.geometry.mesh import (
    MeshData,
    reconstruct_mesh_alpha_shape,
    reconstruct_mesh_ball_pivoting,
    reconstruct_mesh_depth_grid,
    reconstruct_mesh_poisson,
)
from app.geometry.normals import estimate_and_orient_normals
from app.geometry.point_cloud import backproject_depth_to_point_cloud
from app.geometry.refinement import clean_and_refine_mesh
from app.vision.camera import estimate_intrinsics
from app.vision.depth_estimation import create_depth_model
from app.vision.depth_refinement import refine_depth_map
from app.vision.preprocessing import preprocess_image

logger = get_logger("scripts.run_experiment_b")


def save_mesh_ply(mesh: MeshData, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d_mesh = mesh.to_open3d()
    o3d.io.write_triangle_mesh(str(output_path), o3d_mesh, write_ascii=False)
    return output_path


def render_mesh_preview(mesh: MeshData, output_path: Path, canvas_size: int = 480) -> None:
    """Render side-by-side shaded normal-lit front view and wireframe/side view of a mesh."""
    canvas = np.full((canvas_size, canvas_size * 2, 3), 24, dtype=np.uint8)
    if mesh.num_faces == 0 or mesh.num_vertices == 0:
        cv2.imwrite(str(output_path), canvas)
        return

    verts = mesh.vertices
    norms = (
        mesh.vertex_normals
        if mesh.vertex_normals is not None
        else np.zeros_like(verts)
    )
    cols = (
        mesh.vertex_colors
        if mesh.vertex_colors is not None
        else np.full_like(verts, 0.75)
    )

    # Lambertian directional shading combined with vertex color
    light_dir = np.array([0.2, -0.3, -0.93], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    lambert = np.clip(np.abs(np.sum(norms * light_dir, axis=1, keepdims=True)), 0.25, 1.0)
    shaded_bgr = np.clip((cols[:, ::-1] * (0.4 + 0.6 * lambert)) * 255.0, 0, 255).astype(np.uint8)

    min_b = np.percentile(verts, 1.0, axis=0)
    max_b = np.percentile(verts, 99.0, axis=0)
    span = np.maximum(max_b - min_b, 1e-4)
    margin = 24
    usable = canvas_size - 2 * margin

    # Sort triangles back-to-front by centroid Z
    tri_centroids = np.mean(verts[mesh.faces], axis=1)
    tri_order = np.argsort(-tri_centroids[:, 2])

    # Project vertices to Front (XY)
    u_f = margin + ((verts[:, 0] - min_b[0]) / span[0] * usable).astype(np.int32)
    v_f = margin + ((verts[:, 1] - min_b[1]) / span[1] * usable).astype(np.int32)
    pts_2d_f = np.stack([u_f, v_f], axis=-1)

    # Draw up to 40,000 sorted triangles for crisp preview
    step = max(1, len(tri_order) // 40000)
    for tid in tri_order[::step]:
        f = mesh.faces[tid]
        poly = pts_2d_f[f]
        color = tuple(int(c) for c in shaded_bgr[f[0]])
        cv2.fillConvexPoly(canvas[:, :canvas_size], poly, color, lineType=cv2.LINE_AA)

    # Panel 2: Side view (ZY) of mesh vertices — clearly shows Poisson's hallucinated back bubble!
    u_s = canvas_size + margin + ((verts[:, 2] - min_b[2]) / span[2] * usable).astype(np.int32)
    v_s = margin + ((verts[:, 1] - min_b[1]) / span[1] * usable).astype(np.int32)
    valid_s = (
        (u_s >= canvas_size)
        & (u_s < canvas_size * 2)
        & (v_s >= 0)
        & (v_s < canvas_size)
    )
    canvas[v_s[valid_s], u_s[valid_s]] = shaded_bgr[valid_s]

    cv2.putText(
        canvas,
        f"{mesh.method} | F={mesh.num_faces}",
        (14, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Side Profile (ZY)",
        (canvas_size + 14, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


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

    sample_path = settings.datasets_dir / "samples" / "sample_02_aloe_indoor.jpg"
    prep = preprocess_image(sample_path, max_dimension=480)
    intrinsics = estimate_intrinsics(prep.width, prep.height, exif_metadata=prep.exif_metadata)

    model = create_depth_model("depth_anything_v2_small", device=device)
    model.load()
    pred = model.predict(prep.rgb_uint8)
    refined = refine_depth_map(pred.depth_map, prep.rgb_uint8, method="bilateral")

    raw_cloud = backproject_depth_to_point_cloud(
        depth_map=refined.refined_depth,
        rgb_image=prep.rgb_uint8,
        intrinsics=intrinsics,
    )
    cloud_normals = estimate_and_orient_normals(raw_cloud, max_nn=30)
    sor_cloud = statistical_outlier_removal(cloud_normals, nb_neighbors=20, std_ratio=2.0).cloud
    clean_cloud = normal_grazing_angle_filter(sor_cloud, max_grazing_angle_deg=80.0).cloud

    exp_dir = settings.outputs_dir / "experiment_b"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Construct the 6 mesh variants on identical input
    meshes: list[MeshData] = []

    # 1. Depth-grid triangulation (baseline with discontinuity edge cutting)
    m_dg_raw = reconstruct_mesh_depth_grid(
        depth_map=refined.refined_depth,
        rgb_image=prep.rgb_uint8,
        intrinsics=intrinsics,
        relative_depth_jump_threshold=0.08,
        stride=1,
    )
    m_dg_raw.method = "1_depth_grid_discontinuity_cut"
    meshes.append(m_dg_raw)

    # 2. Depth-grid cleaned + small hole repair + Taubin smoothing
    m_dg_clean = clean_and_refine_mesh(
        m_dg_raw,
        min_component_faces=50,
        repair_holes=True,
        max_hole_loop_edges=36,
        taubin_smoothing_iterations=5,
    )
    m_dg_clean.method = "2_depth_grid_cleaned_repaired"
    meshes.append(m_dg_clean)

    # 3. Screened Poisson (octree depth 8, raw untrimmed — shows bubble invention on 2.5D clouds)
    m_poisson_raw = reconstruct_mesh_poisson(
        clean_cloud, octree_depth=8, density_quantile_trim=0.0
    )
    m_poisson_raw.method = "3_poisson_depth8_untrimmed"
    meshes.append(m_poisson_raw)

    # 4. Screened Poisson (octree depth 8, 8% low-density quantile trimmed)
    m_poisson_trim = reconstruct_mesh_poisson(
        clean_cloud, octree_depth=8, density_quantile_trim=0.08
    )
    m_poisson_trim.method = "4_poisson_depth8_trimmed_q08"
    meshes.append(m_poisson_trim)

    # 5. Ball Pivoting Algorithm (BPA)
    m_bpa = reconstruct_mesh_ball_pivoting(
        clean_cloud, radius_multipliers=[1.5, 2.5, 4.0], max_points=55000
    )
    m_bpa.method = "5_ball_pivoting"
    meshes.append(m_bpa)

    # 6. 3D Alpha Shapes
    m_alpha = reconstruct_mesh_alpha_shape(clean_cloud, alpha=None, max_points=35000)
    m_alpha.method = "6_alpha_shape"
    meshes.append(m_alpha)

    results = []
    for m in meshes:
        ev = evaluate_mesh_reconstruction(m, reference_cloud=clean_cloud, threshold_tau=0.04)
        ply_p = save_mesh_ply(m, exp_dir / f"{m.method}.ply")
        prev_p = exp_dir / f"{m.method}_preview.png"
        render_mesh_preview(m, prev_p)

        row = {
            **ev.to_dict(),
            "ply_file": ply_p.name,
            "preview_png": prev_p.name,
        }
        results.append(row)

    payload = {
        "experiment": "Experiment B: Surface Mesh Reconstruction Methods Comparison",
        "git_commit": commit_hash,
        "input_image": sample_path.name,
        "processed_resolution": f"{prep.width}x{prep.height}",
        "reference_clean_cloud_points": clean_cloud.num_points,
        "completeness_threshold_tau": 0.04,
        "methods": results,
    }

    out_json = exp_dir / "experiment_b_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved Experiment B results to %s", out_json)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
