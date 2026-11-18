"""Unit tests for Phase 4 mesh reconstruction, discontinuity edge cutting, hole repair, and 3D metrics."""

import math
import numpy as np
import pytest

from app.evaluation.geometry_metrics import compute_point_cloud_distance_metrics
from app.geometry.mesh import (
    MeshData,
    reconstruct_mesh_alpha_shape,
    reconstruct_mesh_ball_pivoting,
    reconstruct_mesh_depth_grid,
    reconstruct_mesh_poisson,
)
from app.geometry.point_cloud import backproject_depth_to_point_cloud
from app.geometry.refinement import (
    analyze_mesh_topology,
    clean_and_refine_mesh,
    fill_small_mesh_holes,
)
from app.vision.camera import estimate_intrinsics


def test_depth_grid_triangulation_and_discontinuity_edge_cutting() -> None:
    """Hand-computed check:
    For a 3x3 uniform depth grid (2x2 = 4 pixel quads), there are 4 * 2 = 8 triangles and 9 vertices.
    If the rightmost column jumps from Z=2.0 to Z=8.0 (relative jump (8-2)/2 = 3.0 > 0.08),
    all triangles spanning columns 1->2 must be cut, leaving only the 2 quads in columns 0..1 (4 triangles).
    """
    intr = estimate_intrinsics(width=3, height=3, focal_length_px=3.0)
    rgb = np.full((3, 3, 3), 200, dtype=np.uint8)

    uniform_depth = np.full((3, 3), 2.0, dtype=np.float32)
    mesh_uniform = reconstruct_mesh_depth_grid(
        uniform_depth, rgb, intr, relative_depth_jump_threshold=0.08
    )
    assert mesh_uniform.num_vertices == 9
    assert mesh_uniform.num_faces == 8

    step_depth = uniform_depth.copy()
    step_depth[:, 2] = 8.0
    mesh_step = reconstruct_mesh_depth_grid(
        step_depth, rgb, intr, relative_depth_jump_threshold=0.08
    )
    assert mesh_step.num_vertices == 9
    # Only the left 2 quads (columns 0 and 1) survive -> 2 * 2 = 4 triangles
    assert mesh_step.num_faces == 4


def test_hole_detection_and_repair_on_punctured_grid() -> None:
    """Verify a 5x5 grid with the center pixel masked out has 1 interior hole that gets filled."""
    intr = estimate_intrinsics(width=5, height=5, focal_length_px=5.0)
    depth = np.full((5, 5), 2.0, dtype=np.float32)
    rgb = np.full((5, 5, 3), 180, dtype=np.uint8)
    valid = np.ones((5, 5), dtype=bool)
    valid[2, 2] = False  # Puncture center vertex -> creates an interior hole loop

    mesh_punctured = reconstruct_mesh_depth_grid(depth, rgb, intr, valid_mask=valid)
    topo_before = analyze_mesh_topology(mesh_punctured)
    assert topo_before.num_interior_holes == 1

    repaired = fill_small_mesh_holes(mesh_punctured, max_hole_loop_edges=20)
    assert repaired.num_faces > mesh_punctured.num_faces
    topo_after = analyze_mesh_topology(repaired)
    assert topo_after.num_interior_holes == 0


def test_connected_component_filtering_and_decimation() -> None:
    """Verify small disconnected triangle islands are removed and quadric decimation reduces faces."""
    intr = estimate_intrinsics(width=20, height=20, focal_length_px=20.0)
    depth = np.full((20, 20), 2.0, dtype=np.float32)
    # Create a tiny 2x2 island at Z=10.0 in top-left corner separated by a discontinuity
    depth[0:2, 0:2] = 10.0
    rgb = np.full((20, 20, 3), 150, dtype=np.uint8)

    mesh_raw = reconstruct_mesh_depth_grid(depth, rgb, intr, relative_depth_jump_threshold=0.08)
    topo_raw = analyze_mesh_topology(mesh_raw)
    assert topo_raw.num_connected_components == 2

    cleaned = clean_and_refine_mesh(
        mesh_raw,
        min_component_faces=10,
        taubin_smoothing_iterations=2,
        target_faces=200,
    )
    topo_clean = analyze_mesh_topology(cleaned)
    assert topo_clean.num_connected_components == 1
    assert cleaned.num_faces <= 220


def test_open3d_comparison_meshers_execute_cleanly() -> None:
    """Verify Poisson, Ball Pivoting, and Alpha Shapes produce valid non-empty meshes on a curved surface."""
    intr = estimate_intrinsics(width=24, height=24, focal_length_px=24.0)
    u, v = np.meshgrid(np.linspace(-1, 1, 24), np.linspace(-1, 1, 24))
    depth = (3.0 + 0.3 * (u ** 2 + v ** 2)).astype(np.float32)
    rgb = np.full((24, 24, 3), 180, dtype=np.uint8)

    cloud = backproject_depth_to_point_cloud(depth, rgb, intr)

    m_poisson = reconstruct_mesh_poisson(cloud, octree_depth=5, density_quantile_trim=0.05)
    assert m_poisson.num_faces > 10

    m_bpa = reconstruct_mesh_ball_pivoting(cloud)
    assert m_bpa.num_faces > 10

    m_alpha = reconstruct_mesh_alpha_shape(cloud)
    assert m_alpha.num_faces > 10


def test_hand_computed_chamfer_and_hausdorff_metrics() -> None:
    """Hand-computed check for Chamfer, Hausdorff, Precision, and Recall:
    Let Q = [[0, 0, 0], [1, 0, 0]]
    Let P = [[0, 0, 0.1], [1, 0, 0.3]]
    dist(P[0], Q) = 0.1 (nearest Q[0])
    dist(P[1], Q) = 0.3 (nearest Q[1])
    dist(Q[0], P) = 0.1 (nearest P[0])
    dist(Q[1], P) = 0.3 (nearest P[1])
    Accuracy mean = (0.1 + 0.3) / 2 = 0.2
    Completeness mean = (0.1 + 0.3) / 2 = 0.2
    Chamfer L1 = 0.2
    RMS = sqrt((0.01 + 0.09) / 2) = sqrt(0.05) = 0.2236067977
    Hausdorff max = 0.3
    At tau = 0.2: Precision = 0.5, Recall = 0.5, F-score = 0.5.
    """
    Q = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    P = np.array([[0.0, 0.0, 0.1], [1.0, 0.0, 0.3]], dtype=np.float32)

    m = compute_point_cloud_distance_metrics(P, Q, threshold_tau=0.2)
    assert m["chamfer_l1"] == pytest.approx(0.2, abs=1e-6)
    assert m["chamfer_l2"] == pytest.approx(math.sqrt(0.05), abs=1e-6)
    assert m["hausdorff_max"] == pytest.approx(0.3, abs=1e-6)
    assert m["precision"] == pytest.approx(0.5, abs=1e-6)
    assert m["recall"] == pytest.approx(0.5, abs=1e-6)
    assert m["fscore"] == pytest.approx(0.5, abs=1e-6)
