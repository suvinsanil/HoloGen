"""Unit tests for Phase 5 projective UV mapping, coverage computation, and GLB/OBJ/PLY/STL export."""

import numpy as np
import pytest
import trimesh

from app.geometry.export import (
    export_mesh_glb,
    export_mesh_obj,
    export_mesh_ply,
    export_mesh_stl,
    validate_exported_glb,
)
from app.geometry.mesh import reconstruct_mesh_depth_grid
from app.geometry.texture import (
    apply_projective_texturing,
    compute_projective_uvs,
)
from app.vision.camera import estimate_intrinsics


def test_hand_computed_projective_uv_coordinates() -> None:
    """Hand-computed check:
    For W=101, H=51, fx=50.0, fy=50.0, cx=50.0, cy=25.0:
    1) Top-left pixel (u=0, v=0) at Z=2.0 -> X = (0-50)*2/50 = -2.0, Y = (0-25)*2/50 = -1.0
       UV: s = 0 / 100 = 0.0, t = 1 - (0 / 50) = 1.0
    2) Principal point (u=50, v=25) at Z=4.0 -> X = 0.0, Y = 0.0
       UV: s = 50 / 100 = 0.5, t = 1 - (25 / 50) = 0.5
    3) Bottom-right pixel (u=100, v=50) at Z=2.0 -> X = +2.0, Y = +1.0
       UV: s = 100 / 100 = 1.0, t = 1 - (50 / 50) = 0.0
    """
    intr = estimate_intrinsics(
        width=101,
        height=51,
        focal_length_px=50.0,
        principal_point=(50.0, 25.0),
    )
    verts = np.array(
        [
            [-2.0, -1.0, 2.0],
            [0.0, 0.0, 4.0],
            [2.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    uvs = compute_projective_uvs(verts, intr)
    expected_uvs = np.array(
        [
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(uvs, expected_uvs, atol=1e-6)


def test_projective_texturing_resolution_control_and_coverage() -> None:
    """Verify texture resizing and 100% frustum UV coverage on a full depth-grid mesh."""
    intr = estimate_intrinsics(width=32, height=24, focal_length_px=30.0)
    depth = np.full((24, 32), 2.5, dtype=np.float32)
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200
    rgb[:, :, 1] = 100
    rgb[:, :, 2] = 50

    mesh = reconstruct_mesh_depth_grid(depth, rgb, intr)
    tex_data = apply_projective_texturing(
        mesh=mesh,
        rgb_uint8=rgb,
        intrinsics=intr,
        texture_max_dimension=16,
    )
    assert tex_data.original_texture_size == (32, 24)
    assert tex_data.final_texture_size == (16, 12)
    assert tex_data.frustum_coverage_pct == pytest.approx(100.0, abs=0.5)
    assert tex_data.mesh.vertex_uvs is not None
    assert tex_data.mesh.vertex_uvs.shape == (mesh.num_vertices, 2)


def test_glb_obj_ply_stl_exports_and_independent_trimesh_validation(tmp_path) -> None:
    """Verify GLB, OBJ, PLY, and STL exports succeed and GLB validates in trimesh."""
    intr = estimate_intrinsics(width=16, height=16, focal_length_px=16.0)
    depth = np.full((16, 16), 3.0, dtype=np.float32)
    rgb = np.full((16, 16, 3), 180, dtype=np.uint8)

    mesh = reconstruct_mesh_depth_grid(depth, rgb, intr)
    tex_data = apply_projective_texturing(mesh, rgb, intr)

    glb_p = export_mesh_glb(tex_data, tmp_path / "scene.glb")
    obj_p = export_mesh_obj(tex_data, tmp_path / "scene.obj")
    ply_p = export_mesh_ply(tex_data.mesh, tmp_path / "mesh.ply")
    stl_p = export_mesh_stl(tex_data.mesh, tmp_path / "mesh.stl")

    assert glb_p.exists() and glb_p.stat().st_size > 100
    assert obj_p.exists() and obj_p.stat().st_size > 100
    assert ply_p.exists() and ply_p.stat().st_size > 100
    assert stl_p.exists() and stl_p.stat().st_size > 100

    val = validate_exported_glb(glb_p)
    assert val["valid_glb_header"] is True
    assert val["loaded_vertices"] == 256
    assert val["loaded_faces"] == 450
    assert val["has_uv_coordinates"] is True
    assert val["has_embedded_texture"] is True

    # Also verify STL loads cleanly in trimesh
    loaded_stl = trimesh.load(str(stl_p), file_type="stl")
    assert len(loaded_stl.faces) == 450
