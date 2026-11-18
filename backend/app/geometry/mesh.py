"""3D surface mesh reconstruction: Depth-Grid triangulation (baseline), Poisson, Ball Pivoting, and Alpha Shapes."""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import open3d as o3d

from app.core.exceptions import GeometryProcessingError
from app.geometry.normals import estimate_and_orient_normals
from app.geometry.point_cloud import PointCloudData
from app.vision.camera import CameraIntrinsics


@dataclass
class MeshData:
    """Container for a 3D triangle mesh with vertex colors, normals, and optional UV coordinates."""

    vertices: np.ndarray  # float32 (V, 3)
    faces: np.ndarray  # int32 (F, 3) triangle vertex indices
    vertex_colors: Optional[np.ndarray] = None  # float32 (V, 3) in [0, 1]
    vertex_normals: Optional[np.ndarray] = None  # float32 (V, 3) unit normals
    vertex_uvs: Optional[np.ndarray] = None  # float32 (V, 2) in [0, 1] (u_tex, v_tex)
    method: str = "depth_grid"
    generation_time_ms: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])

    def to_open3d(self) -> o3d.geometry.TriangleMesh:
        """Convert to an Open3D TriangleMesh instance."""
        mesh = o3d.geometry.TriangleMesh()
        if self.num_vertices > 0:
            mesh.vertices = o3d.utility.Vector3dVector(self.vertices.astype(np.float64))
        if self.num_faces > 0:
            mesh.triangles = o3d.utility.Vector3iVector(self.faces.astype(np.int32))
        if self.vertex_colors is not None and len(self.vertex_colors) == self.num_vertices:
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.clip(self.vertex_colors, 0.0, 1.0).astype(np.float64)
            )
        if self.vertex_normals is not None and len(self.vertex_normals) == self.num_vertices:
            mesh.vertex_normals = o3d.utility.Vector3dVector(
                self.vertex_normals.astype(np.float64)
            )
        return mesh

    @classmethod
    def from_open3d(
        cls,
        o3d_mesh: o3d.geometry.TriangleMesh,
        method: str,
        generation_time_ms: float,
        parameters: Optional[Dict[str, Any]] = None,
        vertex_uvs: Optional[np.ndarray] = None,
    ) -> "MeshData":
        """Construct MeshData from an Open3D TriangleMesh."""
        if not o3d_mesh.has_vertex_normals() and len(o3d_mesh.triangles) > 0:
            o3d_mesh.compute_vertex_normals()
        verts = np.asarray(o3d_mesh.vertices, dtype=np.float32)
        tris = np.asarray(o3d_mesh.triangles, dtype=np.int32)
        cols = (
            np.asarray(o3d_mesh.vertex_colors, dtype=np.float32)
            if o3d_mesh.has_vertex_colors()
            else None
        )
        norms = (
            np.asarray(o3d_mesh.vertex_normals, dtype=np.float32)
            if o3d_mesh.has_vertex_normals()
            else None
        )
        return cls(
            vertices=verts,
            faces=tris,
            vertex_colors=cols,
            vertex_normals=norms,
            vertex_uvs=vertex_uvs,
            method=method,
            generation_time_ms=generation_time_ms,
            parameters=parameters or {},
        )

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "num_vertices": self.num_vertices,
            "num_faces": self.num_faces,
            "has_vertex_colors": self.vertex_colors is not None,
            "has_vertex_normals": self.vertex_normals is not None,
            "has_vertex_uvs": self.vertex_uvs is not None,
            "generation_time_ms": round(self.generation_time_ms, 3),
            "parameters": self.parameters,
        }


def reconstruct_mesh_depth_grid(
    depth_map: np.ndarray,
    rgb_image: np.ndarray,
    intrinsics: CameraIntrinsics,
    relative_depth_jump_threshold: float = 0.08,
    stride: int = 1,
    valid_mask: Optional[np.ndarray] = None,
) -> MeshData:
    """Reconstruct a triangle mesh directly from a 2D depth grid with discontinuity edge cutting.

    For every 2x2 pixel quad:
        p00 = (u, v),     p10 = (u+s, v)
        p01 = (u, v+s),   p11 = (u+s, v+s)
    Candidate triangles are (p00, p01, p10) and (p10, p01, p11).
    A triangle is rejected if any of its 3 vertices are invalid OR if the relative depth jump
    across any of its 3 edges exceeds `relative_depth_jump_threshold`:
        |Z_a - Z_b| / min(Z_a, Z_b) > relative_depth_jump_threshold.
    """
    t0 = time.perf_counter()
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim != 2:
        raise GeometryProcessingError(f"Expected 2D depth_map, got shape {depth.shape}")
    h, w = depth.shape
    if intrinsics.width != w or intrinsics.height != h:
        raise GeometryProcessingError(
            f"Intrinsics ({intrinsics.width}x{intrinsics.height}) != depth ({w}x{h})"
        )
    if stride < 1:
        raise GeometryProcessingError(f"stride must be >= 1, got {stride}")

    rgb = np.asarray(rgb_image)
    if rgb.dtype == np.uint8:
        colors_norm = rgb.astype(np.float32) / 255.0
    else:
        colors_norm = np.clip(rgb.astype(np.float32), 0.0, 1.0)

    # Subsample grid by stride
    d_sub = depth[::stride, ::stride]
    c_sub = colors_norm[::stride, ::stride]
    gh, gw = d_sub.shape
    if gh < 2 or gw < 2:
        raise GeometryProcessingError(f"Grid dimensions too small for meshing: {gw}x{gh}")

    if valid_mask is not None:
        m_sub = valid_mask[::stride, ::stride].astype(bool) & np.isfinite(d_sub) & (d_sub > 1e-4)
    else:
        m_sub = np.isfinite(d_sub) & (d_sub > 1e-4)

    u_coords = np.arange(0, w, stride, dtype=np.float32)[:gw]
    v_coords = np.arange(0, h, stride, dtype=np.float32)[:gh]
    u_grid, v_grid = np.meshgrid(u_coords, v_coords, indexing="xy")

    x_grid = (u_grid - float(intrinsics.cx)) * d_sub / float(intrinsics.fx)
    y_grid = (v_grid - float(intrinsics.cy)) * d_sub / float(intrinsics.fy)
    xyz_grid = np.stack([x_grid, y_grid, d_sub], axis=-1).astype(np.float32)

    # UV coordinates in [0, 1] (u_tex = u / (W - 1), v_tex = 1.0 - v / (H - 1) for glTF convention)
    u_tex = u_grid / max(float(w - 1), 1.0)
    v_tex = 1.0 - (v_grid / max(float(h - 1), 1.0))
    uv_grid = np.stack([u_tex, v_tex], axis=-1).astype(np.float32)

    # Build vertex index grid (-1 for invalid pixels)
    vertex_idx_grid = np.full((gh, gw), -1, dtype=np.int32)
    valid_count = int(np.sum(m_sub))
    if valid_count < 3:
        raise GeometryProcessingError("Fewer than 3 valid depth pixels for depth-grid meshing.")

    vertex_idx_grid[m_sub] = np.arange(valid_count, dtype=np.int32)
    vertices = xyz_grid[m_sub]
    vertex_colors = c_sub[m_sub]
    vertex_uvs = uv_grid[m_sub]

    # Quad corner indices
    idx00 = vertex_idx_grid[:-1, :-1]
    idx10 = vertex_idx_grid[:-1, 1:]
    idx01 = vertex_idx_grid[1:, :-1]
    idx11 = vertex_idx_grid[1:, 1:]

    z00 = d_sub[:-1, :-1]
    z10 = d_sub[:-1, 1:]
    z01 = d_sub[1:, :-1]
    z11 = d_sub[1:, 1:]

    thresh = float(relative_depth_jump_threshold)

    def _edge_ok(za: np.ndarray, zb: np.ndarray) -> np.ndarray:
        denom = np.maximum(np.minimum(za, zb), 1e-4)
        return (np.abs(za - zb) / denom) <= thresh

    e_horiz_top = _edge_ok(z00, z10)
    e_horiz_bot = _edge_ok(z01, z11)
    e_vert_left = _edge_ok(z00, z01)
    e_vert_right = _edge_ok(z10, z11)
    e_diag_01_10 = _edge_ok(z01, z10)

    # Triangle 1: (p00, p01, p10)
    valid_t1 = (
        (idx00 >= 0)
        & (idx01 >= 0)
        & (idx10 >= 0)
        & e_vert_left
        & e_horiz_top
        & e_diag_01_10
    )
    # Triangle 2: (p10, p01, p11)
    valid_t2 = (
        (idx10 >= 0)
        & (idx01 >= 0)
        & (idx11 >= 0)
        & e_diag_01_10
        & e_horiz_bot
        & e_vert_right
    )

    tris_1 = np.stack([idx00[valid_t1], idx01[valid_t1], idx10[valid_t1]], axis=-1)
    tris_2 = np.stack([idx10[valid_t2], idx01[valid_t2], idx11[valid_t2]], axis=-1)

    faces = np.vstack([tris_1, tris_2]).astype(np.int32)
    if faces.shape[0] == 0:
        raise GeometryProcessingError(
            "Depth-grid triangulation produced 0 faces; check relative_depth_jump_threshold."
        )

    # Compute vertex normals via Open3D
    temp_mesh = o3d.geometry.TriangleMesh()
    temp_mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    temp_mesh.triangles = o3d.utility.Vector3iVector(faces)
    temp_mesh.compute_vertex_normals()
    vertex_normals = np.asarray(temp_mesh.vertex_normals, dtype=np.float32)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return MeshData(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        vertex_normals=vertex_normals,
        vertex_uvs=vertex_uvs,
        method="depth_grid",
        generation_time_ms=elapsed_ms,
        parameters={
            "relative_depth_jump_threshold": relative_depth_jump_threshold,
            "stride": stride,
        },
    )


def reconstruct_mesh_poisson(
    cloud: PointCloudData,
    octree_depth: int = 8,
    density_quantile_trim: float = 0.0,
) -> MeshData:
    """Reconstruct a watertight surface mesh using Screened Poisson Surface Reconstruction.

    Note: On single-view 2.5D point clouds, Poisson reconstruction extrapolates smooth bubble
    surfaces across unobserved back-occluded regions unless low-support vertices are trimmed via
    `density_quantile_trim > 0`.
    """
    t0 = time.perf_counter()
    if octree_depth < 4 or octree_depth > 11:
        raise GeometryProcessingError(f"octree_depth must be in [4, 11], got {octree_depth}")

    c_norm = cloud if cloud.normals is not None else estimate_and_orient_normals(cloud)
    pcd = c_norm.to_open3d()

    o3d_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=int(octree_depth),
        scale=1.1,
        linear_fit=False,
    )

    if density_quantile_trim > 0.0:
        dens_arr = np.asarray(densities, dtype=np.float64)
        cutoff = float(np.quantile(dens_arr, float(density_quantile_trim)))
        o3d_mesh.remove_vertices_by_mask(dens_arr < cutoff)
        o3d_mesh.remove_unreferenced_vertices()

    o3d_mesh.compute_vertex_normals()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    method_name = (
        f"poisson_trimmed_q{int(density_quantile_trim * 100)}"
        if density_quantile_trim > 0.0
        else "poisson_raw"
    )
    return MeshData.from_open3d(
        o3d_mesh,
        method=method_name,
        generation_time_ms=elapsed_ms,
        parameters={
            "octree_depth": octree_depth,
            "density_quantile_trim": density_quantile_trim,
        },
    )


def reconstruct_mesh_ball_pivoting(
    cloud: PointCloudData,
    radius_multipliers: Optional[List[float]] = None,
    max_points: int = 60000,
) -> MeshData:
    """Reconstruct a triangle mesh using the Ball Pivoting Algorithm (BPA)."""
    t0 = time.perf_counter()
    multipliers = radius_multipliers or [1.5, 2.5, 4.0]
    c_norm = cloud if cloud.normals is not None else estimate_and_orient_normals(cloud)
    pcd = c_norm.to_open3d()

    if len(pcd.points) > max_points:
        diag = float(np.linalg.norm(c_norm.extent))
        voxel_sz = max(diag / 160.0, 0.015)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_sz)

    dists = np.asarray(pcd.compute_nearest_neighbor_distance(), dtype=np.float64)
    mean_nn = max(float(np.mean(dists)), 1e-3)
    radii = [mean_nn * float(m) for m in multipliers]

    o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii),
    )
    o3d_mesh.compute_vertex_normals()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return MeshData.from_open3d(
        o3d_mesh,
        method="ball_pivoting",
        generation_time_ms=elapsed_ms,
        parameters={
            "radii": [round(r, 5) for r in radii],
            "radius_multipliers": multipliers,
        },
    )


def reconstruct_mesh_alpha_shape(
    cloud: PointCloudData,
    alpha: Optional[float] = None,
    max_points: int = 35000,
) -> MeshData:
    """Reconstruct a triangle mesh using 3D Alpha Shapes (Edelsbrunner & Mucke, 1994)."""
    t0 = time.perf_counter()
    c_norm = cloud if cloud.normals is not None else estimate_and_orient_normals(cloud)
    pcd = c_norm.to_open3d()

    if len(pcd.points) > max_points:
        diag = float(np.linalg.norm(c_norm.extent))
        voxel_sz = max(diag / 120.0, 0.025)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_sz)

    dists = np.asarray(pcd.compute_nearest_neighbor_distance(), dtype=np.float64)
    mean_nn = max(float(np.mean(dists)), 1e-3)
    if alpha is None:
        diag = float(np.linalg.norm(c_norm.extent))
        alpha_val = max(mean_nn * 4.5, diag * 0.08)
    else:
        alpha_val = float(alpha)

    # Apply tiny deterministic general-position perturbation (1e-3 * mean_nn) so regular pixel grids
    # do not produce cospherical/coplanar degenerate tetrahedra in Qhull Delaunay triangulation
    rng = np.random.default_rng(42)
    pts_arr = np.asarray(pcd.points, dtype=np.float64)
    pts_jittered = pts_arr + rng.normal(0.0, mean_nn * 2e-3, size=pts_arr.shape)
    pcd_alpha = o3d.geometry.PointCloud()
    pcd_alpha.points = o3d.utility.Vector3dVector(pts_jittered)

    o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd_alpha,
        alpha=alpha_val,
    )
    # Alpha shape does not automatically copy vertex colors from pcd in Open3D; transfer via KDTree
    if len(o3d_mesh.vertices) > 0 and pcd.has_colors():
        from scipy.spatial import cKDTree

        src_pts = np.asarray(pcd.points, dtype=np.float32)
        src_cols = np.asarray(pcd.colors, dtype=np.float32)
        mesh_verts = np.asarray(o3d_mesh.vertices, dtype=np.float32)
        tree = cKDTree(src_pts)
        _, nn_idx = tree.query(mesh_verts, k=1)
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(src_cols[nn_idx].astype(np.float64))

    o3d_mesh.compute_vertex_normals()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return MeshData.from_open3d(
        o3d_mesh,
        method="alpha_shape",
        generation_time_ms=elapsed_ms,
        parameters={"alpha": round(alpha_val, 5)},
    )
