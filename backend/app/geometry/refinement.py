"""Mesh cleanup, connected-component filtering, hole detection & repair, smoothing, and decimation."""

from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import open3d as o3d

from app.core.exceptions import GeometryProcessingError
from app.geometry.mesh import MeshData


@dataclass
class MeshTopologyReport:
    """Topological statistics for a triangle mesh (components, boundary edges, and holes)."""

    num_vertices: int
    num_faces: int
    num_connected_components: int
    num_boundary_edges: int
    num_boundary_loops: int
    num_interior_holes: int  # Boundary loops excluding the outer perimeter boundary of each sheet
    largest_loop_edges: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_vertices": self.num_vertices,
            "num_faces": self.num_faces,
            "num_connected_components": self.num_connected_components,
            "num_boundary_edges": self.num_boundary_edges,
            "num_boundary_loops": self.num_boundary_loops,
            "num_interior_holes": self.num_interior_holes,
            "largest_loop_edges": self.largest_loop_edges,
        }


def extract_boundary_loops(faces: np.ndarray) -> Tuple[int, List[List[int]]]:
    """Identify undirected edges shared by exactly 1 triangle and group them into closed cycle loops."""
    if faces.shape[0] == 0:
        return 0, []

    e0 = np.sort(faces[:, [0, 1]], axis=1)
    e1 = np.sort(faces[:, [1, 2]], axis=1)
    e2 = np.sort(faces[:, [2, 0]], axis=1)
    all_edges = np.vstack([e0, e1, e2])

    # Pack edge pairs into 64-bit integer keys for fast vectorized unique counting
    keys = (all_edges[:, 0].astype(np.int64) << 32) | all_edges[:, 1].astype(np.int64)
    uniq_keys, counts = np.unique(keys, return_counts=True)
    boundary_keys = uniq_keys[counts == 1]
    num_boundary_edges = int(boundary_keys.size)
    if num_boundary_edges == 0:
        return 0, []

    u_verts = (boundary_keys >> 32).astype(np.int32)
    v_verts = (boundary_keys & 0xFFFFFFFF).astype(np.int32)

    G = nx.Graph()
    G.add_edges_from(zip(u_verts.tolist(), v_verts.tolist()))
    loops: List[List[int]] = []
    for comp in nx.connected_components(G):
        if len(comp) >= 3:
            subg = G.subgraph(comp)
            # Order cycle vertices sequentially via DFS preorder
            ordered = list(nx.dfs_preorder_nodes(subg))
            loops.append(ordered)

    loops.sort(key=len, reverse=True)
    return num_boundary_edges, loops


def analyze_mesh_topology(mesh: MeshData) -> MeshTopologyReport:
    """Analyze connected components, boundary edges, and interior holes of a mesh."""
    if mesh.num_faces == 0:
        return MeshTopologyReport(
            num_vertices=mesh.num_vertices,
            num_faces=0,
            num_connected_components=0,
            num_boundary_edges=0,
            num_boundary_loops=0,
            num_interior_holes=0,
            largest_loop_edges=0,
        )

    o3d_mesh = mesh.to_open3d()
    _, cluster_n_tris, _ = o3d_mesh.cluster_connected_triangles()
    num_components = len(cluster_n_tris)

    num_b_edges, loops = extract_boundary_loops(mesh.faces)
    num_loops = len(loops)
    # For open surface sheets, the largest boundary loop per major sheet is the outer frustum border;
    # remaining smaller loops are interior surface holes.
    num_interior_holes = max(0, num_loops - 1) if num_loops > 0 else 0
    largest_loop = len(loops[0]) if loops else 0

    return MeshTopologyReport(
        num_vertices=mesh.num_vertices,
        num_faces=mesh.num_faces,
        num_connected_components=num_components,
        num_boundary_edges=num_b_edges,
        num_boundary_loops=num_loops,
        num_interior_holes=num_interior_holes,
        largest_loop_edges=largest_loop,
    )


def fill_small_mesh_holes(
    mesh: MeshData,
    max_hole_loop_edges: int = 40,
    max_relative_depth_span: float = 0.15,
    max_absolute_depth_span: float = 0.40,
) -> MeshData:
    """Repair small interior boundary loops (length <= `max_hole_loop_edges`) via centroid fan triangulation.

    Guards against bridging depth layers: loops spanning across depth discontinuities
    or silhouette edges (relative Z span > `max_relative_depth_span` or absolute delta Z > `max_absolute_depth_span`)
    are skipped to prevent artificial curtain geometry connecting foreground to background.
    """
    t0 = time.perf_counter()
    _, loops = extract_boundary_loops(mesh.faces)
    if len(loops) <= 1:
        return mesh

    # Group boundary loops by connected component.
    # For open surface sheets, the largest boundary loop of each component is its outer perimeter;
    # only smaller loops within that same component are true interior surface holes.
    o3d_m = mesh.to_open3d()
    tri_clusters, _, _ = o3d_m.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters, dtype=np.int32)
    vert_comp = np.full(mesh.num_vertices, -1, dtype=np.int32)
    for c_id in np.unique(tri_clusters):
        c_verts = np.unique(mesh.faces[tri_clusters == c_id])
        vert_comp[c_verts] = c_id

    comp_loops = defaultdict(list)
    for lp in loops:
        c_id = int(vert_comp[lp[0]])
        comp_loops[c_id].append(lp)

    candidate_holes = []
    for c_id, c_lps in comp_loops.items():
        if len(c_lps) > 1:
            c_lps.sort(key=len, reverse=True)
            for lp in c_lps[1:]:
                if 3 <= len(lp) <= max_hole_loop_edges:
                    candidate_holes.append(lp)

    if not candidate_holes:
        return mesh

    # Filter candidates to only genuine planar surface holes
    valid_holes = []
    for lp in candidate_holes:
        lp_arr = np.asarray(lp, dtype=np.int32)
        v_z = mesh.vertices[lp_arr, 2]
        z_min = float(np.min(v_z))
        z_max = float(np.max(v_z))
        z_span = z_max - z_min
        rel_span = z_span / max(z_min, 1e-4)
        if rel_span > max_relative_depth_span or z_span > max_absolute_depth_span:
            continue
        valid_holes.append(lp)

    if not valid_holes:
        return mesh

    new_verts = [mesh.vertices]
    new_cols = [mesh.vertex_colors] if mesh.vertex_colors is not None else None
    new_uvs = [mesh.vertex_uvs] if mesh.vertex_uvs is not None else None
    new_faces = [mesh.faces]

    next_v_idx = mesh.num_vertices
    for loop in valid_holes:
        loop_arr = np.asarray(loop, dtype=np.int32)
        centroid_v = np.mean(mesh.vertices[loop_arr], axis=0, keepdims=True).astype(np.float32)
        new_verts.append(centroid_v)

        if new_cols is not None and mesh.vertex_colors is not None:
            centroid_c = np.mean(mesh.vertex_colors[loop_arr], axis=0, keepdims=True).astype(
                np.float32
            )
            new_cols.append(centroid_c)

        if new_uvs is not None and mesh.vertex_uvs is not None:
            centroid_uv = np.mean(mesh.vertex_uvs[loop_arr], axis=0, keepdims=True).astype(
                np.float32
            )
            new_uvs.append(centroid_uv)

        k = len(loop)
        fan_tris = np.zeros((k, 3), dtype=np.int32)
        for i in range(k):
            fan_tris[i, 0] = loop[i]
            fan_tris[i, 1] = loop[(i + 1) % k]
            fan_tris[i, 2] = next_v_idx
        new_faces.append(fan_tris)
        next_v_idx += 1

    verts_combined = np.vstack(new_verts)
    faces_combined = np.vstack(new_faces)
    cols_combined = np.vstack(new_cols) if new_cols is not None else None
    uvs_combined = np.vstack(new_uvs) if new_uvs is not None else None

    o3d_m = o3d.geometry.TriangleMesh()
    o3d_m.vertices = o3d.utility.Vector3dVector(verts_combined.astype(np.float64))
    o3d_m.triangles = o3d.utility.Vector3iVector(faces_combined)
    o3d_m.compute_vertex_normals()
    norms_combined = np.asarray(o3d_m.vertex_normals, dtype=np.float32)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return MeshData(
        vertices=verts_combined,
        faces=faces_combined,
        vertex_colors=cols_combined,
        vertex_normals=norms_combined,
        vertex_uvs=uvs_combined,
        method=mesh.method,
        generation_time_ms=mesh.generation_time_ms + elapsed_ms,
        parameters={**mesh.parameters, "filled_holes_count": len(valid_holes)},
    )


def clean_and_refine_mesh(
    mesh: MeshData,
    min_component_faces: int = 60,
    keep_only_largest_component: bool = False,
    repair_holes: bool = True,
    max_hole_loop_edges: int = 40,
    max_relative_depth_span: float = 0.15,
    max_absolute_depth_span: float = 0.40,
    taubin_smoothing_iterations: int = 5,
    target_faces: Optional[int] = None,
) -> MeshData:
    """Clean, filter disconnected islands, repair small holes, smooth, and optionally decimate a mesh."""
    t0 = time.perf_counter()
    if mesh.num_faces == 0:
        raise GeometryProcessingError("Cannot clean a mesh with 0 faces.")

    orig_verts = np.asarray(mesh.vertices, dtype=np.float32)
    orig_uvs = (
        np.asarray(mesh.vertex_uvs, dtype=np.float32)
        if mesh.vertex_uvs is not None and len(mesh.vertex_uvs) == mesh.num_vertices
        else None
    )

    o3d_mesh = mesh.to_open3d()
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_non_manifold_edges()
    o3d_mesh.remove_unreferenced_vertices()

    # Connected-component filtering
    triangle_clusters, cluster_n_triangles, _ = o3d_mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(triangle_clusters, dtype=np.int32)
    cluster_sizes = np.asarray(cluster_n_triangles, dtype=np.int32)

    if cluster_sizes.size > 0:
        if keep_only_largest_component:
            largest_cluster = int(np.argmax(cluster_sizes))
            remove_mask = tri_clusters != largest_cluster
        else:
            small_clusters = np.where(cluster_sizes < int(min_component_faces))[0]
            # Always keep at least the largest cluster even if it has fewer than min_component_faces
            if small_clusters.size == cluster_sizes.size:
                largest_cluster = int(np.argmax(cluster_sizes))
                remove_mask = tri_clusters != largest_cluster
            else:
                remove_mask = np.isin(tri_clusters, small_clusters)
        if np.any(remove_mask):
            o3d_mesh.remove_triangles_by_mask(remove_mask)
            o3d_mesh.remove_unreferenced_vertices()

    if taubin_smoothing_iterations > 0 and len(o3d_mesh.triangles) > 0:
        verts_before = np.asarray(o3d_mesh.vertices, dtype=np.float64).copy()
        o3d_mesh = o3d_mesh.filter_smooth_taubin(
            number_of_iterations=int(taubin_smoothing_iterations)
        )
        verts_after = np.asarray(o3d_mesh.vertices, dtype=np.float64)
        bad_verts = ~np.all(np.isfinite(verts_after), axis=1)
        if np.any(bad_verts):
            verts_after[bad_verts] = verts_before[bad_verts]
            o3d_mesh.vertices = o3d.utility.Vector3dVector(verts_after)

    if target_faces is not None and len(o3d_mesh.triangles) > int(target_faces):
        # Preserve colors via KD-Tree transfer after quadric decimation
        orig_verts = np.asarray(o3d_mesh.vertices, dtype=np.float32)
        orig_cols = (
            np.asarray(o3d_mesh.vertex_colors, dtype=np.float32)
            if o3d_mesh.has_vertex_colors()
            else None
        )
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(
            target_number_of_triangles=int(target_faces)
        )
        o3d_mesh.remove_unreferenced_vertices()
        if orig_cols is not None and len(o3d_mesh.vertices) > 0:
            from scipy.spatial import cKDTree

            tree = cKDTree(orig_verts)
            _, nn_idx = tree.query(np.asarray(o3d_mesh.vertices, dtype=np.float32), k=1)
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
                orig_cols[nn_idx].astype(np.float64)
            )

    o3d_mesh.compute_vertex_normals()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    out_uvs = None
    if orig_uvs is not None and len(o3d_mesh.vertices) > 0:
        from scipy.spatial import cKDTree

        tree = cKDTree(orig_verts)
        _, nn_idx = tree.query(np.asarray(o3d_mesh.vertices, dtype=np.float32), k=1)
        out_uvs = orig_uvs[nn_idx]

    refined = MeshData.from_open3d(
        o3d_mesh,
        method=f"{mesh.method}_cleaned",
        generation_time_ms=mesh.generation_time_ms + elapsed_ms,
        parameters={
            **mesh.parameters,
            "min_component_faces": min_component_faces,
            "taubin_smoothing_iterations": taubin_smoothing_iterations,
            "target_faces": target_faces,
        },
        vertex_uvs=out_uvs,
    )

    if repair_holes:
        refined = fill_small_mesh_holes(
            refined,
            max_hole_loop_edges=max_hole_loop_edges,
            max_relative_depth_span=max_relative_depth_span,
            max_absolute_depth_span=max_absolute_depth_span,
        )

    return refined
