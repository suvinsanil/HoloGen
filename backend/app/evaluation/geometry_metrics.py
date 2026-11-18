"""3D geometry evaluation metrics for meshes and point clouds (Chamfer, Hausdorff, Completeness, Noise)."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

from app.geometry.mesh import MeshData
from app.geometry.point_cloud import PointCloudData
from app.geometry.refinement import analyze_mesh_topology


@dataclass
class MeshReconstructionEval:
    """Comprehensive evaluation report for a reconstructed 3D surface mesh against its source cloud."""

    method: str
    time_ms: float
    num_vertices: int
    num_faces: int
    num_connected_components: int
    num_boundary_edges: int
    num_interior_holes: int
    completeness_pct: float  # % of reference cloud points within threshold tau of mesh surface
    surface_noise_rms: float  # RMS distance from mesh vertices to nearest reference cloud point
    chamfer_l2: float  # Bidirectional mean L2 distance (mesh <-> reference cloud)
    hausdorff_95: float  # 95th-percentile Hausdorff distance
    threshold_tau: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "time_ms": round(self.time_ms, 2),
            "num_vertices": self.num_vertices,
            "num_faces": self.num_faces,
            "num_connected_components": self.num_connected_components,
            "num_boundary_edges": self.num_boundary_edges,
            "num_interior_holes": self.num_interior_holes,
            "completeness_pct": round(self.completeness_pct, 2),
            "surface_noise_rms": round(self.surface_noise_rms, 5),
            "chamfer_l2": round(self.chamfer_l2, 5),
            "hausdorff_95": round(self.hausdorff_95, 5),
            "threshold_tau": round(self.threshold_tau, 5),
        }


def sample_points_on_mesh(
    mesh: MeshData,
    num_samples: int = 50000,
    seed: int = 42,
) -> np.ndarray:
    """Sample points area-uniformly across mesh triangles (combined with vertices)."""
    if mesh.num_faces == 0 or mesh.num_vertices == 0:
        return mesh.vertices.copy()

    rng = np.random.default_rng(seed)
    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = float(np.sum(areas))
    if total_area <= 1e-12:
        return mesh.vertices.copy()

    probs = areas / total_area
    tri_idx = rng.choice(mesh.num_faces, size=int(num_samples), replace=True, p=probs)

    r1 = np.sqrt(rng.random(int(num_samples), dtype=np.float32))[:, None]
    r2 = rng.random(int(num_samples), dtype=np.float32)[:, None]
    pts = (1.0 - r1) * v0[tri_idx] + (r1 * (1.0 - r2)) * v1[tri_idx] + (r1 * r2) * v2[tri_idx]
    return np.vstack([mesh.vertices, pts.astype(np.float32)])


def compute_point_cloud_distance_metrics(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    threshold_tau: float = 0.05,
) -> Dict[str, float]:
    """Compute bidirectional Chamfer distance, Hausdorff distance, Precision, Recall, and F-score.

    Given predicted points P (source_pts) and ground-truth/reference points Q (target_pts):
    - Accuracy (P -> Q): distances from each p in P to nearest q in Q.
    - Completeness (Q -> P): distances from each q in Q to nearest p in P.
    - Precision(@tau): fraction of p in P with dist(p, Q) <= tau.
    - Recall(@tau): fraction of q in Q with dist(q, P) <= tau.
    """
    P = np.asarray(source_pts, dtype=np.float64)
    Q = np.asarray(target_pts, dtype=np.float64)
    if P.shape[0] == 0 or Q.shape[0] == 0:
        raise ValueError("Cannot compute distance metrics on empty point sets.")

    tree_P = cKDTree(P)
    tree_Q = cKDTree(Q)

    dist_p_to_q, _ = tree_Q.query(P, k=1, workers=-1)
    dist_q_to_p, _ = tree_P.query(Q, k=1, workers=-1)

    acc_mean = float(np.mean(dist_p_to_q))
    comp_mean = float(np.mean(dist_q_to_p))
    chamfer_l1 = 0.5 * (acc_mean + comp_mean)
    chamfer_l2 = float(0.5 * (np.sqrt(np.mean(dist_p_to_q ** 2)) + np.sqrt(np.mean(dist_q_to_p ** 2))))

    hausdorff_max = float(max(np.max(dist_p_to_q), np.max(dist_q_to_p)))
    hausdorff_95 = float(
        max(np.percentile(dist_p_to_q, 95.0), np.percentile(dist_q_to_p, 95.0))
    )

    tau = float(threshold_tau)
    precision = float(np.mean(dist_p_to_q <= tau))
    recall = float(np.mean(dist_q_to_p <= tau))
    fscore = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 1e-8
        else 0.0
    )

    rms_p_to_q = float(np.sqrt(np.mean(dist_p_to_q ** 2)))

    return {
        "accuracy_mean": acc_mean,
        "completeness_mean": comp_mean,
        "chamfer_l1": chamfer_l1,
        "chamfer_l2": chamfer_l2,
        "hausdorff_max": hausdorff_max,
        "hausdorff_95": hausdorff_95,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "surface_noise_rms": rms_p_to_q,
    }


def evaluate_mesh_reconstruction(
    mesh: MeshData,
    reference_cloud: PointCloudData,
    threshold_tau: Optional[float] = None,
) -> MeshReconstructionEval:
    """Evaluate a reconstructed mesh against the reference point cloud for Experiment B."""
    topo = analyze_mesh_topology(mesh)

    ref_pts = reference_cloud.points
    if threshold_tau is None:
        tree_ref = cKDTree(ref_pts)
        nn_d, _ = tree_ref.query(ref_pts, k=2, workers=-1)
        mean_nn = float(np.mean(nn_d[:, 1])) if nn_d.shape[1] > 1 else 0.02
        tau = max(mean_nn * 2.5, 0.025)
    else:
        tau = float(threshold_tau)

    mesh_surface_pts = sample_points_on_mesh(mesh, num_samples=40000)
    dist_metrics = compute_point_cloud_distance_metrics(
        source_pts=mesh_surface_pts,
        target_pts=ref_pts,
        threshold_tau=tau,
    )

    return MeshReconstructionEval(
        method=mesh.method,
        time_ms=mesh.generation_time_ms,
        num_vertices=topo.num_vertices,
        num_faces=topo.num_faces,
        num_connected_components=topo.num_connected_components,
        num_boundary_edges=topo.num_boundary_edges,
        num_interior_holes=topo.num_interior_holes,
        completeness_pct=dist_metrics["recall"] * 100.0,
        surface_noise_rms=dist_metrics["surface_noise_rms"],
        chamfer_l2=dist_metrics["chamfer_l2"],
        hausdorff_95=dist_metrics["hausdorff_95"],
        threshold_tau=tau,
    )
