"""Multi-format 3D asset export (GLB, OBJ+MTL+PNG, PLY, STL), depth visualizations, and reports."""

from pathlib import Path
from typing import Any, Dict, Literal

import cv2
import numpy as np
import open3d as o3d
import trimesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial, SimpleMaterial

from app.core.exceptions import ExportError
from app.geometry.mesh import MeshData
from app.geometry.point_cloud import PointCloudData
from app.geometry.texture import TexturedMeshData


def render_depth_visualization(
    depth_or_disparity: np.ndarray,
    mode: Literal["heatmap", "grayscale", "inverted"] = "heatmap",
    colormap: int = cv2.COLORMAP_INFERNO,
) -> np.ndarray:
    """Render a 2D depth or disparity map to an HxWx3 RGB uint8 image."""
    arr = np.asarray(depth_or_disparity, dtype=np.float32)
    if arr.ndim != 2:
        raise ExportError(f"Expected 2D array for depth visualization, got shape {arr.shape}")

    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)

    vmin = float(np.min(arr[finite_mask]))
    vmax = float(np.max(arr[finite_mask]))
    if vmax - vmin < 1e-8:
        norm = np.zeros_like(arr, dtype=np.float32)
    else:
        norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)

    if mode == "inverted":
        norm = 1.0 - norm

    gray_u8 = np.round(norm * 255.0).astype(np.uint8)
    if mode in ("grayscale", "inverted"):
        return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)

    bgr = cv2.applyColorMap(gray_u8, colormap)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def save_depth_visualization_png(
    depth_or_disparity: np.ndarray,
    output_path: Path,
    mode: Literal["heatmap", "grayscale", "inverted"] = "heatmap",
) -> Path:
    """Save a visualized depth/disparity map to a PNG file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = render_depth_visualization(depth_or_disparity, mode=mode)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(output_path), bgr)
    if not ok or not output_path.exists():
        raise ExportError(f"Failed to write depth visualization PNG to {output_path}")
    return output_path


def save_point_cloud_ply(
    cloud: PointCloudData,
    output_path: Path,
    write_ascii: bool = False,
) -> Path:
    """Save a PointCloudData object to a standard PLY file via Open3D."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cloud.num_points == 0:
        raise ExportError("Cannot export an empty point cloud to PLY.")

    pcd = cloud.to_open3d()
    ok = o3d.io.write_point_cloud(
        str(output_path),
        pcd,
        write_ascii=write_ascii,
        compressed=False,
    )
    if not ok or not output_path.exists():
        raise ExportError(f"Failed to write PLY point cloud to {output_path}")
    return output_path


def camera_to_gltf_coords(coords: np.ndarray) -> np.ndarray:
    """Convert OpenCV camera coordinates (+X right, +Y down, +Z forward) to glTF/Three.js
    right-handed Y-up coordinates (+X right, +Y up, -Z forward):
        (X_gltf, Y_gltf, Z_gltf) = (X_cam, -Y_cam, -Z_cam).
    Note: Determinant of diag(1, -1, -1) is +1 (a 180-deg rotation around X), so triangle
    winding order is preserved without inversion!
    """
    out = np.asarray(coords, dtype=np.float32).copy()
    out[:, 1] = -out[:, 1]
    out[:, 2] = -out[:, 2]
    return out


def build_trimesh_from_textured_mesh(
    textured: TexturedMeshData,
    use_gltf_coords: bool = True,
    pbr_material: bool = True,
) -> trimesh.Trimesh:
    """Construct a Trimesh object with bound UV coordinates and texture material."""
    mesh = textured.mesh
    verts = camera_to_gltf_coords(mesh.vertices) if use_gltf_coords else mesh.vertices.copy()
    norms = (
        camera_to_gltf_coords(mesh.vertex_normals)
        if (use_gltf_coords and mesh.vertex_normals is not None)
        else (mesh.vertex_normals.copy() if mesh.vertex_normals is not None else None)
    )

    pil_tex = textured.pil_texture
    if pbr_material:
        material = PBRMaterial(
            name="hologen_surface",
            baseColorTexture=pil_tex,
            metallicFactor=0.0,
            roughnessFactor=0.85,
            doubleSided=True,
        )
    else:
        material = SimpleMaterial(image=pil_tex)

    visuals = TextureVisuals(
        uv=mesh.vertex_uvs,
        image=pil_tex,
        material=material,
    )

    tm = trimesh.Trimesh(
        vertices=verts,
        faces=mesh.faces.copy(),
        vertex_normals=norms,
        visual=visuals,
        process=False,
    )
    return tm


def export_mesh_glb(
    textured: TexturedMeshData,
    output_path: Path,
) -> Path:
    """Export a textured mesh to binary glTF 2.0 (.glb) format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tm = build_trimesh_from_textured_mesh(textured, use_gltf_coords=True, pbr_material=True)
        glb_bytes = tm.export(file_type="glb")
        output_path.write_bytes(glb_bytes)
    except Exception as exc:
        raise ExportError(f"Failed to export GLB to {output_path}: {exc}") from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ExportError(f"GLB export produced empty file at {output_path}")
    return output_path


def validate_exported_glb(glb_path: Path) -> Dict[str, Any]:
    """Independently load and validate a .glb file via trimesh."""
    glb_path = Path(glb_path)
    if not glb_path.exists():
        raise ExportError(f"GLB file not found for validation: {glb_path}")

    raw_bytes = glb_path.read_bytes()
    if len(raw_bytes) < 12 or raw_bytes[:4] != b"glTF":
        raise ExportError(f"Invalid glTF binary magic header in {glb_path}")

    scene = trimesh.load(str(glb_path), force="scene")
    geoms = list(scene.geometry.values())
    if not geoms:
        raise ExportError(f"Loaded GLB scene contains 0 geometries: {glb_path}")

    first_geom = geoms[0]
    has_uv = (
        hasattr(first_geom.visual, "uv")
        and first_geom.visual.uv is not None
        and len(first_geom.visual.uv) == len(first_geom.vertices)
    )
    has_tex = (
        hasattr(first_geom.visual, "material")
        and getattr(first_geom.visual.material, "baseColorTexture", None) is not None
    )

    return {
        "valid_glb_header": True,
        "file_size_bytes": len(raw_bytes),
        "num_geometries": len(geoms),
        "loaded_vertices": int(len(first_geom.vertices)),
        "loaded_faces": int(len(first_geom.faces)),
        "has_uv_coordinates": bool(has_uv),
        "has_embedded_texture": bool(has_tex),
    }


def export_mesh_obj(
    textured: TexturedMeshData,
    output_path: Path,
) -> Path:
    """Export a textured mesh to Wavefront OBJ + MTL + texture PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tm = build_trimesh_from_textured_mesh(textured, use_gltf_coords=True, pbr_material=False)
        tm.export(str(output_path), file_type="obj", include_texture=True)
    except Exception as exc:
        raise ExportError(f"Failed to export OBJ to {output_path}: {exc}") from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ExportError(f"OBJ export failed at {output_path}")
    return output_path


def export_mesh_ply(
    mesh: MeshData,
    output_path: Path,
) -> Path:
    """Export a vertex-colored triangle mesh to binary PLY format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d_mesh = mesh.to_open3d()
    ok = o3d.io.write_triangle_mesh(str(output_path), o3d_mesh, write_ascii=False)
    if not ok or not output_path.exists():
        raise ExportError(f"Failed to export mesh PLY to {output_path}")
    return output_path


def export_mesh_stl(
    mesh: MeshData,
    output_path: Path,
) -> Path:
    """Export a triangle mesh to binary STL format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tm = trimesh.Trimesh(
            vertices=camera_to_gltf_coords(mesh.vertices),
            faces=mesh.faces.copy(),
            process=False,
        )
        stl_bytes = tm.export(file_type="stl")
        output_path.write_bytes(stl_bytes)
    except Exception as exc:
        raise ExportError(f"Failed to export STL to {output_path}: {exc}") from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ExportError(f"STL export produced empty file at {output_path}")
    return output_path


def generate_markdown_reconstruction_report(report_data: Dict[str, Any]) -> str:
    """Generate a structured Markdown reconstruction report from measured report_data."""
    prep = report_data["preprocessing"]
    intr = report_data["camera_intrinsics"]
    depth = report_data["depth_estimation"]
    ref = report_data.get("depth_refinement", {})
    unc = report_data.get("uncertainty_proxy", {})
    pc = report_data["point_cloud"]
    mesh = report_data.get("mesh", {})
    tex = report_data.get("texturing", {})
    cov = report_data.get("estimated_coverage", {})
    timings = report_data["timings_ms"]
    exports = report_data.get("export_availability", {})

    lines = [
        f"# HoloGen 3D Reconstruction Report — `{report_data['run_id']}`",
        "",
        f"- **Timestamp (UTC):** `{report_data['timestamp_utc']}`",
        f"- **Pipeline Stage:** `{report_data['pipeline_stage']}`",
        f"- **Compute Device:** `{report_data['hardware']['compute_device']}` (`{report_data['hardware']['gpu_name']}`)",
        "",
        "## 1. Input & Preprocessing",
        f"- **Source Image:** `{prep.get('source_path') or 'in-memory'}`",
        f"- **Original Resolution:** `{prep['original_size']['width']} x {prep['original_size']['height']}`",
        f"- **Processed Resolution:** `{prep['processed_size']['width']} x {prep['processed_size']['height']}` (scale `{prep['resize_scale']}`)",
        f"- **Laplacian Blur Variance:** `{prep['quality']['blur_laplacian_variance']}` (`is_blurry={prep['quality']['is_blurry']}`)",
        f"- **Exposure Status:** `{prep['quality']['exposure']['status']}` (mean luminance `{prep['quality']['exposure']['mean_luminance']}`, dynamic range `{prep['quality']['exposure']['dynamic_range']}`)",
        "",
        "## 2. Camera Intrinsics",
        f"- **Source:** `{intr['source']}`",
        f"- **Focal Length:** `fx = {intr['fx']} px`, `fy = {intr['fy']} px`",
        f"- **Principal Point:** `cx = {intr['cx']} px`, `cy = {intr['cy']} px`",
        f"- **Field of View:** `{intr['fov_x_deg']} deg (H) x {intr['fov_y_deg']} deg (V)`",
        "",
        "## 3. Depth Estimation, Refinement & Uncertainty Proxy",
        f"- **Model:** `{depth['model_name']}` (`{depth['model_id']}`, License: `{depth['license']}`)",
        f"- **Scale Semantics:** `{depth['scale_semantics']}` — *{depth['scale_disclaimer']}*",
        f"- **Inference Time:** `{depth['inference_time_ms']} ms` (Peak VRAM: `{depth['vram_allocated_mb']} MB`)",
        f"- **Refinement Method:** `{ref.get('method', 'none')}` (Discontinuity fraction: `{ref.get('discontinuity_fraction', 0.0)}`)",
        f"- **Uncertainty Proxy (Uncalibrated):** Mean `{unc.get('mean_uncertainty_proxy', 0.0)}`, P90 `{unc.get('p90_uncertainty_proxy', 0.0)}`, Reliable Pixels `{round(unc.get('reliable_pixel_fraction', 1.0) * 100.0, 2)}%`",
        "",
        "## 4. 3D Point Cloud, Surface Mesh & Coverage",
        f"- **Raw Back-Projected Points:** `{pc.get('raw_points', pc['num_points'])}`",
        f"- **Cleaned Point Cloud Points:** `{pc['num_points']}`",
        f"- **Mesh Method:** `{mesh.get('method', 'N/A')}`",
        f"- **Mesh Geometry:** `{mesh.get('num_vertices', 0)}` vertices, `{mesh.get('num_faces', 0)}` triangles",
        f"- **Topology:** `{mesh.get('topology', {}).get('num_connected_components', 0)}` components, `{mesh.get('topology', {}).get('num_interior_holes', 0)}` interior holes",
        f"- **Frustum Mesh Coverage:** `{cov.get('frustum_mesh_coverage_pct', tex.get('frustum_coverage_pct', 0.0))}%` ({cov.get('definition', '')})",
        "",
        "## 5. Stage Timings",
        "| Stage | Time (ms) |",
        "| :--- | :---: |",
    ]
    for k, v in timings.items():
        lines.append(f"| `{k}` | `{v}` |")

    lines.extend(
        [
            "",
            "## 6. Exported Assets & Validation",
            "| Format | Filename | Available | Size (bytes) |",
            "| :--- | :--- | :---: | :---: |",
        ]
    )
    for fmt, info in exports.items():
        if isinstance(info, dict) and "filename" in info:
            lines.append(
                f"| **{fmt.upper()}** | `{info['filename']}` | `{info['available']}` | `{info['size_bytes']}` |"
            )

    return "\n".join(lines) + "\n"
