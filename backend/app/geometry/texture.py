"""Projective camera UV mapping, texture resolution control, and View-Frustum Coverage computation."""

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from app.core.exceptions import GeometryProcessingError
from app.geometry.mesh import MeshData
from app.vision.camera import CameraIntrinsics


@dataclass
class TexturedMeshData:
    """Container for a UV-mapped 3D triangle mesh with a bound RGB texture image."""

    mesh: MeshData  # Guaranteed to have vertex_uvs (V, 2) in [0, 1]
    texture_image: np.ndarray  # uint8 (H_tex, W_tex, 3) RGB texture map
    original_texture_size: tuple[int, int]  # (W, H)
    final_texture_size: tuple[int, int]  # (W_tex, H_tex)
    view_frustum_coverage_pct: float  # % of camera image area covered by projected mesh triangles
    texturing_time_ms: float

    @property
    def frustum_coverage_pct(self) -> float:
        """Backward-compatible alias for view_frustum_coverage_pct."""
        return self.view_frustum_coverage_pct

    @property
    def pil_texture(self) -> Image.Image:
        return Image.fromarray(self.texture_image, mode="RGB")

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "original_texture_size": {
                "width": self.original_texture_size[0],
                "height": self.original_texture_size[1],
            },
            "final_texture_size": {
                "width": self.final_texture_size[0],
                "height": self.final_texture_size[1],
            },
            "view_frustum_coverage_pct": round(self.view_frustum_coverage_pct, 2),
            "coverage_definition": (
                "View-Frustum Coverage is the percentage of 2D input camera image pixels (W x H) "
                "subtended by valid reconstructed mesh triangles when rasterized in projective UV space. "
                "NOTE: It measures 2D view-frustum projection occupancy only and does NOT measure "
                "3D scene completeness (occluded and back-facing surfaces are unobserved)."
            ),
            "texturing_time_ms": round(self.texturing_time_ms, 3),
        }


def compute_projective_uvs(
    vertices: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Project 3D camera-space vertices (X, Y, Z) onto normalized glTF UV coordinates (s, t) in [0, 1]^2.

    Formulation:
        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy
        s = clip(u / (W - 1), 0.0, 1.0)
        t = clip(1.0 - v / (H - 1), 0.0, 1.0)  (glTF/OpenGL texture origin is bottom-left)
    """
    pts = np.asarray(vertices, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise GeometryProcessingError(f"Expected Vx3 vertex array, got shape {pts.shape}")

    z = np.maximum(pts[:, 2], 1e-5)
    u = float(intrinsics.fx) * (pts[:, 0] / z) + float(intrinsics.cx)
    v = float(intrinsics.fy) * (pts[:, 1] / z) + float(intrinsics.cy)

    w_denom = max(float(intrinsics.width - 1), 1.0)
    h_denom = max(float(intrinsics.height - 1), 1.0)

    s = np.clip(u / w_denom, 0.0, 1.0)
    t = np.clip(1.0 - (v / h_denom), 0.0, 1.0)
    return np.stack([s, t], axis=-1).astype(np.float32)


def compute_view_frustum_coverage_pct(
    uvs: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
) -> float:
    """Compute View-Frustum Coverage: the percentage of the 2D camera image plane covered by
    projected mesh triangles. Does NOT measure 3D volumetric scene completeness.
    """
    if faces.shape[0] == 0 or uvs.shape[0] == 0 or width <= 0 or height <= 0:
        return 0.0

    u_px = np.round(uvs[:, 0] * float(width - 1)).astype(np.int32)
    v_px = np.round((1.0 - uvs[:, 1]) * float(height - 1)).astype(np.int32)
    pts_2d = np.stack([u_px, v_px], axis=-1)

    mask = np.zeros((height, width), dtype=np.uint8)
    for tri in faces:
        poly = pts_2d[tri]
        cv2.fillConvexPoly(mask, poly, 255)

    covered = int(np.count_nonzero(mask))
    return float(covered) * 100.0 / float(width * height)


# Backward-compatible alias
compute_frustum_uv_coverage_pct = compute_view_frustum_coverage_pct


def apply_projective_texturing(
    mesh: MeshData,
    rgb_uint8: np.ndarray,
    intrinsics: CameraIntrinsics,
    texture_max_dimension: Optional[int] = None,
) -> TexturedMeshData:
    """Project the source RGB image onto a 3D mesh via pinhole camera projection."""
    t0 = time.perf_counter()
    if mesh.num_vertices == 0 or mesh.num_faces == 0:
        raise GeometryProcessingError("Cannot texture an empty mesh.")
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise GeometryProcessingError(f"Expected HxWx3 RGB uint8 texture, got {rgb_uint8.shape}")

    orig_h, orig_w = int(rgb_uint8.shape[0]), int(rgb_uint8.shape[1])

    uvs = compute_projective_uvs(mesh.vertices, intrinsics)

    tex_img = rgb_uint8.copy()
    if texture_max_dimension is not None and texture_max_dimension >= 8:
        max_side = max(orig_w, orig_h)
        if max_side != int(texture_max_dimension):
            scale = float(texture_max_dimension) / float(max_side)
            new_w = max(2, int(round(orig_w * scale)))
            new_h = max(2, int(round(orig_h * scale)))
            pil_resized = Image.fromarray(tex_img, mode="RGB").resize(
                (new_w, new_h), resample=Image.Resampling.LANCZOS
            )
            tex_img = np.asarray(pil_resized, dtype=np.uint8)

    final_h, final_w = int(tex_img.shape[0]), int(tex_img.shape[1])

    sample_u = np.clip(np.round(uvs[:, 0] * float(final_w - 1)).astype(np.int32), 0, final_w - 1)
    sample_v = np.clip(
        np.round((1.0 - uvs[:, 1]) * float(final_h - 1)).astype(np.int32), 0, final_h - 1
    )
    sampled_colors = tex_img[sample_v, sample_u].astype(np.float32) / 255.0

    coverage_pct = compute_view_frustum_coverage_pct(
        uvs=uvs,
        faces=mesh.faces,
        width=intrinsics.width,
        height=intrinsics.height,
    )

    updated_mesh = MeshData(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        vertex_colors=sampled_colors,
        vertex_normals=mesh.vertex_normals.copy() if mesh.vertex_normals is not None else None,
        vertex_uvs=uvs,
        method=mesh.method,
        generation_time_ms=mesh.generation_time_ms,
        parameters=mesh.parameters.copy(),
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return TexturedMeshData(
        mesh=updated_mesh,
        texture_image=tex_img,
        original_texture_size=(orig_w, orig_h),
        final_texture_size=(final_w, final_h),
        view_frustum_coverage_pct=coverage_pct,
        texturing_time_ms=elapsed_ms,
    )
