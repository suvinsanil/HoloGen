"""Camera intrinsics estimation (EXIF and horizontal FOV fallback) and projection math."""

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.core.exceptions import IntrinsicsEstimationError


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters and matrix representation."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    source: str
    fov_x_deg: float
    fov_y_deg: float

    @property
    def K(self) -> np.ndarray:
        """Return 3x3 pinhole camera intrinsic matrix K."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def K_inv(self) -> np.ndarray:
        """Return analytical 3x3 inverse intrinsic matrix K^-1."""
        return np.array(
            [
                [1.0 / self.fx, 0.0, -self.cx / self.fx],
                [0.0, 1.0 / self.fy, -self.cy / self.fy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def scale(self, scale_x: float, scale_y: Optional[float] = None) -> "CameraIntrinsics":
        """Return scaled intrinsics for a resized image."""
        sy = scale_x if scale_y is None else scale_y
        if scale_x <= 0 or sy <= 0:
            raise IntrinsicsEstimationError("Scale factors must be positive.")
        new_w = max(1, int(round(self.width * scale_x)))
        new_h = max(1, int(round(self.height * sy)))
        new_fx = self.fx * scale_x
        new_fy = self.fy * sy
        new_cx = self.cx * scale_x
        new_cy = self.cy * sy
        fov_x = math.degrees(2.0 * math.atan(new_w / (2.0 * new_fx)))
        fov_y = math.degrees(2.0 * math.atan(new_h / (2.0 * new_fy)))
        return CameraIntrinsics(
            width=new_w,
            height=new_h,
            fx=new_fx,
            fy=new_fy,
            cx=new_cx,
            cy=new_cy,
            source=self.source,
            fov_x_deg=fov_x,
            fov_y_deg=fov_y,
        )

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project Nx3 3D camera-space points (X, Y, Z) to Nx2 pixel coordinates (u, v)."""
        pts = np.asarray(points_3d, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise IntrinsicsEstimationError(f"Expected Nx3 array, got shape {pts.shape}")
        z = pts[:, 2]
        if np.any(z <= 1e-8):
            raise IntrinsicsEstimationError("All points must have positive depth Z > 0 for projection.")
        u = self.fx * (pts[:, 0] / z) + self.cx
        v = self.fy * (pts[:, 1] / z) + self.cy
        return np.stack([u, v], axis=-1)

    def backproject_pixel(self, u: float, v: float, z: float) -> Tuple[float, float, float]:
        """Back-project a single pixel (u, v) with depth Z to camera 3D coordinates (X, Y, Z)."""
        if z <= 0.0:
            raise IntrinsicsEstimationError(f"Depth Z must be positive, got {z}")
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return (float(x), float(y), float(z))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize camera intrinsics to a JSON-friendly dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "fx": round(self.fx, 4),
            "fy": round(self.fy, 4),
            "cx": round(self.cx, 4),
            "cy": round(self.cy, 4),
            "source": self.source,
            "fov_x_deg": round(self.fov_x_deg, 3),
            "fov_y_deg": round(self.fov_y_deg, 3),
            "K": [[round(float(val), 4) for val in row] for row in self.K],
        }


def focal_length_from_fov(dimension_px: int, fov_deg: float) -> float:
    """Compute focal length in pixels from image dimension and field of view in degrees."""
    if dimension_px <= 0:
        raise IntrinsicsEstimationError(f"Image dimension must be > 0, got {dimension_px}")
    if not (1.0 <= fov_deg < 179.0):
        raise IntrinsicsEstimationError(f"FOV must be in [1, 179) degrees, got {fov_deg}")
    half_fov_rad = math.radians(fov_deg * 0.5)
    return float(dimension_px) / (2.0 * math.tan(half_fov_rad))


def estimate_intrinsics(
    width: int,
    height: int,
    exif_metadata: Optional[Dict[str, Any]] = None,
    fov_deg: float = 60.0,
    focal_length_px: Optional[float] = None,
    principal_point: Optional[Tuple[float, float]] = None,
    sensor_width_mm: float = 36.0,
) -> CameraIntrinsics:
    """Estimate pinhole camera intrinsics from override, EXIF tags, or horizontal FOV fallback.

    Priority:
    1. Explicit `focal_length_px` parameter if provided.
    2. EXIF `focal_length_35mm` (using 36mm full-frame reference across longest image edge).
    3. EXIF `focal_length_mm` (using `sensor_width_mm` across longest image edge).
    4. Horizontal field-of-view fallback (`fov_deg`, default 60.0 degrees).
    """
    if width <= 0 or height <= 0:
        raise IntrinsicsEstimationError(f"Invalid image dimensions: {width}x{height}")
    if sensor_width_mm <= 0.0:
        raise IntrinsicsEstimationError(f"sensor_width_mm must be > 0, got {sensor_width_mm}")

    cx = float(width) / 2.0 if principal_point is None else float(principal_point[0])
    cy = float(height) / 2.0 if principal_point is None else float(principal_point[1])

    fx: Optional[float] = None
    source = "fov_default"

    if focal_length_px is not None:
        if focal_length_px <= 0.0:
            raise IntrinsicsEstimationError(f"focal_length_px must be > 0, got {focal_length_px}")
        fx = float(focal_length_px)
        source = "explicit_override"
    elif exif_metadata:
        f35 = exif_metadata.get("focal_length_35mm")
        f_mm = exif_metadata.get("focal_length_mm")
        max_dim = float(max(width, height))
        if f35 is not None and float(f35) > 1.0:
            fx = (float(f35) / 36.0) * max_dim
            source = "exif_35mm"
        elif f_mm is not None and float(f_mm) > 0.5:
            fx = (float(f_mm) / float(sensor_width_mm)) * max_dim
            source = "exif_focal_mm"

    if fx is None or fx <= 1.0:
        fx = focal_length_from_fov(width, fov_deg)
        source = "fov_default"

    fy = fx
    fov_x = math.degrees(2.0 * math.atan(float(width) / (2.0 * fx)))
    fov_y = math.degrees(2.0 * math.atan(float(height) / (2.0 * fy)))

    return CameraIntrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        source=source,
        fov_x_deg=fov_x,
        fov_y_deg=fov_y,
    )
