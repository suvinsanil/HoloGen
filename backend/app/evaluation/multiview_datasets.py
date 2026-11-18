"""Loaders for multi-view calibrated datasets:
1. Open3D SampleFountainRGBDImages — labeled strictly as 'reference geometry (RGB-D reconstruction)'.
2. ETH3D 'pipes' DSLR multi-view dataset — ground-truth camera poses and Faro laser scan.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.vision.camera import CameraIntrinsics


@dataclass
class CalibratedView:
    """Single calibrated camera view with world-to-camera pose and intrinsics."""

    view_id: str
    image_path: Path
    intrinsics_full: CameraIntrinsics
    R_w2c: np.ndarray  # float64 (3, 3) world-to-camera rotation
    t_w2c: np.ndarray  # float64 (3,) world-to-camera translation
    pose_semantics: str  # "reference_rgbd_trajectory" or "ground_truth_calibrated"

    @property
    def camera_center_world(self) -> np.ndarray:
        """Camera optical center in world coordinates: C_w = -R_w2c^T @ t_w2c."""
        return -(self.R_w2c.T @ self.t_w2c)

    def load_rgb_resized(
        self, max_dim: Optional[int] = None
    ) -> Tuple[np.ndarray, CameraIntrinsics]:
        """Load RGB uint8 image and return scaled CameraIntrinsics."""
        pil_img = Image.open(self.image_path).convert("RGB")
        orig_w, orig_h = pil_img.size
        if max_dim is not None and max(orig_w, orig_h) > max_dim:
            scale = float(max_dim) / float(max(orig_w, orig_h))
            new_w = max(16, int(round(orig_w * scale)))
            new_h = max(16, int(round(orig_h * scale)))
            pil_img = pil_img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
            intr = self.intrinsics_full.scale(
                float(new_w) / float(orig_w), float(new_h) / float(orig_h)
            )
        else:
            intr = self.intrinsics_full
        return np.asarray(pil_img, dtype=np.uint8), intr


def quaternion_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert unit quaternion (qw, qx, qy, qz) to a 3x3 rotation matrix (COLMAP convention)."""
    n = float(np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qw * qz),
                2.0 * (qx * qz + qw * qy),
            ],
            [
                2.0 * (qx * qy + qw * qz),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qw * qx),
            ],
            [
                2.0 * (qx * qz - qw * qy),
                2.0 * (qy * qz + qw * qx),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def relative_pose_between_views(
    view1: CalibratedView,
    view2: CalibratedView,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute relative pose (R_12, t_12) mapping Camera 1 coordinates to Camera 2 coordinates:
        X_cam2 = R_12 @ X_cam1 + t_12.
    """
    R_12 = view2.R_w2c @ view1.R_w2c.T
    t_12 = view2.t_w2c - (R_12 @ view1.t_w2c)
    return R_12, t_12


def load_fountain_dataset(datasets_dir: Path) -> Tuple[List[CalibratedView], Path]:
    """Load Open3D SampleFountainRGBDImages (33 views + reference RGB-D mesh)."""
    extract_root = Path(datasets_dir) / "fountain" / "extract" / "SampleFountainRGBDImages"
    img_dir = extract_root / "image"
    key_log = extract_root / "scene" / "key.log"
    ref_mesh_path = extract_root / "scene" / "integrated.ply"

    img_paths = sorted(img_dir.glob("*.jpg"))
    lines = [ln.strip() for ln in key_log.read_text(encoding="utf-8").splitlines() if ln.strip()]

    # PrimeSense 640x480 calibrated intrinsics
    fov_x = float(np.degrees(2.0 * np.arctan(640.0 / (2.0 * 525.0))))
    fov_y = float(np.degrees(2.0 * np.arctan(480.0 / (2.0 * 525.0))))
    intr = CameraIntrinsics(
        width=640,
        height=480,
        fx=525.0,
        fy=525.0,
        cx=319.5,
        cy=239.5,
        source="primesense_calibrated",
        fov_x_deg=fov_x,
        fov_y_deg=fov_y,
    )

    views: List[CalibratedView] = []
    num_frames = min(len(img_paths), len(lines) // 5)
    for i in range(num_frames):
        mat_lines = lines[i * 5 + 1 : i * 5 + 5]
        T_c2w = np.array(
            [[float(x) for x in row.split()] for row in mat_lines],
            dtype=np.float64,
        )
        T_w2c = np.linalg.inv(T_c2w)
        R_w2c = T_w2c[:3, :3]
        t_w2c = T_w2c[:3, 3]
        views.append(
            CalibratedView(
                view_id=f"fountain_{i:02d}_{img_paths[i].stem}",
                image_path=img_paths[i],
                intrinsics_full=intr,
                R_w2c=R_w2c,
                t_w2c=t_w2c,
                pose_semantics="reference_rgbd_trajectory",
            )
        )
    return views, ref_mesh_path


def load_eth3d_pipes_dataset(datasets_dir: Path) -> Tuple[List[CalibratedView], Path]:
    """Load ETH3D 'pipes' DSLR views (14 images + ground-truth poses + Faro laser scan)."""
    pipes_root = Path(datasets_dir) / "eth3d" / "pipes_extracted" / "pipes"
    calib_dir = pipes_root / "dslr_calibration_jpg"
    images_dir = pipes_root / "images"
    scan_path = pipes_root / "dslr_scan_eval" / "scan1.ply"

    # Parse cameras.txt
    cam_lines = [
        ln.strip()
        for ln in (calib_dir / "cameras.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    parts = cam_lines[0].split()
    w, h = int(parts[2]), int(parts[3])
    fx, fy, cx, cy = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
    fov_x = float(np.degrees(2.0 * np.arctan(w / (2.0 * fx))))
    fov_y = float(np.degrees(2.0 * np.arctan(h / (2.0 * fy))))
    intr_full = CameraIntrinsics(
        width=w,
        height=h,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        source="eth3d_calibrated",
        fov_x_deg=fov_x,
        fov_y_deg=fov_y,
    )

    # Parse images.txt (every first non-comment line of a 2-line block is the image header)
    img_lines = [
        ln.strip()
        for ln in (calib_dir / "images.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]

    views_by_name: Dict[str, CalibratedView] = {}
    for idx in range(0, len(img_lines), 2):
        toks = img_lines[idx].split()
        qw, qx, qy, qz = float(toks[1]), float(toks[2]), float(toks[3]), float(toks[4])
        tx, ty, tz = float(toks[5]), float(toks[6]), float(toks[7])
        rel_name = toks[9]
        img_p = images_dir / rel_name
        R_w2c = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        t_w2c = np.array([tx, ty, tz], dtype=np.float64)
        views_by_name[img_p.name] = CalibratedView(
            view_id=f"pipes_{img_p.stem}",
            image_path=img_p,
            intrinsics_full=intr_full,
            R_w2c=R_w2c,
            t_w2c=t_w2c,
            pose_semantics="ground_truth_calibrated",
        )

    sorted_views = [views_by_name[k] for k in sorted(views_by_name.keys())]
    return sorted_views, scan_path
