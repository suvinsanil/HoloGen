"""Unit tests for pinhole camera intrinsics estimation and projection math with hand-computed values."""

import math
import numpy as np
import pytest

from app.core.exceptions import IntrinsicsEstimationError
from app.vision.camera import (
    estimate_intrinsics,
    focal_length_from_fov,
)


def test_hand_computed_fov_60_deg_intrinsics() -> None:
    """Hand-computed check: for W=640, H=480, FOV_x=60 deg:
    tan(30 deg) = 1 / sqrt(3)
    fx = fy = 640 / (2 * (1 / sqrt(3))) = 320 * sqrt(3) = 554.2562584220407
    cx = 320.0, cy = 240.0
    """
    expected_fx = 320.0 * math.sqrt(3.0)
    intr = estimate_intrinsics(width=640, height=480, fov_deg=60.0)

    assert intr.source == "fov_default"
    assert intr.fx == pytest.approx(expected_fx, rel=1e-9)
    assert intr.fy == pytest.approx(expected_fx, rel=1e-9)
    assert intr.cx == pytest.approx(320.0, abs=1e-9)
    assert intr.cy == pytest.approx(240.0, abs=1e-9)
    assert intr.fov_x_deg == pytest.approx(60.0, abs=1e-6)

    # Verify K and K_inv multiplication equals 3x3 identity
    identity = intr.K @ intr.K_inv
    np.testing.assert_allclose(identity, np.eye(3), atol=1e-12)


def test_hand_computed_fov_90_deg_intrinsics() -> None:
    """Hand-computed check: for W=800, H=600, FOV_x=90 deg:
    tan(45 deg) = 1.0 -> fx = fy = 800 / 2 = 400.0, cx = 400.0, cy = 300.0.
    """
    fx = focal_length_from_fov(800, 90.0)
    assert fx == pytest.approx(400.0, abs=1e-9)

    intr = estimate_intrinsics(width=800, height=600, fov_deg=90.0)
    assert intr.fx == pytest.approx(400.0, abs=1e-9)
    assert intr.fy == pytest.approx(400.0, abs=1e-9)
    assert intr.cx == pytest.approx(400.0, abs=1e-9)
    assert intr.cy == pytest.approx(300.0, abs=1e-9)


def test_exif_35mm_and_physical_focal_length_priority() -> None:
    """Hand-computed check:
    For W=600, H=400, EXIF FocalLengthIn35mmFilm=50mm:
    fx = (50 / 36) * max(600, 400) = (50 / 36) * 600 = 833.3333333333334.
    """
    exif_35 = {"focal_length_35mm": 50.0, "focal_length_mm": 24.0}
    intr_35 = estimate_intrinsics(width=600, height=400, exif_metadata=exif_35)
    assert intr_35.source == "exif_35mm"
    assert intr_35.fx == pytest.approx((50.0 / 36.0) * 600.0, rel=1e-9)

    # Physical focal length only: 18mm on 36mm sensor -> fx = (18 / 36) * 600 = 300.0
    exif_phys = {"focal_length_35mm": None, "focal_length_mm": 18.0}
    intr_phys = estimate_intrinsics(
        width=600, height=400, exif_metadata=exif_phys, sensor_width_mm=36.0
    )
    assert intr_phys.source == "exif_focal_mm"
    assert intr_phys.fx == pytest.approx(300.0, rel=1e-9)

    # Explicit override takes highest priority over EXIF
    intr_override = estimate_intrinsics(
        width=600, height=400, exif_metadata=exif_35, focal_length_px=500.0
    )
    assert intr_override.source == "explicit_override"
    assert intr_override.fx == pytest.approx(500.0, rel=1e-9)


def test_hand_computed_project_and_backproject_roundtrip() -> None:
    """Hand-computed check:
    Given fx=500, fy=500, cx=320, cy=240:
    Pixel (u=420, v=140) at depth Z=2.5:
      X = (420 - 320) * 2.5 / 500 = 100 * 2.5 / 500 = 0.5
      Y = (140 - 240) * 2.5 / 500 = -100 * 2.5 / 500 = -0.5
      Z = 2.5
    Re-projecting (0.5, -0.5, 2.5) must return (420.0, 140.0).
    """
    intr = estimate_intrinsics(width=640, height=480, focal_length_px=500.0)
    x, y, z = intr.backproject_pixel(u=420.0, v=140.0, z=2.5)
    assert x == pytest.approx(0.5, abs=1e-9)
    assert y == pytest.approx(-0.5, abs=1e-9)
    assert z == pytest.approx(2.5, abs=1e-9)

    uv = intr.project(np.array([[x, y, z]], dtype=np.float64))
    assert uv[0, 0] == pytest.approx(420.0, abs=1e-9)
    assert uv[0, 1] == pytest.approx(140.0, abs=1e-9)


def test_intrinsics_scaling() -> None:
    """Verify scaling intrinsics by 0.5 halves width, height, fx, fy, cx, cy while preserving FOV."""
    intr = estimate_intrinsics(width=640, height=480, fov_deg=60.0)
    scaled = intr.scale(0.5)
    assert scaled.width == 320
    assert scaled.height == 240
    assert scaled.fx == pytest.approx(intr.fx * 0.5, rel=1e-9)
    assert scaled.cx == pytest.approx(160.0, abs=1e-9)
    assert scaled.cy == pytest.approx(120.0, abs=1e-9)
    assert scaled.fov_x_deg == pytest.approx(intr.fov_x_deg, abs=1e-6)


def test_invalid_intrinsics_parameters_raise() -> None:
    """Verify non-physical dimensions, FOV, or Z<=0 raise IntrinsicsEstimationError."""
    with pytest.raises(IntrinsicsEstimationError):
        estimate_intrinsics(width=0, height=480)
    with pytest.raises(IntrinsicsEstimationError):
        focal_length_from_fov(640, 185.0)
    intr = estimate_intrinsics(width=640, height=480, focal_length_px=500.0)
    with pytest.raises(IntrinsicsEstimationError):
        intr.backproject_pixel(100.0, 100.0, z=0.0)
