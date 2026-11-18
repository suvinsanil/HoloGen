"""Unit tests for image preprocessing, EXIF orientation, blur, and exposure statistics."""

import numpy as np
from PIL import Image
import pytest

from app.core.exceptions import ImagePreprocessingError
from app.vision.preprocessing import (
    compute_blur_score,
    compute_exposure_stats,
    preprocess_image,
)


def test_aspect_preserving_resize_and_normalization() -> None:
    """Verify 1200x600 image resized with max_dimension=600 becomes 600x300 and [0, 1] normalized."""
    arr = np.zeros((600, 1200, 3), dtype=np.uint8)
    arr[:, :, 0] = 255  # Pure red
    arr[:, :, 1] = 128
    arr[:, :, 2] = 0

    res = preprocess_image(arr, max_dimension=600)
    assert res.original_size == (1200, 600)
    assert res.processed_size == (600, 300)
    assert pytest.approx(res.resize_scale, rel=1e-6) == 0.5
    assert res.rgb_uint8.shape == (300, 600, 3)
    assert res.rgb_normalized.shape == (300, 600, 3)
    assert res.rgb_normalized.dtype == np.float32
    assert pytest.approx(float(res.rgb_normalized[0, 0, 0]), abs=1e-5) == 1.0
    assert pytest.approx(float(res.rgb_normalized[0, 0, 1]), abs=1e-3) == 128.0 / 255.0
    assert pytest.approx(float(res.rgb_normalized[0, 0, 2]), abs=1e-5) == 0.0


def test_exif_orientation_transposition(tmp_path) -> None:
    """Verify EXIF Orientation=6 (Rotate 270 CW / 90 CCW) transposes 200x100 into 100x200."""
    img = Image.new("RGB", (200, 100), color=(40, 120, 200))
    exif = img.getexif()
    exif[0x0112] = 6  # Orientation = 6
    img_path = tmp_path / "oriented.jpg"
    img.save(img_path, exif=exif.tobytes())

    res = preprocess_image(img_path, max_dimension=640)
    assert res.exif_orientation_applied is True
    assert res.exif_metadata["orientation"] == 6
    assert res.original_size == (100, 200)
    assert res.processed_size == (100, 200)


def test_blur_detection_checkerboard_vs_uniform() -> None:
    """Verify high-frequency checkerboard has high Laplacian variance while constant image has 0."""
    uniform = np.full((128, 128, 3), 120, dtype=np.uint8)
    checker = np.zeros((128, 128, 3), dtype=np.uint8)
    checker[::2, ::2] = 255
    checker[1::2, 1::2] = 255

    var_uniform = compute_blur_score(uniform)
    var_checker = compute_blur_score(checker)

    assert var_uniform == pytest.approx(0.0, abs=1e-6)
    assert var_checker > 10000.0

    res_blur = preprocess_image(uniform, blur_threshold=80.0)
    res_sharp = preprocess_image(checker, blur_threshold=80.0)
    assert res_blur.is_blurry is True
    assert res_sharp.is_blurry is False


def test_exposure_stats_hand_computed() -> None:
    """Verify hand-computed Rec.709 luminance and exposure classification."""
    # Rec.709 luminance: Y = 0.2126*R + 0.7152*G + 0.0722*B
    # For constant (R=0.5, G=0.5, B=0.5), Y = 0.5 everywhere, std = 0.0, dynamic_range = 0.0
    mid_gray = np.full((64, 64, 3), 0.5, dtype=np.float32)
    stats_mid = compute_exposure_stats(mid_gray)
    assert stats_mid.mean_luminance == pytest.approx(0.5, abs=1e-6)
    assert stats_mid.std_luminance == pytest.approx(0.0, abs=1e-6)
    assert stats_mid.dynamic_range == pytest.approx(0.0, abs=1e-6)
    assert stats_mid.status == "low_contrast"

    dark = np.full((64, 64, 3), 0.02, dtype=np.float32)
    stats_dark = compute_exposure_stats(dark)
    assert stats_dark.underexposed_fraction == pytest.approx(1.0, abs=1e-6)
    assert stats_dark.status == "underexposed"

    bright = np.full((64, 64, 3), 0.98, dtype=np.float32)
    stats_bright = compute_exposure_stats(bright)
    assert stats_bright.overexposed_fraction == pytest.approx(1.0, abs=1e-6)
    assert stats_bright.status == "overexposed"


def test_invalid_inputs_raise_typed_error() -> None:
    """Verify invalid image paths or dimensions raise ImagePreprocessingError."""
    with pytest.raises(ImagePreprocessingError):
        preprocess_image("non_existent_file_12345.jpg")
    with pytest.raises(ImagePreprocessingError):
        preprocess_image(np.zeros((10, 10), dtype=np.uint8))
