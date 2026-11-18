"""Image preprocessing, EXIF extraction, normalization, and quality checks."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import time

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps

from app.core.exceptions import ImagePreprocessingError


def _rational_to_float(val: Any) -> Optional[float]:
    """Convert PIL EXIF IFDRational or tuple to float safely."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if hasattr(val, "numerator") and hasattr(val, "denominator"):
        if val.denominator == 0:
            return None
        return float(val.numerator) / float(val.denominator)
    if isinstance(val, (tuple, list)) and len(val) == 2:
        num, den = val
        if den == 0:
            return None
        return float(num) / float(den)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def extract_exif_metadata(img: Image.Image) -> Dict[str, Any]:
    """Extract relevant camera and orientation EXIF tags from a PIL Image."""
    metadata: Dict[str, Any] = {
        "orientation": None,
        "focal_length_mm": None,
        "focal_length_35mm": None,
        "camera_make": None,
        "camera_model": None,
    }
    try:
        exif = img.getexif()
        if not exif:
            return metadata

        tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        # Also inspect Exif IFD (0x8769) where FocalLength and FocalLengthIn35mmFilm often live
        if hasattr(exif, "get_ifd"):
            try:
                exif_ifd = exif.get_ifd(0x8769)
                for k, v in exif_ifd.items():
                    tag_map[ExifTags.TAGS.get(k, k)] = v
            except Exception:
                pass

        if "Orientation" in tag_map:
            try:
                metadata["orientation"] = int(tag_map["Orientation"])
            except (TypeError, ValueError):
                pass

        if "FocalLength" in tag_map:
            metadata["focal_length_mm"] = _rational_to_float(tag_map["FocalLength"])

        if "FocalLengthIn35mmFilm" in tag_map:
            metadata["focal_length_35mm"] = _rational_to_float(tag_map["FocalLengthIn35mmFilm"])

        if "Make" in tag_map and tag_map["Make"]:
            metadata["camera_make"] = str(tag_map["Make"]).strip("\x00").strip()

        if "Model" in tag_map and tag_map["Model"]:
            metadata["camera_model"] = str(tag_map["Model"]).strip("\x00").strip()

    except Exception:
        pass
    return metadata


@dataclass
class ExposureStats:
    """Luminance and exposure statistics for an RGB image."""

    mean_luminance: float
    std_luminance: float
    p5_luminance: float
    p95_luminance: float
    dynamic_range: float
    underexposed_fraction: float
    overexposed_fraction: float
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_luminance": round(self.mean_luminance, 4),
            "std_luminance": round(self.std_luminance, 4),
            "p5_luminance": round(self.p5_luminance, 4),
            "p95_luminance": round(self.p95_luminance, 4),
            "dynamic_range": round(self.dynamic_range, 4),
            "underexposed_fraction": round(self.underexposed_fraction, 4),
            "overexposed_fraction": round(self.overexposed_fraction, 4),
            "status": self.status,
        }


@dataclass
class PreprocessingResult:
    """Structured output from the image preprocessing and quality pipeline."""

    rgb_uint8: np.ndarray
    rgb_normalized: np.ndarray
    original_size: Tuple[int, int]  # (width, height)
    processed_size: Tuple[int, int]  # (width, height)
    resize_scale: float
    exif_orientation_applied: bool
    exif_metadata: Dict[str, Any]
    blur_laplacian_variance: float
    is_blurry: bool
    exposure: ExposureStats
    processing_time_ms: float
    source_path: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def width(self) -> int:
        return self.processed_size[0]

    @property
    def height(self) -> int:
        return self.processed_size[1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "original_size": {"width": self.original_size[0], "height": self.original_size[1]},
            "processed_size": {"width": self.processed_size[0], "height": self.processed_size[1]},
            "resize_scale": round(self.resize_scale, 6),
            "exif_orientation_applied": self.exif_orientation_applied,
            "exif_metadata": self.exif_metadata,
            "quality": {
                "blur_laplacian_variance": round(self.blur_laplacian_variance, 4),
                "is_blurry": self.is_blurry,
                "exposure": self.exposure.to_dict(),
                "warnings": self.warnings,
            },
            "processing_time_ms": round(self.processing_time_ms, 3),
        }


def compute_blur_score(rgb_uint8: np.ndarray) -> float:
    """Compute variance of the Laplacian on grayscale image (higher = sharper)."""
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ImagePreprocessingError("Expected HxWx3 RGB uint8 image for blur computation.")
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_exposure_stats(
    rgb_normalized: np.ndarray,
    under_thresh: float = 0.05,
    over_thresh: float = 0.95,
) -> ExposureStats:
    """Compute Rec.709 luminance statistics and exposure classification."""
    if rgb_normalized.ndim != 3 or rgb_normalized.shape[2] != 3:
        raise ImagePreprocessingError("Expected HxWx3 float RGB image in [0, 1].")
    luminance = (
        0.2126 * rgb_normalized[:, :, 0]
        + 0.7152 * rgb_normalized[:, :, 1]
        + 0.0722 * rgb_normalized[:, :, 2]
    )
    mean_lum = float(np.mean(luminance))
    std_lum = float(np.std(luminance))
    p5_lum, p95_lum = [float(v) for v in np.percentile(luminance, [5.0, 95.0])]
    dyn_range = p95_lum - p5_lum
    under_frac = float(np.mean(luminance < under_thresh))
    over_frac = float(np.mean(luminance > over_thresh))

    if under_frac > 0.40 or mean_lum < 0.12:
        status = "underexposed"
    elif over_frac > 0.40 or mean_lum > 0.88:
        status = "overexposed"
    elif dyn_range < 0.15:
        status = "low_contrast"
    else:
        status = "normal"

    return ExposureStats(
        mean_luminance=mean_lum,
        std_luminance=std_lum,
        p5_luminance=p5_lum,
        p95_luminance=p95_lum,
        dynamic_range=dyn_range,
        underexposed_fraction=under_frac,
        overexposed_fraction=over_frac,
        status=status,
    )


def preprocess_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    max_dimension: int = 640,
    blur_threshold: float = 80.0,
    underexposure_threshold: float = 0.05,
    overexposure_threshold: float = 0.95,
) -> PreprocessingResult:
    """Load, orient via EXIF, aspect-resize, normalize, and check image quality."""
    t0 = time.perf_counter()
    if max_dimension < 16:
        raise ImagePreprocessingError(f"max_dimension must be >= 16, got {max_dimension}")

    source_path: Optional[str] = None
    exif_metadata: Dict[str, Any] = {
        "orientation": None,
        "focal_length_mm": None,
        "focal_length_35mm": None,
        "camera_make": None,
        "camera_model": None,
    }
    exif_orientation_applied = False

    if isinstance(image_input, (str, Path)):
        path_obj = Path(image_input)
        if not path_obj.exists():
            raise ImagePreprocessingError(f"Image file does not exist: {path_obj}")
        source_path = str(path_obj.resolve())
        try:
            pil_img = Image.open(path_obj)
            pil_img.load()
        except Exception as exc:
            raise ImagePreprocessingError(f"Failed to decode image {path_obj}: {exc}") from exc
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.copy()
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim != 3 or image_input.shape[2] not in (3, 4):
            raise ImagePreprocessingError(
                f"Expected HxWx3 or HxWx4 numpy array, got shape {image_input.shape}"
            )
        arr = image_input
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr[:, :, :3], mode="RGB")
    else:
        raise ImagePreprocessingError(f"Unsupported image input type: {type(image_input)}")

    # Extract EXIF before transpose (transpose strips the Orientation tag)
    if isinstance(image_input, (str, Path, Image.Image)):
        exif_metadata = extract_exif_metadata(pil_img)
        orientation_tag = exif_metadata.get("orientation")
        if orientation_tag is not None and orientation_tag != 1:
            pil_img = ImageOps.exif_transpose(pil_img)
            exif_orientation_applied = True

    pil_img = pil_img.convert("RGB")
    orig_w, orig_h = pil_img.size
    if orig_w < 2 or orig_h < 2:
        raise ImagePreprocessingError(f"Image dimensions too small: {orig_w}x{orig_h}")

    # Aspect-preserving resize if larger than max_dimension
    max_orig = max(orig_w, orig_h)
    if max_orig > max_dimension:
        resize_scale = float(max_dimension) / float(max_orig)
        new_w = max(2, int(round(orig_w * resize_scale)))
        new_h = max(2, int(round(orig_h * resize_scale)))
        pil_img = pil_img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    else:
        resize_scale = 1.0
        new_w, new_h = orig_w, orig_h

    rgb_uint8 = np.asarray(pil_img, dtype=np.uint8)
    rgb_normalized = rgb_uint8.astype(np.float32) / 255.0

    blur_var = compute_blur_score(rgb_uint8)
    is_blurry = bool(blur_var < blur_threshold)

    exposure = compute_exposure_stats(
        rgb_normalized,
        under_thresh=underexposure_threshold,
        over_thresh=overexposure_threshold,
    )

    warnings: list[str] = []
    if is_blurry:
        warnings.append(
            f"Image appears blurry (Laplacian variance {blur_var:.1f} < threshold {blur_threshold:.1f})."
        )
    if exposure.status != "normal":
        warnings.append(f"Image exposure flagged as '{exposure.status}'.")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return PreprocessingResult(
        rgb_uint8=rgb_uint8,
        rgb_normalized=rgb_normalized,
        original_size=(orig_w, orig_h),
        processed_size=(new_w, new_h),
        resize_scale=resize_scale,
        exif_orientation_applied=exif_orientation_applied,
        exif_metadata=exif_metadata,
        blur_laplacian_variance=blur_var,
        is_blurry=is_blurry,
        exposure=exposure,
        processing_time_ms=elapsed_ms,
        source_path=source_path,
        warnings=warnings,
    )
