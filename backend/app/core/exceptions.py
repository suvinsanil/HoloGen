"""Custom typed exceptions for the HoloGen reconstruction pipeline."""


class HoloGenError(Exception):
    """Base exception for all HoloGen errors."""


class ConfigurationError(HoloGenError):
    """Raised when configuration or hardware settings are invalid."""


class ImagePreprocessingError(HoloGenError):
    """Raised when an input image cannot be loaded, decoded, or preprocessed."""


class IntrinsicsEstimationError(HoloGenError):
    """Raised when camera intrinsics parameters are invalid or non-physical."""


class DepthEstimationError(HoloGenError):
    """Raised when depth model loading or inference fails."""


class GeometryProcessingError(HoloGenError):
    """Raised when point cloud or mesh reconstruction fails."""


class ExportError(HoloGenError):
    """Raised when saving 3D assets or reports fails."""


class JobCancelledError(HoloGenError, RuntimeError):
    """Raised when a reconstruction job is cancelled by the user."""


class JobTimeoutError(HoloGenError, RuntimeError):
    """Raised when a reconstruction job exceeds its hard wall-clock timeout."""


class QueueFullError(HoloGenError):
    """Raised when the reconstruction queue depth limit has been reached."""


class DuplicateJobError(HoloGenError):
    """Raised when a reconstruction job is already active or queued."""
