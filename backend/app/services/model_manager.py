"""Singleton depth model manager ensuring weights are loaded once and never duplicated per request."""

import threading
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.vision.depth_estimation import DepthModel, create_depth_model

logger = get_logger("services.model_manager")


class DepthModelSingletonManager:
    """Thread-safe singleton cache keeping at most one depth model in GPU/CPU memory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_model: Optional[DepthModel] = None
        self._current_model_name: Optional[str] = None

    @property
    def loaded_model_name(self) -> Optional[str]:
        return self._current_model_name

    def get_or_load_model(self, model_name: Optional[str] = None) -> DepthModel:
        """Return the already-loaded DepthModel if matching, or swap cleanly under a lock."""
        settings = get_settings()
        target_name = (model_name or settings.default_depth_model).strip().lower()

        with self._lock:
            if self._current_model is not None and self._current_model_name == target_name:
                return self._current_model

            if self._current_model is not None:
                logger.info(
                    "Unloading previous depth model '%s' before loading '%s' (2GB VRAM guard)",
                    self._current_model_name,
                    target_name,
                )
                self._current_model.unload()
                self._current_model = None
                self._current_model_name = None

            logger.info("Singleton loading depth model '%s'", target_name)
            model = create_depth_model(
                model_name=target_name,
                device=settings.resolve_device(),
            )
            model.load()
            self._current_model = model
            self._current_model_name = target_name
            return model


model_manager = DepthModelSingletonManager()
