"""Concurrency-1 reconstruction job queue with granular stage status reporting."""

import json
from pathlib import Path
import queue
import threading
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.pipelines.single_view import SingleViewPipeline
from app.schemas.reconstruction import ReconstructRequest
from app.services.database import DatabaseManager
from app.services.model_manager import model_manager

logger = get_logger("services.job_queue")


def export_viewer_point_cloud_json(
    points: np.ndarray,
    colors: np.ndarray,
    normals: Optional[np.ndarray],
    uncertainty_map: Optional[np.ndarray],
    pixel_coords: Optional[np.ndarray],
    output_path: Path,
    max_viewer_points: int = 25000,
) -> Path:
    """Write a compact JSON point cloud payload (in glTF Y-up coordinates) for interactive R3F viewing.

    Includes positions, RGB colors, surface normals, depth values, and uncertainty proxy values
    per point so the web viewer can switch across RGB, Depth, Point Cloud, Normals, and
    Uncertainty Proxy modes at 60 FPS.
    """
    n_total = int(points.shape[0])
    if n_total == 0:
        payload = {"count": 0, "positions": [], "colors": [], "normals": [], "depths": [], "uncertainty": []}
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return output_path

    step = max(1, n_total // max_viewer_points)
    idx = np.arange(0, n_total, step, dtype=np.int64)

    pts_sub = points[idx].copy()
    depths_sub = pts_sub[:, 2].copy()
    # Convert OpenCV (+X right, +Y down, +Z forward) to glTF (+X right, +Y up, -Z forward)
    pts_sub[:, 1] = -pts_sub[:, 1]
    pts_sub[:, 2] = -pts_sub[:, 2]

    cols_sub = np.clip(colors[idx], 0.0, 1.0)
    if normals is not None and len(normals) == n_total:
        norms_sub = normals[idx].copy()
        norms_sub[:, 1] = -norms_sub[:, 1]
        norms_sub[:, 2] = -norms_sub[:, 2]
    else:
        norms_sub = np.zeros_like(pts_sub)
        norms_sub[:, 2] = 1.0

    if (
        uncertainty_map is not None
        and pixel_coords is not None
        and len(pixel_coords) == n_total
    ):
        uv_sub = pixel_coords[idx]
        h, w = uncertainty_map.shape
        u_c = np.clip(uv_sub[:, 0], 0, w - 1)
        v_c = np.clip(uv_sub[:, 1], 0, h - 1)
        unc_sub = uncertainty_map[v_c, u_c]
    else:
        unc_sub = np.zeros(len(idx), dtype=np.float32)

    payload = {
        "count": int(len(idx)),
        "positions": np.round(pts_sub, 4).ravel().tolist(),
        "colors": np.round(cols_sub, 3).ravel().tolist(),
        "normals": np.round(norms_sub, 3).ravel().tolist(),
        "depths": np.round(depths_sub, 4).tolist(),
        "uncertainty": np.round(unc_sub, 3).tolist(),
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    return output_path


class ReconstructionJobQueue:
    """Single-worker (concurrency = 1) job queue protecting 2GB VRAM and 7.7GB RAM."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self._queue: "queue.Queue[tuple[str, ReconstructRequest, Optional[threading.Event]]]" = (
            queue.Queue()
        )
        self._queued_ids: List[str] = []
        self._active_job_id: Optional[str] = None
        self._lock = threading.Lock()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="HoloGen-Worker-1", daemon=True
        )
        self._worker_thread.start()

    @property
    def active_job_id(self) -> Optional[str]:
        with self._lock:
            return self._active_job_id

    @property
    def queue_length(self) -> int:
        with self._lock:
            return len(self._queued_ids)

    def get_queue_position(self, project_id: str) -> int:
        with self._lock:
            if self._active_job_id == project_id:
                return 0
            if project_id in self._queued_ids:
                return self._queued_ids.index(project_id) + 1
            return 0

    def enqueue(
        self,
        project_id: str,
        params: ReconstructRequest,
        wait_for_completion: bool = False,
    ) -> int:
        """Enqueue a reconstruction job and optionally wait for completion."""
        done_event = threading.Event() if wait_for_completion else None
        with self._lock:
            if project_id not in self._queued_ids and self._active_job_id != project_id:
                self._queued_ids.append(project_id)
            pos = (
                self._queued_ids.index(project_id) + 1
                if project_id in self._queued_ids
                else 0
            )

        self.db.update_project_status(
            project_id=project_id,
            status="queued",
            progress_pct=10,
            stage_message=f"Queued for execution (position {pos}, concurrency=1)",
            parameters=params.model_dump(),
            error_message=None,
        )
        self._queue.put((project_id, params, done_event))

        if done_event is not None:
            done_event.wait()
        return pos

    def _worker_loop(self) -> None:
        while True:
            project_id, params, done_event = self._queue.get()
            with self._lock:
                if project_id in self._queued_ids:
                    self._queued_ids.remove(project_id)
                self._active_job_id = project_id

            try:
                self._execute_job(project_id, params)
            except Exception as exc:
                logger.error("Job %s failed: %s\n%s", project_id, exc, traceback.format_exc())
                self.db.update_project_status(
                    project_id=project_id,
                    status="failed",
                    progress_pct=0,
                    stage_message=f"Reconstruction failed: {exc}",
                    error_message=str(exc),
                )
            finally:
                with self._lock:
                    self._active_job_id = None
                if done_event is not None:
                    done_event.set()
                self._queue.task_done()

    def _execute_job(self, project_id: str, params: ReconstructRequest) -> None:
        rec = self.db.get_project(project_id)
        if rec is None:
            raise RuntimeError(f"Project {project_id} not found in database.")

        settings = get_settings()
        src_path = Path(rec.source_image_path)
        out_dir = Path(rec.output_dir)

        self.db.update_project_status(
            project_id=project_id,
            status="estimating_depth",
            progress_pct=30,
            stage_message=f"Running depth inference ({params.model_name}) & geometry pipeline...",
            parameters=params.model_dump(),
        )

        # Retrieve singleton loaded model (never loads per-request if already loaded)
        depth_model = model_manager.get_or_load_model(params.model_name)
        pipeline = SingleViewPipeline(settings=settings, depth_model=depth_model)

        result = pipeline.run(
            image_input=src_path,
            run_id=project_id,
            max_dimension=params.max_dimension,
            fov_deg=params.fov_deg,
            refinement_method=params.refinement_method,
            mesh_method=params.mesh_method,
            quality_preset=params.quality_preset,
            viewer_target_faces=params.viewer_target_faces,
            export_both_presets=True,
            discontinuity_threshold=params.discontinuity_threshold,
            uncertainty_threshold=params.uncertainty_threshold,
            smoothing_iterations=params.smoothing_iterations,
            output_base_dir=out_dir.parent,
        )

        self.db.update_project_status(
            project_id=project_id,
            status="exporting",
            progress_pct=90,
            stage_message="Generating interactive viewer point cloud & finalizing exports...",
        )

        export_viewer_point_cloud_json(
            points=result.point_cloud.points,
            colors=result.point_cloud.colors,
            normals=result.point_cloud.normals,
            uncertainty_map=result.uncertainty_proxy.uncertainty_proxy_map,
            pixel_coords=result.point_cloud.pixel_coords,
            output_path=result.output_dir / "viewer_cloud.json",
        )

        self.db.update_project_status(
            project_id=project_id,
            status="completed",
            progress_pct=100,
            stage_message="Reconstruction complete",
            parameters=params.model_dump(),
            report=result.report_data,
            error_message=None,
        )
