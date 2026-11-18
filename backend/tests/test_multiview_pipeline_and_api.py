"""Tests for Phase 7d: MultiViewPipeline (decimation + full-res + camera frustums + cancel/timeout)
and FastAPI multi-view endpoints (/api/upload-multi, /api/cancel/{project_id}).
"""

import io
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from app.core.config import get_settings
from app.evaluation.multiview_datasets import load_fountain_dataset
from app.main import app
from app.pipelines.multi_view import JobCancelledError, JobTimeoutError, MultiViewPipeline


def _make_png_bytes(w: int = 120, h: int = 90) -> bytes:
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_multiview_upload_and_cancel_endpoints(tmp_path) -> None:
    """Verify /api/upload-multi validation, multi-view project creation, and /api/cancel."""
    client = TestClient(app)

    # Reject < 2 files
    r_bad = client.post(
        "/api/upload-multi",
        files=[("files", ("v0.png", _make_png_bytes(), "image/png"))],
    )
    assert r_bad.status_code == 400

    # Accept 3 valid images
    r_ok = client.post(
        "/api/upload-multi",
        files=[
            ("files", ("v0.png", _make_png_bytes(), "image/png")),
            ("files", ("v1.png", _make_png_bytes(), "image/png")),
            ("files", ("v2.png", _make_png_bytes(), "image/png")),
        ],
    )
    assert r_ok.status_code == 200
    proj = r_ok.json()
    assert proj["mode"] == "multi_view"
    pid = proj["project_id"]

    # Cancel endpoint works cleanly
    r_cancel = client.post(f"/api/cancel/{pid}")
    assert r_cancel.status_code == 200


def test_multiview_pipeline_cancel_and_timeout(tmp_path) -> None:
    """Verify cooperative cancellation and hard timeout raise typed errors immediately."""
    settings = get_settings()
    views, _ = load_fountain_dataset(settings.datasets_dir)
    img_paths = [v.image_path for v in views[:3]]
    pipe = MultiViewPipeline(settings=settings)

    cancel_evt = threading.Event()
    cancel_evt.set()
    with pytest.raises(JobCancelledError):
        pipe.run(image_paths=img_paths, run_id="test_mv_cancel", output_base_dir=tmp_path, cancel_event=cancel_evt)

    with pytest.raises(JobTimeoutError):
        pipe.run(image_paths=img_paths, run_id="test_mv_timeout", output_base_dir=tmp_path, timeout_seconds=1e-6)


def test_multiview_pipeline_e2e_fountain_4views(tmp_path) -> None:
    """Run MultiViewPipeline end-to-end on 4 Fountain views and verify viewer GLB, full GLB, frustums, and RSS."""
    settings = get_settings()
    views, _ = load_fountain_dataset(settings.datasets_dir)
    img_paths = [v.image_path for v in views[:4]]

    stages = []
    pipe = MultiViewPipeline(settings=settings)
    res = pipe.run(
        image_paths=img_paths,
        run_id="test_mv_fountain_4",
        output_base_dir=tmp_path,
        viewer_target_faces=120000,
        progress_callback=lambda s, p, m: stages.append(s),
    )

    assert "running_sfm" in stages
    assert "running_depth_alignment" in stages
    assert "running_fusion" in stages
    assert "running_export" in stages
    assert res.viewer_mesh.num_faces <= 120000
    assert (res.output_dir / "scene.glb").exists()
    assert (res.output_dir / "scene_full.glb").exists()
    assert (res.output_dir / "reconstruction_report.json").exists()
    assert len(res.camera_frustums) == res.sfm_result.num_registered_images
    assert res.whole_pipeline_peak_rss_mb > 100.0
