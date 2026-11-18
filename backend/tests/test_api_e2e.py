"""End-to-end and security tests for Phase 6 / 6.5 FastAPI application:
- Upload validation, generated filenames only (no user path injection),
- project_id path traversal prevention,
- localhost-only CORS restriction,
- Full E2E upload -> reconstruct -> status -> export GLB flow.
"""

import io
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.api.routes import validate_project_id
from app.core.config import get_settings
from app.main import app


def test_generated_filenames_only_and_path_traversal_guards() -> None:
    """Verify user filenames are never used in filesystem paths and traversal project_ids fail."""
    settings = get_settings()
    with TestClient(app) as client:
        # 1. Upload with a malicious path-traversal filename
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        malicious_filename = "../../evil_folder/../../pwned_payload.png"
        resp = client.post(
            "/api/upload",
            files={"file": (malicious_filename, buf.getvalue(), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        project_id = data["project_id"]

        # project_id must be purely server-generated ('proj_<ts>_<hex8>') and contain no user filename parts
        assert project_id.startswith("proj_")
        assert "evil" not in project_id
        assert "pwned" not in project_id
        assert ".." not in project_id

        # Verify on-disk directory is strictly inside outputs_dir and uses generated 'input_original.png'
        proj_dir = (settings.outputs_dir / project_id).resolve()
        assert proj_dir.is_relative_to(settings.outputs_dir.resolve())
        assert (proj_dir / "input_original.png").exists()
        assert not (proj_dir / "pwned_payload.png").exists()

        # 2. Verify validate_project_id rejects traversal and special characters directly and via HTTP
        for bad_id in [
            "../secret",
            "..\\Windows",
            "proj_123/../../etc",
            "proj..123",
            "proj$bad",
            "proj;rm",
        ]:
            with pytest.raises(Exception):
                validate_project_id(bad_id)

        for endpoint in [
            "/api/status/..%5C..%5CWindows",
            "/api/status/proj..evil",
            "/api/depth/proj..evil",
            "/api/export/proj..evil/glb",
        ]:
            r = client.get(endpoint)
            assert r.status_code in (400, 404, 422)

        # 3. Verify CORS is restricted to localhost origins only
        ok_cors = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert ok_cors.headers.get("access-control-allow-origin") == "http://localhost:5173"

        bad_cors = client.options(
            "/api/health",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert bad_cors.headers.get("access-control-allow-origin") is None


def test_api_upload_validation_and_e2e_reconstruct_export() -> None:
    """Verify upload limits/validation and full upload -> reconstruct -> status -> export GLB flow."""
    with TestClient(app) as client:
        # 1. Health check verifies singleton model preloaded
        health = client.get("/api/health")
        assert health.status_code == 200
        h_data = health.json()
        assert h_data["status"] == "ok"
        assert h_data["loaded_model"] == "depth_anything_v2_small"

        # 2. Reject invalid file extension (.txt)
        bad_ext = client.post(
            "/api/upload",
            files={"file": ("malicious.txt", b"not an image", "text/plain")},
        )
        assert bad_ext.status_code == 400

        # 3. Reject corrupted image payload with .png extension
        bad_img = client.post(
            "/api/upload",
            files={"file": ("broken.png", b"corrupted_bytes_12345", "image/png")},
        )
        assert bad_img.status_code == 400

        # 4. Upload a valid synthetic test scene (128x96 RGB image)
        img = Image.new("RGB", (128, 96), color=(90, 140, 210))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        up_resp = client.post(
            "/api/upload",
            files={"file": ("e2e_test_scene.png", buf.getvalue(), "image/png")},
            data={"name": "E2E Test Scene", "mode": "single_view"},
        )
        assert up_resp.status_code == 200
        up_json = up_resp.json()
        project_id = up_json["project_id"]
        assert project_id.startswith("proj_")
        assert up_json["status"] == "uploaded"

        # 5. Trigger reconstruction synchronously via the concurrency-1 worker queue
        rec_resp = client.post(
            f"/api/reconstruct/{project_id}",
            json={
                "mode": "single_view",
                "model_name": "depth_anything_v2_small",
                "max_dimension": 128,
                "fov_deg": 60.0,
                "refinement_method": "none",
                "mesh_method": "depth_grid",
                "quality_preset": "viewer",
                "viewer_target_faces": 20000,
                "smoothing_iterations": 2,
                "uncertainty_threshold": 0.75,
                "discontinuity_threshold": 0.08,
                "run_synchronously": True,
            },
        )
        assert rec_resp.status_code == 200
        rec_json = rec_resp.json()
        assert rec_json["status"] == "completed"
        assert rec_json["progress_pct"] == 100
        assert "glb" in rec_json["available_exports"]
        assert rec_json["report"]["estimated_coverage"]["view_frustum_coverage_pct"] > 50.0

        # 6. Fetch depth heatmap and uncertainty proxy PNGs
        depth_resp = client.get(f"/api/depth/{project_id}?mode=heatmap")
        assert depth_resp.status_code == 200
        assert depth_resp.headers["content-type"] == "image/png"

        unc_resp = client.get(f"/api/depth/{project_id}?mode=uncertainty_proxy")
        assert unc_resp.status_code == 200

        # 7. Fetch interactive point cloud JSON for React Three Fiber viewer
        pc_resp = client.get(f"/api/projects/{project_id}/point-cloud-json")
        assert pc_resp.status_code == 200
        pc_data = pc_resp.json()
        assert pc_data["count"] > 100
        assert len(pc_data["positions"]) == pc_data["count"] * 3

        # 8. Download exported GLB and verify binary glTF magic header
        glb_resp = client.get(f"/api/export/{project_id}/glb")
        assert glb_resp.status_code == 200
        assert glb_resp.content[:4] == b"glTF"
        assert len(glb_resp.content) > 500
