"""End-to-end tests for Phase 6 FastAPI application: upload validation, queue, status, and GLB export."""

import io
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


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
