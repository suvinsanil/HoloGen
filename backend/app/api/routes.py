"""FastAPI REST endpoints for upload, reconstruction, status, depth visualization, export, and projects."""

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
from typing import List, Literal, Optional
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
import open3d as o3d
from PIL import Image, ImageOps
import torch

from app.core.config import get_settings
from app.core.logging import get_logger
from app.geometry.export import save_depth_visualization_png
from app.schemas.reconstruction import (
    HealthResponse,
    ProjectListResponse,
    ProjectStatusResponse,
    ReconstructRequest,
)
from app.services.database import DatabaseManager, ProjectRecord
from app.services.job_queue import (
    ReconstructionJobQueue,
    export_viewer_point_cloud_json,
)
from app.services.model_manager import model_manager
from app.vision.preprocessing import preprocess_image

logger = get_logger("api.routes")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB strict upload limit
MAX_RAW_DIMENSION = 4096  # 4096x4096 max raw pixel dimension guard
MAX_QUEUE_DEPTH = 10  # Cap pending queue depth to prevent resource exhaustion
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}
SAFE_PROJECT_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,80}$")

# Enforce a strict pixel ceiling matching MAX_RAW_DIMENSION against decompression bombs
Image.MAX_IMAGE_PIXELS = MAX_RAW_DIMENSION * MAX_RAW_DIMENSION

router = APIRouter(prefix="/api")

_db_instance: Optional[DatabaseManager] = None
_queue_instance: Optional[ReconstructionJobQueue] = None


def validate_project_id(project_id: str) -> str:
    """Validate project_id against path traversal and enforce containment inside outputs_dir."""
    if not project_id or not SAFE_PROJECT_ID_RE.fullmatch(project_id) or ".." in project_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid project_id format: only alphanumeric, '_', and '-' characters are permitted.",
        )
    settings = get_settings()
    base_dir = settings.outputs_dir.resolve()
    candidate = (base_dir / project_id).resolve()
    if not candidate.is_relative_to(base_dir) or candidate == base_dir:
        raise HTTPException(
            status_code=400,
            detail="Path traversal detected in project_id.",
        )
    return project_id


def get_db() -> DatabaseManager:
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance


def get_job_queue() -> ReconstructionJobQueue:
    global _queue_instance
    if _queue_instance is None:
        _queue_instance = ReconstructionJobQueue(get_db())
    return _queue_instance


def _record_to_status_response(rec: ProjectRecord) -> ProjectStatusResponse:
    q = get_job_queue()
    out_dir = Path(rec.output_dir)
    exports: List[str] = []
    export_map = {
        "glb": "scene.glb",
        "glb_full": "scene_full.glb",
        "obj": "scene.obj",
        "ply": "point_cloud.ply",
        "mesh_ply": "mesh.ply",
        "stl": "mesh.stl",
        "depth_png": "depth_visualization.png",
        "uncertainty_png": "uncertainty_proxy.png",
        "report_json": "reconstruction_report.json",
        "report_md": "reconstruction_report.md",
    }
    if out_dir.exists():
        for key, fname in export_map.items():
            if (out_dir / fname).exists():
                exports.append(key)

    params = {}
    if rec.parameters_json:
        try:
            params = json.loads(rec.parameters_json)
        except Exception:
            pass

    report = None
    if rec.report_json:
        try:
            report = json.loads(rec.report_json)
        except Exception:
            pass

    return ProjectStatusResponse(
        project_id=rec.project_id,
        name=rec.name,
        mode=rec.mode,
        status=rec.status,
        progress_pct=rec.progress_pct,
        stage_message=rec.stage_message,
        queue_position=q.get_queue_position(rec.project_id),
        created_at=rec.created_at,
        updated_at=rec.updated_at,
        error_message=rec.error_message,
        parameters=params,
        report=report,
        available_exports=exports,
    )


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    settings = get_settings()
    q = get_job_queue()
    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Only"
    )
    return HealthResponse(
        status="ok",
        device=settings.resolve_device(),
        gpu_name=gpu_name,
        loaded_model=model_manager.loaded_model_name,
        queue_length=q.queue_length,
        active_job_id=q.active_job_id,
        max_upload_mb=round(MAX_UPLOAD_BYTES / (1024.0 * 1024.0), 1),
        max_image_pixels=f"{MAX_RAW_DIMENSION}x{MAX_RAW_DIMENSION}",
    )


async def _validate_and_read_image(file: UploadFile) -> tuple[bytes, str, str]:
    """Validate uploaded image extension, content-type, byte size, and decoded dimensions."""
    filename = file.filename or "upload.png"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content-type '{file.content_type}'.",
        )

    raw_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img.verify()
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img.load()
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail="Corrupted or invalid image file."
        ) from exc

    orig_w, orig_h = pil_img.size
    if orig_w > MAX_RAW_DIMENSION or orig_h > MAX_RAW_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image resolution {orig_w}x{orig_h} exceeds safety limit "
                f"{MAX_RAW_DIMENSION}x{MAX_RAW_DIMENSION}."
            ),
        )

    # Strip EXIF/GPS/XMP/device-fingerprint metadata before persisting to disk
    fmt_map = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}
    save_fmt = fmt_map.get(ext, "PNG")
    if save_fmt == "JPEG" and pil_img.mode not in ("RGB", "L"):
        pil_img = pil_img.convert("RGB")
    clean_img = Image.new(pil_img.mode, pil_img.size)
    clean_img.paste(pil_img)
    buf = io.BytesIO()
    if save_fmt == "JPEG":
        clean_img.save(buf, format="JPEG", quality=95)
    elif save_fmt == "WEBP":
        clean_img.save(buf, format="WEBP", quality=95)
    else:
        clean_img.save(buf, format="PNG")
    return buf.getvalue(), ext, filename


@router.post("/upload", response_model=ProjectStatusResponse)
async def upload_image(
    file: UploadFile = File(...),
    name: Optional[str] = Form(default=None),
    mode: str = Form(default="single_view"),
) -> ProjectStatusResponse:
    """Validate uploaded image type, byte size, and pixel dimensions, then create a project."""
    raw_bytes, ext, filename = await _validate_and_read_image(file)

    settings = get_settings()
    # Security: Use purely server-generated project_id and filename on disk; never embed user filename in paths
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uid = uuid.uuid4().hex[:8]
    project_id = f"proj_{ts}_{short_uid}"
    validate_project_id(project_id)

    display_name = re.sub(r"[^a-zA-Z0-9_\-\s]", "", (name or Path(filename).stem)).strip()[:48]
    if not display_name:
        display_name = project_id

    out_dir = settings.outputs_dir / project_id
    out_dir.mkdir(parents=True, exist_ok=True)
    src_path = out_dir / f"input_original{ext}"
    src_path.write_bytes(raw_bytes)

    prep = preprocess_image(src_path, max_dimension=settings.max_image_dimension)
    Image.fromarray(prep.rgb_uint8, mode="RGB").save(out_dir / "input_preprocessed.png")

    initial_report = {
        "run_id": project_id,
        "preprocessing": prep.to_dict(),
    }

    db = get_db()
    rec = db.create_project(
        project_id=project_id,
        name=display_name,
        source_image_path=src_path,
        output_dir=out_dir,
        mode=mode if mode in ("single_view", "multi_view") else "single_view",
        initial_report=initial_report,
    )
    return _record_to_status_response(rec)


@router.post("/upload-multi", response_model=ProjectStatusResponse)
async def upload_multi_images(
    files: List[UploadFile] = File(...),
    name: Optional[str] = Form(default=None),
) -> ProjectStatusResponse:
    """Validate 2-40 uploaded images and create a multi-view reconstruction project."""
    if len(files) < 2 or len(files) > 40:
        raise HTTPException(
            status_code=400,
            detail=f"Multi-view upload requires 2 to 40 images, got {len(files)}.",
        )

    settings = get_settings()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uid = uuid.uuid4().hex[:8]
    project_id = f"proj_{ts}_{short_uid}"
    validate_project_id(project_id)

    out_dir = settings.outputs_dir / project_id
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for idx, f in enumerate(files):
        raw_bytes, ext, _ = await _validate_and_read_image(f)
        dst = inputs_dir / f"view_{idx:03d}{ext}"
        dst.write_bytes(raw_bytes)
        saved_paths.append(dst)

    # Preview thumbnail from first view
    prep = preprocess_image(saved_paths[0], max_dimension=settings.max_image_dimension)
    Image.fromarray(prep.rgb_uint8, mode="RGB").save(out_dir / "input_preprocessed.png")

    display_name = re.sub(r"[^a-zA-Z0-9_\-\s]", "", (name or f"MultiView ({len(saved_paths)} views)")).strip()[:48]
    if not display_name:
        display_name = project_id

    db = get_db()
    rec = db.create_project(
        project_id=project_id,
        name=display_name,
        source_image_path=saved_paths[0],
        output_dir=out_dir,
        mode="multi_view",
        initial_report={"run_id": project_id, "num_input_images": len(saved_paths)},
    )
    return _record_to_status_response(rec)


@router.post("/cancel/{project_id}", response_model=ProjectStatusResponse)
def cancel_reconstruction(project_id: str) -> ProjectStatusResponse:
    """Cancel a queued or running reconstruction job."""
    validate_project_id(project_id)
    db = get_db()
    rec = db.get_project(project_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    q = get_job_queue()
    q.cancel(project_id)
    updated = db.get_project(project_id)
    return _record_to_status_response(updated or rec)


@router.post("/reconstruct/{project_id}", response_model=ProjectStatusResponse)
def start_reconstruction(
    project_id: str,
    request: ReconstructRequest,
) -> ProjectStatusResponse:
    """Enqueue a project for 3D reconstruction in the concurrency-1 worker queue."""
    validate_project_id(project_id)
    db = get_db()
    rec = db.get_project(project_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    q = get_job_queue()
    if q.is_active_or_queued(project_id):
        raise HTTPException(
            status_code=409,
            detail=f"Project '{project_id}' is already queued or running.",
        )
    if q.queue_depth() >= MAX_QUEUE_DEPTH:
        raise HTTPException(
            status_code=429,
            detail=f"Reconstruction queue is full (max {MAX_QUEUE_DEPTH} pending jobs). Try again later.",
        )
    q.enqueue(
        project_id=project_id,
        params=request,
        wait_for_completion=request.run_synchronously,
    )
    updated = db.get_project(project_id)
    assert updated is not None
    return _record_to_status_response(updated)


@router.get("/status/{project_id}", response_model=ProjectStatusResponse)
def get_project_status(project_id: str) -> ProjectStatusResponse:
    """Fetch live status, progress, and reconstruction report for a project."""
    validate_project_id(project_id)
    db = get_db()
    rec = db.get_project(project_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    return _record_to_status_response(rec)


@router.get("/projects", response_model=ProjectListResponse)
def list_all_projects() -> ProjectListResponse:
    """List all reconstruction projects in SQLite."""
    db = get_db()
    records = db.list_projects()
    return ProjectListResponse(projects=[_record_to_status_response(r) for r in records])


@router.get("/depth/{project_id}")
def get_depth_image(
    project_id: str,
    mode: Literal["heatmap", "grayscale", "inverted", "uncertainty_proxy", "input"] = Query(
        default="heatmap"
    ),
) -> FileResponse:
    """Return depth visualization, uncertainty proxy map, or preprocessed input PNG."""
    validate_project_id(project_id)
    db = get_db()
    rec = db.get_project(project_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    out_dir = Path(rec.output_dir)
    if mode == "input":
        p = out_dir / "input_preprocessed.png"
        if p.exists():
            return FileResponse(p, media_type="image/png")
        raise HTTPException(status_code=404, detail="Preprocessed input image not found.")

    if mode == "uncertainty_proxy":
        p = out_dir / "uncertainty_proxy.png"
        if p.exists():
            return FileResponse(p, media_type="image/png")
        raise HTTPException(status_code=404, detail="Uncertainty proxy visualization not found.")

    if mode == "heatmap":
        p = out_dir / "depth_visualization.png"
        if p.exists():
            return FileResponse(p, media_type="image/png")

    # For grayscale or inverted, render on-the-fly from depth_visualization if needed
    base_depth_png = out_dir / "depth_visualization.png"
    if not base_depth_png.exists():
        raise HTTPException(status_code=404, detail="Depth visualization not yet generated.")

    custom_png = out_dir / f"depth_{mode}.png"
    if not custom_png.exists():
        gray_arr = np.asarray(Image.open(base_depth_png).convert("L"), dtype=np.float32) / 255.0
        save_depth_visualization_png(gray_arr, custom_png, mode=mode)
    return FileResponse(custom_png, media_type="image/png")


@router.get("/projects/{project_id}/point-cloud-json")
def get_viewer_point_cloud_json(project_id: str) -> FileResponse:
    """Return subsampled 3D point cloud JSON (positions, colors, normals, depth, uncertainty)."""
    validate_project_id(project_id)
    db = get_db()
    rec = db.get_project(project_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    out_dir = Path(rec.output_dir)
    json_path = out_dir / "viewer_cloud.json"
    if not json_path.exists():
        ply_path = out_dir / "point_cloud.ply"
        if not ply_path.exists():
            raise HTTPException(status_code=404, detail="Point cloud not yet generated.")
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = np.asarray(pcd.points, dtype=np.float32)
        cols = (
            np.asarray(pcd.colors, dtype=np.float32)
            if pcd.has_colors()
            else np.full_like(pts, 0.7)
        )
        norms = (
            np.asarray(pcd.normals, dtype=np.float32)
            if pcd.has_normals()
            else None
        )
        unc_png = out_dir / "uncertainty_proxy.png"
        unc_map = None
        if unc_png.exists():
            unc_map = np.asarray(Image.open(unc_png).convert("L"), dtype=np.float32) / 255.0
        export_viewer_point_cloud_json(
            points=pts,
            colors=cols,
            normals=norms,
            uncertainty_map=unc_map,
            pixel_coords=None,
            output_path=json_path,
        )
    return FileResponse(json_path, media_type="application/json")


@router.get("/export/{project_id}/{format_key}")
def download_export_asset(project_id: str, format_key: str) -> FileResponse:
    """Download an exported 3D asset or reconstruction report."""
    validate_project_id(project_id)
    db = get_db()
    rec = db.get_project(project_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    out_dir = Path(rec.output_dir)
    file_mapping = {
        "glb": ("scene.glb", "model/gltf-binary", f"{project_id}.glb"),
        "glb_full": ("scene_full.glb", "model/gltf-binary", f"{project_id}_full.glb"),
        "obj": ("scene.obj", "text/plain", f"{project_id}.obj"),
        "ply": ("point_cloud.ply", "application/octet-stream", f"{project_id}_point_cloud.ply"),
        "mesh_ply": ("mesh.ply", "application/octet-stream", f"{project_id}_mesh.ply"),
        "stl": ("mesh.stl", "model/stl", f"{project_id}.stl"),
        "depth_png": ("depth_visualization.png", "image/png", f"{project_id}_depth.png"),
        "uncertainty_png": (
            "uncertainty_proxy.png",
            "image/png",
            f"{project_id}_uncertainty_proxy.png",
        ),
        "report_json": (
            "reconstruction_report.json",
            "application/json",
            f"{project_id}_report.json",
        ),
        "report_md": (
            "reconstruction_report.md",
            "text/markdown",
            f"{project_id}_report.md",
        ),
    }
    if format_key not in file_mapping:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown export format '{format_key}'. Valid: {sorted(file_mapping.keys())}",
        )

    fname, media_type, download_name = file_mapping[format_key]
    target_path = out_dir / fname
    if not target_path.exists() and format_key == "glb_full":
        target_path = out_dir / "scene.glb"
    if not target_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Asset '{fname}' not found for project '{project_id}'."
        )

    return FileResponse(
        path=target_path,
        media_type=media_type,
        filename=download_name,
    )
