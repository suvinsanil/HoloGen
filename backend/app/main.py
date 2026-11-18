"""FastAPI application entry point with singleton depth model startup preloading."""

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import get_db, get_job_queue, router
from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.model_manager import model_manager

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize SQLite database, job queue worker, and preload the singleton depth model once."""
    settings = get_settings()
    settings.configure_environment()
    _ = get_db()
    _ = get_job_queue()
    if os.environ.get("HOLOGEN_SKIP_STARTUP_MODEL_LOAD", "0") != "1":
        logger.info("Preloading singleton depth model '%s' at startup...", settings.default_depth_model)
        model_manager.get_or_load_model(settings.default_depth_model)
    yield


app = FastAPI(
    title="HoloGen: AI-Powered 3D Scene Reconstruction Platform",
    description="From pixels to spatial intelligence — Single-view & Multi-view 3D Reconstruction API",
    version="0.6.0",
    lifespan=lifespan,
)

LOCALHOST_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=LOCALHOST_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(router)

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
