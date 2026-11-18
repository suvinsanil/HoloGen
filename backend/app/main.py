"""FastAPI application entry point with singleton depth model startup preloading."""

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

ALLOWED_LOCAL_HOSTS = {
    "localhost",
    "127.0.0.1",
    "[::1]",
    "testserver",
}

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


@app.middleware("http")
async def local_origin_and_host_guard(request: Request, call_next):
    """Prevent DNS rebinding, cross-origin state-changing attacks, and attach security headers."""
    # 1. DNS Rebinding check: Host header must target localhost/127.0.0.1/[::1]/testserver
    raw_host = request.headers.get("host", "")
    host = raw_host.split(":")[0].strip().lower()
    if host and host not in ALLOWED_LOCAL_HOSTS:
        return JSONResponse(
            status_code=403,
            content={"detail": f"Forbidden: DNS rebinding protection rejected external host '{raw_host}'."},
            headers=SECURITY_HEADERS,
        )

    # 2. Cross-origin state-changing protection for POST/PUT/DELETE/PATCH
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        origin = request.headers.get("origin")
        if origin:
            origin_clean = origin.strip().rstrip("/")
            allowed = any(
                origin_clean == loc.rstrip("/")
                or origin_clean.startswith(("http://localhost:", "http://127.0.0.1:", "http://[::1]:"))
                or origin_clean in ("http://testserver", "https://testserver")
                for loc in LOCALHOST_CORS_ORIGINS
            )
            if not allowed:
                return JSONResponse(
                    status_code=403,
                    content={"detail": f"Forbidden: Cross-origin request from external origin '{origin}' is not permitted."},
                    headers=SECURITY_HEADERS,
                )

        referer = request.headers.get("referer")
        if referer and not origin:
            is_local_ref = any(
                referer.startswith(prefix)
                for prefix in (
                    "http://localhost",
                    "http://127.0.0.1",
                    "http://[::1]",
                    "http://testserver",
                    "https://testserver",
                )
            )
            if not is_local_ref:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Forbidden: Cross-origin request from external referer is not permitted."},
                    headers=SECURITY_HEADERS,
                )

    response = await call_next(request)
    for k, v in SECURITY_HEADERS.items():
        response.headers[k] = v
    return response


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
