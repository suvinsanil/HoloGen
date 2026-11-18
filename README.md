# HoloGen: AI-Powered 3D Scene Reconstruction & Generation Platform

> *"From pixels to spatial intelligence."*

HoloGen is a modular computer vision and 3D geometry engineering platform that transforms single or multiple 2D RGB images into clean, textured, exportable 3D scenes (`GLB`, `OBJ`, `PLY`, `STL`, `JSON`) with measured reconstruction reports.

---

## Key Capabilities & Engineering Principles

- **Strict Scale Honesty:** Monocular depth models (`Depth Anything V2 Small`, `DPT-Hybrid MiDaS`) produce affine-invariant relative disparity. HoloGen explicitly labels single-view outputs as `relative_unknown_scale` unless a calibrated metric model (`Depth Anything V2 Metric Indoor Small`) or multi-view baseline is used.
- **Explicit Pinhole Camera Geometry:** Configurable focal length extraction from EXIF (`FocalLengthIn35mmFilm` / `FocalLength`) with horizontal field-of-view fallback (default $60^\circ$) and analytical intrinsic matrix $K$ / $K^{-1}$ operations.
- **Edge-Preserving Depth & Uncertainty Proxy:** Guided and bilateral depth refinement paired with a multi-cue monocular **uncertainty proxy** (spatial depth-gradient discontinuity, horizontal flip consistency, and inter-model disagreement).
- **Discontinuity-Aware Geometry Processing:** Statistical/radius outlier filtering, PCA normal estimation with camera-facing orientation, depth-grid triangulation with automatic discontinuity edge cutting, and Open3D surface reconstruction comparisons (Poisson, Ball Pivoting, Alpha Shapes).
- **Projective UV Texturing & Multi-Format Export:** Camera-projected UV coordinates with material-bound `GLB`, `OBJ`, `PLY`, `STL`, and comprehensive JSON/Markdown reconstruction reports.

---

## Quickstart (Clean Checkout)

### 1. Environment Setup (Windows PowerShell)

```powershell
# Create virtual environment
python -m venv .venv

# Install pinned dependencies (includes CUDA 12.1 wheels compatible with Pascal/MX330 GPUs)
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Fetch Freely Licensed Sample Images

```powershell
.\.venv\Scripts\python.exe scripts/fetch_sample_images.py
```

### 3. Run Unit Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

### 4. Run Single-View Reconstruction CLI (Viewer & Full-Res Presets)

```powershell
.\.venv\Scripts\python.exe scripts/run_baseline.py
```

Outputs are written to `outputs/<run_id>/`:
- `input_preprocessed.png` — EXIF-oriented, aspect-resized input RGB image
- `depth_visualization.png` & `uncertainty_proxy.png` — Inferno disparity and uncertainty proxy maps
- `point_cloud.ply` & `mesh.ply` — Cleaned colored 3D point cloud and surface mesh
- `scene.glb` — Viewer-quality textured glTF 2.0 binary (`~120k` faces, `< 5 MB`)
- `scene_full.glb` — Full-resolution textured glTF 2.0 binary (`380k–675k` faces)
- `scene.obj` & `mesh.stl` — Wavefront OBJ and binary STL exports
- `reconstruction_report.json` & `reconstruction_report.md` — Measured timings, hardware telemetry, intrinsics $K$, View-Frustum Coverage, and uncertainty proxy histogram

### 5. Launch Phase 6 Web Application (FastAPI + React Three Fiber)

```powershell
# Terminal 1: Start FastAPI backend (preloads singleton depth model on startup, concurrency=1 queue)
$env:PYTHONPATH = "backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Start React + Vite + Three.js frontend (use npm.cmd on Windows PowerShell)
cd frontend
npm.cmd install
npm.cmd run dev
```

Open `http://localhost:5173` in your browser to upload images, adjust reconstruction parameters, inspect 2D depth/uncertainty layers alongside the interactive 3D viewport (`Mesh`, `Wireframe`, `RGB`, `Depth`, `Point Cloud`, `Normals`, `Uncertainty Proxy`), and download exported assets.

> [!NOTE]
> **Windows 11 Smart App Control Workaround:** On Windows 11 machines with Smart App Control enabled, unsigned native `.node` / `.exe` binaries inside `node_modules` (specifically `@rollup/rollup-win32-x64-msvc` and `esbuild.exe`) and unsigned `.ps1` / `pip.exe` shims can be blocked by OS policy. To ensure a reproducible build without weakening OS security settings:
> 1. Always invoke Python tools via the signed interpreter (`.\.venv\Scripts\python.exe -m pip`, `.\.venv\Scripts\python.exe -m pytest`) and Node via `npm.cmd`.
> 2. `frontend/package.json` pins `overrides` for `rollup` (`npm:@rollup/wasm-node@4.27.3`) and `esbuild` (`npm:esbuild-wasm@0.21.5`) so Vite builds inside the signed Node.js V8 WebAssembly runtime.

---

## Repository Structure

```text
HoloGen/
├── backend/
│   ├── app/
│   │   ├── core/          # Pydantic config, hardware auto-detection, logging, exceptions
│   │   ├── vision/        # Preprocessing, camera intrinsics, depth estimation, refinement, uncertainty
│   │   ├── geometry/      # Point cloud back-projection, filtering, normals, meshing, texturing, export
│   │   ├── pipelines/     # Single-view and multi-view orchestrators
│   │   └── evaluation/    # Depth metrics, 3D geometry metrics, and benchmark harnesses
│   └── tests/             # Unit tests with hand-computed mathematical verifications
├── scripts/               # CLI runners and reproducible experiment generators
├── datasets/              # Sample images (with license manifest) and benchmark subsets
├── models/                # Model registry & license documentation (weights cached locally, never committed)
├── outputs/               # Generated 3D scenes and reconstruction reports
└── docs/                  # Architecture, methodology, experiments, and evaluation documentation
```

---

## Documentation Index

- [Architecture Overview](docs/architecture.md)
- [Mathematical Methodology](docs/methodology.md)
- [Experimental Results & Ablations](docs/experiments.md)
- [Evaluation Metrics & Limitations](docs/evaluation.md)
- [Pretrained Models & Licenses](models/README.md)
