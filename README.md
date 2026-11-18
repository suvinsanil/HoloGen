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

### 4. Run Phase 1 Baseline Pipeline

```powershell
.\.venv\Scripts\python.exe scripts/run_baseline.py
```

Outputs are written to `outputs/<run_id>/`:
- `input_preprocessed.png` — EXIF-oriented, aspect-resized input RGB image
- `depth_visualization.png` — Inferno colormap disparity visualization
- `point_cloud.ply` — Colored 3D point cloud in camera coordinates
- `run_report.json` — Measured timings, hardware telemetry, intrinsics $K$, quality metrics, and point cloud bounds

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
