# HoloGen

HoloGen is a local 3D reconstruction platform that converts single-view and multi-view images into interactive 3D meshes and point clouds. It integrates monocular depth estimation, Structure-from-Motion (SfM), and volumetric TSDF fusion into a unified web application.

## Features

- **Single-View Reconstruction**: Depth estimation with Depth Anything V2, point cloud back-projection, depth-discontinuity filtering, surface meshing (depth-grid triangulation, Screened Poisson, Ball Pivoting), and projective UV texturing.
- **Multi-View Reconstruction**: Incremental Structure-from-Motion via pycolmap, sparse depth alignment, multi-view consistency filtering, and Open3D TSDF volumetric fusion.
- **Interactive 3D Viewer**: Web-based Three.js viewer with orbit controls, wireframe toggles, and camera frustum visualization.
- **Model Export**: Export reconstructed 3D geometry in GLB, OBJ, PLY, and STL formats.

## Requirements

- Python 3.10+
- Node.js 18+
- (Optional) NVIDIA GPU with CUDA for faster inference; CPU fallback is supported automatically.

## Quick Start

### 1. Clone the repository

```bash
git clone <repo-url>
cd HoloGen
```

### 2. Set up backend

```bash
python -m venv .venv

# On Windows:
.\.venv\Scripts\activate

# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note for CUDA users**: To ensure GPU acceleration with PyTorch, you can install the CUDA-enabled wheel:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 3. Build the frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### 4. Run the application

Start the FastAPI server:

```bash
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open your browser and navigate to:

```
http://localhost:8000
```

If you prefer to run the frontend in development mode with hot reload:

```bash
# In frontend directory
npm run dev
```

The dev server will run at `http://localhost:5173` and proxy API calls to port 8000.

## Running Tests

Run the test suite to verify the installation:

```bash
# Backend tests
pytest backend/tests

# Frontend build check
cd frontend && npm run build
```

## Project Structure

```text
HoloGen/
├── backend/          # FastAPI backend
│   ├── app/          # Core APIs, geometry, pipelines, models
│   └── tests/        # Backend test suite
├── frontend/         # React + Vite + Three.js application
├── docs/             # Technical architecture and evaluation documentation
├── scripts/          # Benchmark and reproduction scripts
├── requirements.txt  # Python package dependencies
└── LICENSE           # MIT License
```

## Documentation

Detailed documentation and benchmark analyses are available in `docs/`:

- [Architecture Overview](docs/architecture.md)
- [Evaluation Protocols & Results](docs/evaluation.md)
- [Experiments & Ablation Studies](docs/experiments.md)

## License

This project is licensed under the [MIT License](LICENSE).
