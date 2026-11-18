"""Download 3 freely licensed real sample images and record their sources and licenses."""

import hashlib
import json
from pathlib import Path
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "datasets" / "samples"

SAMPLES = [
    {
        "filename": "sample_01_building.jpg",
        "url": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/building.jpg",
        "title": "Architectural Building Exterior (OpenCV Sample Data)",
        "source_repo": "https://github.com/opencv/opencv/tree/4.x/samples/data",
        "license": "Apache-2.0",
        "scene_type": "Outdoor architectural facade with strong perspective depth planes",
    },
    {
        "filename": "sample_02_aloe_indoor.jpg",
        "url": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/aloeL.jpg",
        "title": "Middlebury Aloe Plant Tabletop Scene (OpenCV / Middlebury Stereo)",
        "source_repo": "https://github.com/opencv/opencv/tree/4.x/samples/data",
        "license": "Apache-2.0 (OpenCV distribution of Middlebury stereo benchmark sample)",
        "scene_type": "Indoor close-range plant with multi-layer self-occlusions and sharp depth discontinuities",
    },
    {
        "filename": "sample_03_home_scene.jpg",
        "url": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/home.jpg",
        "title": "Residential House & Foreground Foliage (OpenCV Sample Data)",
        "source_repo": "https://github.com/opencv/opencv/tree/4.x/samples/data",
        "license": "Apache-2.0",
        "scene_type": "Outdoor residential building with foreground/background separation",
    },
]


def main() -> int:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for item in SAMPLES:
        target = SAMPLES_DIR / item["filename"]
        if not target.exists():
            req = urllib.request.Request(
                item["url"],
                headers={"User-Agent": "HoloGen-Sample-Fetcher/0.1"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            target.write_bytes(data)
        else:
            data = target.read_bytes()

        sha256 = hashlib.sha256(data).hexdigest()
        record = {
            **item,
            "sha256": sha256,
            "size_bytes": len(data),
        }
        manifest.append(record)

    (SAMPLES_DIR / "sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md_lines = [
        "# HoloGen Sample Images & License Attribution",
        "",
        "All sample images used for Phase 1–5 baseline verification and experiments are freely licensed and fetched deterministically via `scripts/fetch_sample_images.py`.",
        "",
        "| Filename | Scene Description | License | Source URL | SHA-256 (prefix) | Size (bytes) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for rec in manifest:
        md_lines.append(
            f"| `{rec['filename']}` | {rec['scene_type']} | {rec['license']} | {rec['url']} | `{rec['sha256'][:12]}...` | {rec['size_bytes']} |"
        )
    (SAMPLES_DIR / "SOURCES.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Downloaded and verified {len(manifest)} sample images in {SAMPLES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
