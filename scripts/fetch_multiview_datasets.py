"""Download and extract calibrated multi-view benchmark datasets:
1. Open3D SampleFountainRGBDImages (19.72 MB) — labeled strictly as 'reference geometry (RGB-D reconstruction)'.
2. ETH3D 'pipes' DSLR multi-view dataset (104.59 MB) + 'pipes_dslr_scan_eval' Faro laser scan ground truth (51.36 MB).
"""

from pathlib import Path
import urllib.request

import open3d as o3d
import py7zr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"


def download_with_progress(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        print(f"Already downloaded: {dest.name} ({dest.stat().st_size / (1024*1024):.2f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "HoloGen-Benchmark-Fetcher/0.7"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
    print(f"Finished {dest.name} ({dest.stat().st_size / (1024*1024):.2f} MB)")


def safe_extract_7z(archive_path: Path, target_dir: Path) -> None:
    """Safely extract 7z archive preventing Zip Slip / path traversal attacks."""
    resolved_target = target_dir.resolve()
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        for fname in z.getnames():
            dest = (resolved_target / fname).resolve()
            if not dest.is_relative_to(resolved_target):
                raise RuntimeError(f"Zip slip / path traversal detected in archive member: {fname}")
        z.extractall(path=resolved_target)


def main() -> int:
    # 1. Open3D SampleFountainRGBDImages (reference RGB-D reconstruction)
    fountain_root = DATASETS_DIR / "fountain"
    fountain = o3d.data.SampleFountainRGBDImages(data_root=str(fountain_root))
    print(
        f"Fountain dataset ready: {len(fountain.color_paths)} views, mesh={fountain.reconstruction_path}"
    )

    # 2. ETH3D 'pipes' DSLR images + Ground-Truth Laser Scan
    eth3d_root = DATASETS_DIR / "eth3d"
    jpg_7z = eth3d_root / "downloads" / "pipes_dslr_jpg.7z"
    scan_7z = eth3d_root / "downloads" / "pipes_dslr_scan_eval.7z"

    download_with_progress("https://www.eth3d.net/data/pipes_dslr_jpg.7z", jpg_7z)
    download_with_progress("https://www.eth3d.net/data/pipes_dslr_scan_eval.7z", scan_7z)

    extract_dir = eth3d_root / "pipes_extracted"
    if not (extract_dir / "pipes").exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {jpg_7z.name} ...")
        safe_extract_7z(jpg_7z, extract_dir)
        print(f"Extracting {scan_7z.name} ...")
        safe_extract_7z(scan_7z, extract_dir)

    print("All multi-view datasets ready in", extract_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
