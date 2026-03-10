"""
Data preparation script for ClimateVision forest segmentation.

Two modes:
  --mode synthetic   Generate fractal-noise synthetic Sentinel-2 patches (no data required)
  --mode gee         Download real Sentinel-2 L2A tiles via Google Earth Engine

Usage:
  # Quick start — 2 000 synthetic patches, default 70/15/15 split:
  python scripts/prepare_data.py --mode synthetic --n-patches 2000 --out data/processed

  # Fewer patches for a fast smoke test:
  python scripts/prepare_data.py --mode synthetic --n-patches 200 --out data/processed

  # Real data via GEE (requires authenticated `earthengine-api`):
  python scripts/prepare_data.py --mode gee \\
      --bbox 2.3 48.8 2.5 49.0 \\
      --start 2022-01-01 --end 2023-12-31 \\
      --out data/processed
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# Synthetic mode
# ---------------------------------------------------------------------------

def generate_synthetic(
    n_patches: int,
    out_dir: Path,
    patch_size: int,
    train_ratio: float,
    val_ratio: float,
) -> None:
    """Delegate entirely to the built-in synthetic generator."""
    try:
        from climatevision.data.synthetic import generate_synthetic_dataset
    except ImportError as exc:
        logger.error("Cannot import climatevision package: %s", exc)
        logger.error("Run `pip install -e .` from the project root first.")
        sys.exit(1)

    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    n_train = int(n_patches * train_ratio)
    n_val   = int(n_patches * val_ratio)
    n_test  = max(0, n_patches - n_train - n_val)

    logger.info(
        "Generating %d synthetic patches  "
        "(train=%d / val=%d / test=%d)  patch_size=%d",
        n_patches, n_train, n_val, n_test, patch_size,
    )

    generate_synthetic_dataset(
        output_dir=out_dir,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        patch_size=patch_size,
    )

    logger.info("Dataset written to %s", out_dir)


# ---------------------------------------------------------------------------
# GEE mode
# ---------------------------------------------------------------------------

def download_gee(
    bbox: tuple[float, float, float, float],
    start: str,
    end: str,
    out_dir: Path,
    patch_size: int,
    max_patches: int,
    train_ratio: float,
    val_ratio: float,
    cloud_threshold: float,
) -> None:
    try:
        import ee
    except ImportError:
        logger.error("earthengine-api not installed. Run: pip install earthengine-api")
        sys.exit(1)

    try:
        ee.Initialize()
    except Exception as exc:
        logger.error("GEE auth failed: %s", exc)
        logger.error("Run: earthengine authenticate")
        sys.exit(1)

    try:
        import rasterio
        import numpy as np
    except ImportError:
        logger.error("rasterio not installed. Run: pip install rasterio")
        sys.exit(1)

    import random, shutil, urllib.request, tempfile, os

    west, south, east, north = bbox
    region = ee.Geometry.Rectangle([west, south, east, north])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold * 100))
        .select(["B4", "B3", "B2", "B8"])  # R, G, B, NIR
    )

    # Dynamic World forest labels (class 1 = trees)
    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(region)
        .filterDate(start, end)
        .select("label")
        .mode()
    )
    forest_mask = dw.eq(1).rename("forest")

    count = collection.size().getInfo()
    logger.info("Found %d Sentinel-2 scenes", count)

    image    = collection.median().clip(region)
    combined = image.addBands(forest_mask)

    url = combined.getDownloadURL({
        "region": region,
        "scale":  10,
        "format": "GEO_TIFF",
    })

    logger.info("Downloading GEE composite…")
    tmp = tempfile.mktemp(suffix=".tif")
    urllib.request.urlretrieve(url, tmp)

    with rasterio.open(tmp) as src:
        full    = src.read()   # (5, H, W)
        profile = src.profile

    image_data = full[:4].astype(np.float32)
    mask_data  = (full[4] > 0).astype(np.uint8)
    _, H, W    = image_data.shape

    # Extract patches
    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            if len(patches) >= max_patches:
                break
            patches.append((
                image_data[:, y:y + patch_size, x:x + patch_size],
                mask_data[    y:y + patch_size, x:x + patch_size],
            ))
        else:
            continue
        break

    os.unlink(tmp)
    logger.info("Extracted %d patches", len(patches))

    # Shuffle + split
    random.seed(42)
    random.shuffle(patches)
    n       = len(patches)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    splits  = {
        "train": patches[:n_train],
        "val":   patches[n_train:n_train + n_val],
        "test":  patches[n_train + n_val:],
    }

    p = profile.copy()
    for split, split_patches in splits.items():
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "masks").mkdir(parents=True, exist_ok=True)
        for idx, (img_patch, mask_patch) in enumerate(split_patches):
            stem = f"patch_{idx:05d}"
            p.update(count=4, dtype="float32", height=patch_size, width=patch_size)
            with rasterio.open(out_dir / split / "images" / f"{stem}.tif", "w", **p) as dst:
                dst.write(img_patch)
            p.update(count=1, dtype="uint8")
            with rasterio.open(out_dir / split / "masks" / f"{stem}.tif", "w", **p) as dst:
                dst.write(mask_patch[np.newaxis])
        logger.info("  %s: %d patches", split, len(split_patches))

    logger.info("Dataset written to %s", out_dir)


# ---------------------------------------------------------------------------
# Normaliser fitting
# ---------------------------------------------------------------------------

def fit_normalizer(data_dir: Path, out_path: Path) -> None:
    """Compute per-band mean/std on the training set and save to JSON."""
    try:
        from climatevision.data.preprocessing import Sentinel2Normalizer
    except ImportError as exc:
        logger.warning("Could not fit normalizer: %s", exc)
        return

    import glob
    import rasterio

    tifs = sorted(glob.glob(str(data_dir / "train" / "images" / "*.tif")))
    if not tifs:
        logger.warning("No training images found — skipping normalizer fit")
        return

    logger.info("Fitting normalizer on %d training images…", len(tifs))
    arrays = []
    for p in tifs:
        with rasterio.open(p) as src:
            arrays.append(src.read().astype("float32"))
    norm = Sentinel2Normalizer()
    norm.fit(arrays)
    norm.save(out_path)
    logger.info("Normalizer stats saved to %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare dataset for ClimateVision training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["synthetic", "gee"], default="synthetic")
    p.add_argument("--out",  type=Path, default=Path("data/processed"),
                   help="Output directory (created if needed)")

    # Synthetic options
    p.add_argument("--n-patches", type=int, default=2000,
                   help="[synthetic] Total number of patches to generate")
    p.add_argument("--patch-size", type=int, default=256,
                   help="Spatial size of each patch in pixels")

    # GEE options
    p.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"),
                   help="[gee] Bounding box: west south east north")
    p.add_argument("--start", type=str, default="2022-01-01",
                   help="[gee] Start date YYYY-MM-DD")
    p.add_argument("--end",   type=str, default="2023-12-31",
                   help="[gee] End date YYYY-MM-DD")
    p.add_argument("--max-patches", type=int, default=5000,
                   help="[gee] Maximum patches to extract from download")
    p.add_argument("--cloud-threshold", type=float, default=0.2,
                   help="[gee] Max cloud fraction (0–1)")

    # Split ratios
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio",   type=float, default=0.15)

    # Normalizer
    p.add_argument("--fit-normalizer", action="store_true",
                   help="Fit per-band stats on training set after generation")
    p.add_argument("--normalizer-out", type=Path, default=None,
                   help="Where to write normalizer JSON (default: <out>/normalizer.json)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.train_ratio + args.val_ratio > 1.0:
        logger.error("--train-ratio + --val-ratio must be ≤ 1.0")
        sys.exit(1)

    if args.mode == "synthetic":
        generate_synthetic(
            n_patches=args.n_patches,
            out_dir=args.out,
            patch_size=args.patch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
    else:
        if not args.bbox:
            logger.error("--bbox W S E N is required for --mode gee")
            sys.exit(1)
        download_gee(
            bbox=tuple(args.bbox),          # type: ignore[arg-type]
            start=args.start,
            end=args.end,
            out_dir=args.out,
            patch_size=args.patch_size,
            max_patches=args.max_patches,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            cloud_threshold=args.cloud_threshold,
        )

    if args.fit_normalizer:
        norm_out = args.normalizer_out or (args.out / "normalizer.json")
        fit_normalizer(args.out, norm_out)


if __name__ == "__main__":
    main()
