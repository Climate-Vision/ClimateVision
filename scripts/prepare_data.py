"""
Data preparation script for ClimateVision.

Modes:
  --mode synthetic   Generate synthetic patches for testing/smoke tests
  --mode gee         Download real Sentinel-2 L2A tiles via Google Earth Engine

Usage:
  # Synthetic flood patches for testing:
  python scripts/prepare_data.py --mode synthetic --analysis-type flooding --n-patches 200 --out data/processed/flood

  # Real data via GEE:
  python scripts/prepare_data.py --mode gee --analysis-type flooding \
      --bbox 36.7 -1.4 37.0 -1.1 \
      --start 2024-04-01 --end 2024-04-10 \
      --out data/processed/flood
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
    analysis_type: str,
    n_patches: int,
    out_dir: Path,
    patch_size: int,
    train_ratio: float,
    val_ratio: float,
) -> None:
    """Generate synthetic patches for testing."""
    from climatevision.data.synthetic import (
        generate_synthetic_dataset,
        generate_flood_test_patches,
    )

    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    n_train = int(n_patches * train_ratio)
    n_val = int(n_patches * val_ratio)
    n_test = max(0, n_patches - n_train - n_val)

    if analysis_type == "flooding":
        logger.info(
            "Generating %d synthetic flood patches (train=%d / val=%d / test=%d)",
            n_patches, n_train, n_val, n_test,
        )
        for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
            if n > 0:
                split_dir = out_dir / split
                generate_flood_test_patches(
                    output_dir=split_dir,
                    n=n,
                    patch_size=patch_size,
                    use_sar=False,
                    seed=42 if split == "train" else 99 if split == "val" else 123,
                )
    else:
        logger.info(
            "Generating %d synthetic forest patches (train=%d / val=%d / test=%d)",
            n_patches, n_train, n_val, n_test,
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
    analysis_type: str,
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
        import os
        svc_account = os.getenv("GEE_SERVICE_ACCOUNT")
        key_file = os.getenv("GEE_SERVICE_ACCOUNT_KEY")
        project = os.getenv("GEE_PROJECT_ID")

        if key_file and not os.path.isabs(key_file):
            key_file = str(PROJECT_ROOT / key_file)

        if svc_account and key_file and os.path.exists(key_file):
            credentials = ee.ServiceAccountCredentials(svc_account, key_file)
            ee.Initialize(credentials)
        elif project:
            ee.Initialize(project=project)
        else:
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

    import random, urllib.request, tempfile, os

    from climatevision.data.band_mapping import get_bands_for_analysis

    bands = get_bands_for_analysis(analysis_type)
    gee_bands = {
        "B01": "B1", "B02": "B2", "B03": "B3", "B04": "B4",
        "B05": "B5", "B06": "B6", "B07": "B7", "B08": "B8",
        "B8A": "B8A", "B09": "B9", "B10": "B10", "B11": "B11", "B12": "B12",
    }
    selected_bands = [gee_bands[b] for b in bands]

    west, south, east, north = bbox

    TILE_DEG = 0.25
    SCALE_M = 100

    tiles = []
    lat = south
    while lat < north:
        lon = west
        while lon < east:
            tiles.append((
                round(lon, 6),
                round(lat, 6),
                round(min(lon + TILE_DEG, east), 6),
                round(min(lat + TILE_DEG, north), 6),
            ))
            lon += TILE_DEG
        lat += TILE_DEG

    logger.info("Downloading %d tiles (%.2f deg each, scale=%dm)…", len(tiles), TILE_DEG, SCALE_M)

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    base_profile = {
        "driver": "GTiff",
        "crs": "EPSG:4326",
        "transform": rasterio.transform.from_bounds(west, south, east, north, patch_size, patch_size),
    }

    for ti, (tw, ts, te, tn) in enumerate(tiles):
        if len(patches) >= max_patches:
            break

        tile_region = ee.Geometry.Rectangle([tw, ts, te, tn])

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(tile_region)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold * 100))
            .select(selected_bands)
        )

        # For flood detection, use JRC Global Surface Water for water mask baseline
        if analysis_type == "flooding":
            water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").clip(tile_region)
            water_mask = water.gt(50).rename("water")
            # We can't easily generate "flooded" labels without event-specific data,
            # so we create a 3-class mask: dry=0, permanent_water=1, flooded=2
            # Here we set flooded=water (simplified — in production use event-specific polygons)
            label = water_mask
        else:
            dw = (
                ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                .filterBounds(tile_region)
                .filterDate(start, end)
                .select("label")
                .mode()
            )
            label = dw.eq(1).rename("forest")

        try:
            image = collection.median().clip(tile_region)
            combined = image.addBands(label)

            url = combined.getDownloadURL({
                "region": tile_region,
                "scale": SCALE_M,
                "format": "GEO_TIFF",
            })

            tmp = tempfile.mktemp(suffix=".tif")
            urllib.request.urlretrieve(url, tmp)

            with rasterio.open(tmp) as src:
                full = src.read()

            os.unlink(tmp)

            n_bands = len(selected_bands)
            if full.shape[0] < n_bands + 1:
                continue

            image_data = full[:n_bands].astype(np.float32)
            mask_data = full[n_bands].astype(np.uint8)
            _, H, W = image_data.shape

            for y in range(0, H - patch_size + 1, patch_size):
                for x in range(0, W - patch_size + 1, patch_size):
                    if len(patches) >= max_patches:
                        break
                    patches.append((
                        image_data[:, y:y + patch_size, x:x + patch_size],
                        mask_data[y:y + patch_size, x:x + patch_size],
                    ))
                if len(patches) >= max_patches:
                    break

            logger.info("  tile %d/%d  ->  %d patches so far", ti + 1, len(tiles), len(patches))

        except Exception as exc:
            logger.warning("  tile %d/%d skipped: %s", ti + 1, len(tiles), exc)
            continue

    if not patches:
        logger.error("No patches extracted — check GEE credentials and bbox")
        sys.exit(1)

    logger.info("Extracted %d patches total", len(patches))

    random.seed(42)
    random.shuffle(patches)
    n = len(patches)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = {
        "train": patches[:n_train],
        "val": patches[n_train:n_train + n_val],
        "test": patches[n_train + n_val:],
    }

    for split, split_patches in splits.items():
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "masks").mkdir(parents=True, exist_ok=True)
        for idx, (img_patch, mask_patch) in enumerate(split_patches):
            stem = f"patch_{idx:05d}"
            img_profile = {**base_profile, "count": img_patch.shape[0], "dtype": "float32",
                           "height": patch_size, "width": patch_size}
            with rasterio.open(out_dir / split / "images" / f"{stem}.tif", "w", **img_profile) as dst:
                dst.write(img_patch)
            msk_profile = {**base_profile, "count": 1, "dtype": "uint8",
                           "height": patch_size, "width": patch_size}
            with rasterio.open(out_dir / split / "masks" / f"{stem}.tif", "w", **msk_profile) as dst:
                dst.write(mask_patch[np.newaxis])
        logger.info("  %s: %d patches", split, len(split_patches))

    logger.info("Dataset written to %s", out_dir)


# ---------------------------------------------------------------------------
# Normaliser fitting
# ---------------------------------------------------------------------------

def fit_normalizer(data_dir: Path, out_path: Path) -> None:
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
    p.add_argument("--analysis-type", default="deforestation",
                   help="Analysis type: deforestation, flooding, ice_melting")
    p.add_argument("--out", type=Path, default=Path("data/processed"),
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
    p.add_argument("--end", type=str, default="2023-12-31",
                   help="[gee] End date YYYY-MM-DD")
    p.add_argument("--max-patches", type=int, default=5000,
                   help="[gee] Maximum patches to extract from download")
    p.add_argument("--cloud-threshold", type=float, default=0.2,
                   help="[gee] Max cloud fraction (0-1)")

    # Split ratios
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)

    # Normalizer
    p.add_argument("--fit-normalizer", action="store_true",
                   help="Fit per-band stats on training set after generation")
    p.add_argument("--normalizer-out", type=Path, default=None,
                   help="Where to write normalizer JSON (default: <out>/normalizer.json)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.train_ratio + args.val_ratio > 1.0:
        logger.error("--train-ratio + --val-ratio must be <= 1.0")
        sys.exit(1)

    if args.mode == "synthetic":
        generate_synthetic(
            analysis_type=args.analysis_type,
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
            analysis_type=args.analysis_type,
            bbox=tuple(args.bbox),
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
