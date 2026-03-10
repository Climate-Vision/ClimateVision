"""
Run ClimateVision inference on real Sentinel-2 data via Google Earth Engine,
or on local GeoTIFF files.

Usage:
  # Real satellite data (requires: earthengine authenticate)
  python scripts/infer.py \\
      --bbox -55.0 -5.0 -54.5 -4.5 \\
      --start 2023-01-01 --end 2023-06-01

  # Single local file:
  python scripts/infer.py --input data/processed/test/images/patch_00000.tif

  # All test patches with accuracy vs ground truth:
  python scripts/infer.py \\
      --input data/processed/test/images/ \\
      --mask  data/processed/test/masks/

  # Save prediction overlay PNG:
  python scripts/infer.py --input ... --save-pred out/pred.png
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
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
# GEE mode — download real Sentinel-2 patch and run inference
# ---------------------------------------------------------------------------

def run_gee(
    bbox: list[float],
    start: str,
    end: str,
    cloud_pct: float,
    save_pred: Path | None,
    out_json: Path | None,
) -> None:
    try:
        import ee
    except ImportError:
        logger.error("earthengine-api not installed. Run: pip install earthengine-api")
        sys.exit(1)

    # Authenticate / initialise
    try:
        ee.Initialize()
        logger.info("GEE initialised")
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize()
            logger.info("GEE authenticated and initialised")
        except Exception as exc:
            logger.error("GEE auth failed: %s", exc)
            logger.error("Run: earthengine authenticate")
            sys.exit(1)

    try:
        import rasterio
        import numpy as np
        import urllib.request
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        sys.exit(1)

    west, south, east, north = bbox
    region = ee.Geometry.Rectangle([west, south, east, north])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .select(["B4", "B3", "B2", "B8"])  # R, G, B, NIR — matches training
    )

    count = collection.size().getInfo()
    logger.info("Found %d Sentinel-2 scenes for the requested bbox/dates", count)
    if count == 0:
        logger.error("No cloud-free scenes found. Try wider dates or higher --cloud-pct.")
        sys.exit(1)

    image = collection.median().clip(region)

    # Download
    url = image.getDownloadURL({
        "region": region,
        "scale":  10,          # 10 m native resolution
        "format": "GEO_TIFF",
        "bands":  ["B4", "B3", "B2", "B8"],
    })

    logger.info("Downloading Sentinel-2 composite from GEE…")
    tmp = tempfile.mktemp(suffix=".tif")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        logger.error("The bounding box may be too large (GEE limit ~100 km²). Try a smaller area.")
        sys.exit(1)

    # Read and inspect
    with rasterio.open(tmp) as src:
        image_data = src.read().astype(np.float32)  # (4, H, W)

    c, H, W = image_data.shape
    logger.info("Downloaded image: %d bands  %d×%d px  (%.1f km²)", c, H, W,
                (east - west) * (north - south) * 12321)  # rough km²

    # NDVI stats
    red = image_data[0].astype(np.float64)
    nir = image_data[3].astype(np.float64)
    ndvi = (nir - red) / (nir + red + 1e-8)
    ndvi_mean = float(ndvi.mean())
    ndvi_min  = float(ndvi.min())
    ndvi_max  = float(ndvi.max())

    # Run model inference
    import torch
    from climatevision.inference.pipeline import _load_model

    model, device = _load_model()

    # Tile large images into 64×64 patches (model input size)
    patch_size = 64
    all_preds = np.zeros((H, W), dtype=np.uint8)

    for y in range(0, H, patch_size):
        for x in range(0, W, patch_size):
            patch = image_data[:, y:y + patch_size, x:x + patch_size]
            ph, pw = patch.shape[1], patch.shape[2]

            # Pad if smaller than patch_size
            if ph < patch_size or pw < patch_size:
                padded = np.zeros((4, patch_size, patch_size), dtype=np.float32)
                padded[:, :ph, :pw] = patch
                patch = padded

            patch_norm = (patch / 10000.0).astype(np.float32)
            tensor = torch.FloatTensor(patch_norm.tolist()).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(tensor).argmax(dim=1).squeeze(0)

            all_preds[y:y + ph, x:x + pw] = pred.cpu().numpy()[:ph, :pw]

    forest_px   = int((all_preds == 1).sum())
    total_px    = H * W
    forest_pct  = forest_px / total_px * 100

    # Summary
    print("\n" + "=" * 58)
    print("  Real Sentinel-2 Inference Results")
    print("=" * 58)
    print(f"  BBox        : {west},{south} → {east},{north}")
    print(f"  Dates       : {start}  →  {end}")
    print(f"  Scenes used : {count} (cloud-free median)")
    print(f"  Resolution  : {H}×{W} px at 10 m")
    print(f"  NDVI mean   : {ndvi_mean:.4f}  (min {ndvi_min:.2f} / max {ndvi_max:.2f})")
    print(f"  Forest      : {forest_px:,} px  ({forest_pct:.1f}%)")
    print(f"  Non-forest  : {total_px - forest_px:,} px  ({100 - forest_pct:.1f}%)")
    print("=" * 58 + "\n")

    # Interpret NDVI
    if ndvi_mean > 0.5:
        print("  Vegetation signal: STRONG — dense forest likely")
    elif ndvi_mean > 0.2:
        print("  Vegetation signal: MODERATE — mixed cover")
    else:
        print("  Vegetation signal: WEAK — sparse/no forest")
    print()

    # Save prediction overlay
    if save_pred:
        _save_gee_overlay(image_data, all_preds, save_pred)

    result = {
        "source": "GEE Sentinel-2",
        "bbox": bbox,
        "start": start,
        "end": end,
        "scenes": count,
        "image_size": [H, W],
        "ndvi_stats": {"NDVI_mean": ndvi_mean, "NDVI_min": ndvi_min, "NDVI_max": ndvi_max},
        "inference": {
            "forest_pixels": forest_px,
            "non_forest_pixels": total_px - forest_px,
            "forest_percentage": round(forest_pct, 4),
        },
    }

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Results saved to %s", out_json)

    Path(tmp).unlink(missing_ok=True)


def _save_gee_overlay(image_data, pred_mask, out_path: Path) -> None:
    try:
        from PIL import Image as PILImage
        import numpy as np
    except ImportError:
        return

    rgb = image_data[:3].transpose(1, 2, 0).astype(np.float32)
    lo, hi = rgb.min(), rgb.max()
    rgb = ((rgb - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)

    overlay = rgb.copy()
    forest = pred_mask.astype(bool)
    overlay[forest, 1] = (overlay[forest, 1].astype(int) + 80).clip(0, 255).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(overlay).save(out_path)
    logger.info("Prediction overlay saved: %s", out_path)


# ---------------------------------------------------------------------------
# Local file mode
# ---------------------------------------------------------------------------

def run_on_file(path: Path, mask_path: Path | None, save_pred: Path | None) -> dict:
    from climatevision.inference.pipeline import run_inference_from_file
    result = run_inference_from_file(str(path))
    inf = result["inference"]

    print(f"\n{'='*50}")
    print(f"  File : {path.name}")
    print(f"{'='*50}")
    print(f"  Size       : {inf['image_size'][0]}×{inf['image_size'][1]} px")
    print(f"  Forest     : {inf['forest_pixels']:,} px  ({inf['forest_percentage']:.1f}%)")
    print(f"  Non-forest : {inf['non_forest_pixels']:,} px")
    print(f"  Confidence : {inf['mean_confidence']:.4f}")
    ndvi = result.get("ndvi_stats", {})
    if any(v != 0 for v in ndvi.values()):
        print(f"  NDVI mean  : {ndvi.get('NDVI_mean', 0):.4f}")

    if mask_path and mask_path.exists():
        _compare_mask(path, mask_path)

    print(f"{'='*50}\n")

    if save_pred:
        _save_file_overlay(path, save_pred)

    return result


def _compare_mask(img_path: Path, mask_path: Path) -> None:
    import torch, numpy as np, rasterio
    from climatevision.inference.pipeline import _load_model, _load_image_file

    image = _load_image_file(str(img_path))
    c, h, w = image.shape
    if c < 4:
        image = np.concatenate([image, np.zeros((4 - c, h, w), dtype=np.float32)], axis=0)
    image = (image / 10000.0).astype(np.float32)

    tensor = torch.FloatTensor(image.tolist()).unsqueeze(0)
    model, device = _load_model()
    with torch.no_grad():
        pred = model(tensor.to(device)).argmax(dim=1).squeeze(0)

    with rasterio.open(mask_path) as src:
        gt = torch.LongTensor((src.read(1) > 0).astype("int64").tolist())

    if pred.shape != gt.shape:
        gt = gt[:pred.shape[0], :pred.shape[1]]

    p, t = pred.view(-1), gt.view(-1)
    eps = 1e-6
    tp = int(((p == 1) & (t == 1)).sum().item())
    fp = int(((p == 1) & (t == 0)).sum().item())
    fn = int(((p == 0) & (t == 1)).sum().item())
    tn = int(((p == 0) & (t == 0)).sum().item())
    iou = tp / (tp + fp + fn + eps)
    f1  = 2 * tp / (2 * tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    print(f"\n  vs ground truth:")
    print(f"  Pixel Acc  : {acc:.4f}")
    print(f"  IoU(forest): {iou:.4f}")
    print(f"  F1 (forest): {f1:.4f}")


def _save_file_overlay(img_path: Path, out_path: Path) -> None:
    import torch, numpy as np
    from PIL import Image as PILImage
    from climatevision.inference.pipeline import _load_model, _load_image_file

    image = _load_image_file(str(img_path))
    c, h, w = image.shape
    if c < 4:
        image = np.concatenate([image, np.zeros((4 - c, h, w), dtype=np.float32)], axis=0)
    image = (image / 10000.0).astype(np.float32)

    tensor = torch.FloatTensor(image.tolist()).unsqueeze(0)
    model, device = _load_model()
    with torch.no_grad():
        pred = model(tensor.to(device)).argmax(dim=1).squeeze(0).cpu().numpy()

    rgb = image[:3].transpose(1, 2, 0)
    rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6) * 255).astype("uint8")
    overlay = rgb.copy()
    overlay[pred.astype(bool), 1] = (overlay[pred.astype(bool), 1].astype(int) + 80).clip(0, 255).astype("uint8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(overlay).save(out_path)
    logger.info("Prediction overlay saved: %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ClimateVision inference on GEE satellite data or local files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real Sentinel-2 data (Amazon rainforest):
  python scripts/infer.py --bbox -55.0 -5.0 -54.5 -4.5 --start 2023-01-01 --end 2023-06-01

  # Congo basin:
  python scripts/infer.py --bbox 23.0 -1.0 23.5 -0.5 --start 2023-01-01 --end 2023-12-31

  # Local file:
  python scripts/infer.py --input data/processed/test/images/patch_00000.tif
        """,
    )

    # GEE mode
    gee = p.add_argument_group("GEE mode (real satellite data)")
    gee.add_argument("--bbox",  type=float, nargs=4, metavar=("W", "S", "E", "N"),
                     help="Bounding box in decimal degrees")
    gee.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    gee.add_argument("--end",   default="2023-12-31", help="End date YYYY-MM-DD")
    gee.add_argument("--cloud-pct", type=float, default=20,
                     help="Max cloud cover %% (default 20)")

    # Local file mode
    loc = p.add_argument_group("Local file mode")
    loc.add_argument("--input",  default=None, help="Path to .tif file or directory")
    loc.add_argument("--mask",   default=None, help="Ground-truth mask .tif or dir")
    loc.add_argument("--limit",  type=int, default=20, help="Max patches for directory mode")

    # Shared
    p.add_argument("--save-pred", default=None, help="Save prediction overlay PNG")
    p.add_argument("--out-json",  default=None, help="Write results JSON to file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.bbox:
        # GEE real-data mode
        run_gee(
            bbox=list(args.bbox),
            start=args.start,
            end=args.end,
            cloud_pct=args.cloud_pct,
            save_pred=Path(args.save_pred) if args.save_pred else None,
            out_json=Path(args.out_json) if args.out_json else None,
        )

    elif args.input:
        input_path = Path(args.input)
        mask_path  = Path(args.mask) if args.mask else None
        save_pred  = Path(args.save_pred) if args.save_pred else None

        if input_path.is_dir():
            tifs = sorted(input_path.glob("*.tif"))[:args.limit]
            results = []
            for tif in tifs:
                m = (mask_path / tif.name) if (mask_path and mask_path.is_dir()) else mask_path
                sp = (save_pred / tif.with_suffix(".png").name) if save_pred else None
                results.append(run_on_file(tif, m if (m and m.exists()) else None, sp))

            avg_forest = sum(r["inference"]["forest_percentage"] for r in results) / len(results)
            print(f"Batch summary ({len(results)} patches):")
            print(f"  Avg forest coverage : {avg_forest:.1f}%")

            if args.out_json:
                with open(args.out_json, "w") as f:
                    json.dump(results, f, indent=2)
        else:
            result = run_on_file(input_path, mask_path, save_pred)
            if args.out_json:
                with open(args.out_json, "w") as f:
                    json.dump(result, f, indent=2)
    else:
        print("Specify --bbox (GEE mode) or --input (local file mode).")
        print("Run with --help for examples.")


if __name__ == "__main__":
    main()
