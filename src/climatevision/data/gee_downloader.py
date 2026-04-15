"""
Google Earth Engine tile downloader for ClimateVision.

Provides analysis-aware Sentinel-2 tile downloads with a synthetic fallback
when GEE credentials are unavailable. Downloaded tiles are saved as GeoTIFF
and include a metadata dict that labels synthetic scenes explicitly.
"""
from __future__ import annotations

import logging
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .band_mapping import get_bands_for_analysis

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SATELLITE_DIR = _PROJECT_ROOT / "data" / "satellite"

# Standard Sentinel-2 band name → GEE asset name mapping
_BAND_NAME_TO_GEE = {
    "B01": "B1",
    "B02": "B2",
    "B03": "B3",
    "B04": "B4",
    "B05": "B5",
    "B06": "B6",
    "B07": "B7",
    "B08": "B8",
    "B8A": "B8A",
    "B09": "B9",
    "B10": "B10",
    "B11": "B11",
    "B12": "B12",
}


def _initialize_ee() -> Any:
    """Lazy import and initialise Google Earth Engine."""
    import ee  # noqa

    project = os.getenv("GEE_PROJECT_ID")
    svc_account = os.getenv("GEE_SERVICE_ACCOUNT")
    key_file = os.getenv("GEE_SERVICE_ACCOUNT_KEY")

    if key_file and not os.path.isabs(key_file):
        key_file = str(_PROJECT_ROOT / key_file)

    if svc_account and key_file and os.path.exists(key_file):
        credentials = ee.ServiceAccountCredentials(svc_account, key_file)
        ee.Initialize(credentials)
    elif project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()
    return ee


def _get_default_tile_size() -> int:
    """Read the default tile size from config.yaml."""
    import yaml

    config_path = _PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    image_size = cfg.get("data", {}).get("image_size", [256, 256])
    return int(image_size[0])


def download_tile_for_analysis(
    bbox: list[float],
    start_date: str,
    end_date: str,
    analysis_type: str = "deforestation",
    output_dir: str | Path | None = None,
    scale_m: int = 100,
    include_scl: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """
    Download a median Sentinel-2 composite for the given bbox and date range.

    Args:
        bbox: [west, south, east, north] in WGS84.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        analysis_type: One of the keys in config.yaml ``analysis_types``.
        output_dir: Where to save the GeoTIFF. Defaults to ``data/satellite/``.
        scale_m: GEE export resolution in metres.
        include_scl: Whether to append the SCL band for cloud masking.

    Returns:
        (file_path, metadata_dict).  If GEE is unavailable, the synthetic
        fallback is used and ``metadata["is_synthetic"]`` is ``True``.
    """
    if output_dir is None:
        output_dir = _SATELLITE_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_start = start_date.replace("-", "")
    safe_end = end_date.replace("-", "")
    stem = f"{analysis_type}_{safe_start}_{safe_end}_{'_'.join(str(round(c, 4)) for c in bbox)}"
    out_path = output_dir / f"{stem}.tif"

    try:
        ee = _initialize_ee()
        rasterio = __import__("rasterio")
    except Exception as exc:
        logger.warning("GEE unavailable (%s). Using synthetic fallback.", exc)
        return _generate_synthetic_tile(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            analysis_type=analysis_type,
            out_path=out_path,
        )

    bands = get_bands_for_analysis(analysis_type)
    gee_bands = [_BAND_NAME_TO_GEE[b] for b in bands]
    if include_scl and "SCL" not in gee_bands:
        gee_bands.append("SCL")

    region = ee.Geometry.Rectangle(bbox)
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .select(gee_bands)
    )

    count = collection.size().getInfo()
    if count == 0:
        logger.warning(
            "No GEE images found for %s %s to %s. Using synthetic fallback.",
            analysis_type, start_date, end_date,
        )
        return _generate_synthetic_tile(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            analysis_type=analysis_type,
            out_path=out_path,
        )

    image = collection.median().clip(region)

    url = image.getDownloadURL({
        "region": region,
        "scale": scale_m,
        "format": "GEO_TIFF",
    })

    tmp = tempfile.mktemp(suffix=".tif")
    urllib.request.urlretrieve(url, tmp)

    with rasterio.open(tmp) as src:
        data = src.read().astype(np.float32)
        profile = src.profile

    os.unlink(tmp)

    # Re-order bands to match project convention if needed
    # (GEE returns in selection order)
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=data.shape[0],
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)

    metadata: dict[str, Any] = {
        "source": "gee",
        "analysis_type": analysis_type,
        "bbox": bbox,
        "start_date": start_date,
        "end_date": end_date,
        "bands": bands,
        "scale_m": scale_m,
        "images_available": count,
        "is_synthetic": False,
        "shape": list(data.shape),
    }

    logger.info("Downloaded real tile to %s (%d images available)", out_path, count)
    return out_path, metadata


def _generate_synthetic_tile(
    bbox: list[float],
    start_date: str,
    end_date: str,
    analysis_type: str,
    out_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """
    Generate a physically plausible synthetic Sentinel-2 tile when GEE fails.
    The output is explicitly tagged ``is_synthetic: True``.
    """
    rasterio = __import__("rasterio")

    bands = get_bands_for_analysis(analysis_type)
    n_bands = len(bands)
    tile_size = _get_default_tile_size()
    h, w = tile_size, tile_size

    # Seed RNG from bbox so the same region is deterministic
    seed = int(abs(sum(v * 1000 * (i + 1) for i, v in enumerate(bbox)))) % (2 ** 31)
    rng = np.random.default_rng(seed)

    # Build a synthetic stack: draw reflectance values typical for mixed forest
    data = np.zeros((n_bands, h, w), dtype=np.float32)
    for b in range(n_bands):
        mean = rng.uniform(500.0, 3000.0)
        std = rng.uniform(200.0, 800.0)
        data[b] = rng.normal(mean, std, (h, w)).clip(0.0, 10000.0)

    # Append an SCL band (all clear = 4)
    scl = np.full((1, h, w), 4, dtype=np.float32)
    data = np.concatenate([data, scl], axis=0)

    transform = rasterio.transform.from_bounds(
        bbox[0], bbox[1], bbox[2], bbox[3], w, h
    )
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": data.shape[0],
        "height": h,
        "width": w,
        "crs": "EPSG:4326",
        "transform": transform,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)

    metadata: dict[str, Any] = {
        "source": "synthetic_fallback",
        "analysis_type": analysis_type,
        "bbox": bbox,
        "start_date": start_date,
        "end_date": end_date,
        "bands": bands,
        "scale_m": 100,
        "images_available": 0,
        "is_synthetic": True,
        "shape": list(data.shape),
    }

    logger.info("Generated synthetic fallback tile to %s", out_path)
    return out_path, metadata
