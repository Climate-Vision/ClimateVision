"""
Google Earth Engine tile downloader for ClimateVision.

Provides analysis-aware Sentinel-2 and Sentinel-1 tile downloads.
GEE failures raise explicit exceptions — NO synthetic fallbacks in production.
"""
from __future__ import annotations

import logging
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

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


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GEEAuthenticationError(RuntimeError):
    """Raised when GEE credentials are missing or invalid."""


class TileNotFoundError(RuntimeError):
    """Raised when no satellite images are found for the given criteria."""


# ---------------------------------------------------------------------------
# GEE initialisation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Sentinel-2 download
# ---------------------------------------------------------------------------

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
        (file_path, metadata_dict).

    Raises:
        GEEAuthenticationError: If GEE cannot be initialised.
        TileNotFoundError: If no images are found for the date range.
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
        raise GEEAuthenticationError(
            f"Google Earth Engine unavailable: {exc}. "
            f"Set GEE_PROJECT_ID or GEE_SERVICE_ACCOUNT + GEE_SERVICE_ACCOUNT_KEY."
        ) from exc

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
        raise TileNotFoundError(
            f"No Sentinel-2 images found for {analysis_type} "
            f"from {start_date} to {end_date} in bbox {bbox}."
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

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=data.shape[0],
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)

    metadata: dict[str, Any] = {
        "source": "gee",
        "satellite": "sentinel2",
        "analysis_type": analysis_type,
        "bbox": bbox,
        "start_date": start_date,
        "end_date": end_date,
        "bands": bands,
        "scale_m": scale_m,
        "images_available": count,
        "shape": list(data.shape),
    }

    logger.info("Downloaded S2 tile to %s (%d images available)", out_path, count)
    return out_path, metadata


# ---------------------------------------------------------------------------
# Sentinel-1 SAR download
# ---------------------------------------------------------------------------

def download_s1_tile(
    bbox: list[float],
    start_date: str,
    end_date: str,
    polarization: list[str] | None = None,
    output_dir: str | Path | None = None,
    scale_m: int = 100,
    orbit: str = "DESCENDING",
) -> tuple[Path, dict[str, Any]]:
    """
    Download a median Sentinel-1 SAR composite for the given bbox and date range.

    Args:
        bbox: [west, south, east, north] in WGS84.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        polarization: List of polarizations, e.g., ["VV", "VH"]. Defaults to both.
        output_dir: Where to save the GeoTIFF.
        scale_m: GEE export resolution in metres.
        orbit: "ASCENDING", "DESCENDING", or "BOTH".

    Returns:
        (file_path, metadata_dict).

    Raises:
        GEEAuthenticationError: If GEE cannot be initialised.
        TileNotFoundError: If no SAR images are found.
    """
    if output_dir is None:
        output_dir = _SATELLITE_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if polarization is None:
        polarization = ["VV", "VH"]

    safe_start = start_date.replace("-", "")
    safe_end = end_date.replace("-", "")
    stem = f"s1_{safe_start}_{safe_end}_{'_'.join(str(round(c, 4)) for c in bbox)}"
    out_path = output_dir / f"{stem}.tif"

    try:
        ee = _initialize_ee()
        rasterio = __import__("rasterio")
    except Exception as exc:
        raise GEEAuthenticationError(
            f"Google Earth Engine unavailable: {exc}."
        ) from exc

    region = ee.Geometry.Rectangle(bbox)
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select(polarization)
    )

    if orbit != "BOTH":
        collection = collection.filter(ee.Filter.eq("orbitProperties_pass", orbit))

    count = collection.size().getInfo()
    if count == 0:
        raise TileNotFoundError(
            f"No Sentinel-1 images found from {start_date} to {end_date} in bbox {bbox}."
        )

    # Convert to linear intensity from dB scale
    image = collection.median().clip(region)
    image = image.pow(2)  # GEE S1 is in dB; convert to linear intensity for ML

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

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=data.shape[0],
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)

    metadata: dict[str, Any] = {
        "source": "gee",
        "satellite": "sentinel1",
        "bbox": bbox,
        "start_date": start_date,
        "end_date": end_date,
        "polarization": polarization,
        "scale_m": scale_m,
        "images_available": count,
        "shape": list(data.shape),
    }

    logger.info("Downloaded S1 tile to %s (%d images available)", out_path, count)
    return out_path, metadata
