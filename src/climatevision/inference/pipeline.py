"""
Inference pipeline for ClimateVision.

Provides:
- run_inference(image_array, bbox, start_date, end_date, analysis_type) — core inference on a numpy array
- run_inference_from_file(path, bbox, start_date, end_date, analysis_type) — load file then infer
- run_inference_from_gee(bbox, start_date, end_date, analysis_type) — GEE NDVI + real tile inference
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from climatevision.data.band_mapping import get_bands_for_analysis, get_model_config
from climatevision.models.unet import UNet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths (mirrors run_training.py conventions, NOT Config.MODELS_DIR)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR = _PROJECT_ROOT / "models"
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Per-analysis-type model cache
# ---------------------------------------------------------------------------
_model_cache: dict[str, tuple[UNet, torch.device]] = {}


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _find_best_checkpoint(analysis_type: str) -> Optional[Path]:
    """
    Search for the best available checkpoint for an analysis type.
    Priority: config.yaml weight path > models/best_model.pth > newest models/*/best_model.pth
    """
    model_cfg = get_model_config(analysis_type)
    config_path = model_cfg.get("weights")
    if config_path:
        p = _PROJECT_ROOT / config_path
        if p.exists():
            return p

    direct = _MODELS_DIR / "best_model.pth"
    if direct.exists():
        return direct
    candidates = sorted(
        _MODELS_DIR.glob("*/best_model.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_model(analysis_type: str = "deforestation") -> tuple[UNet, torch.device]:
    """Load (or return cached) U-Net model configured for the analysis type."""
    if analysis_type in _model_cache:
        return _model_cache[analysis_type]

    device = _get_device()
    model_cfg = get_model_config(analysis_type)
    n_channels = model_cfg.get("in_channels", 4)
    n_classes = model_cfg.get("num_classes", 2)

    model = UNet(n_channels=n_channels, n_classes=n_classes)

    model_path = _find_best_checkpoint(analysis_type)
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)

        # Load full state first (includes BatchNorm running stats)
        model_state = checkpoint.get("model_state_dict")
        ema_state   = checkpoint.get("ema_state_dict")

        if model_state is not None:
            model.load_state_dict(model_state, strict=False)
        # Overlay EMA parameters on top (better generalisation)
        if ema_state is not None:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in ema_state:
                        param.data.copy_(ema_state[name])

        logger.info(
            "Loaded %s model from %s  (epoch %s  val_iou %.4f)",
            analysis_type,
            model_path,
            checkpoint.get("epoch", "?"),
            checkpoint.get("val_iou", 0.0),
        )
    else:
        logger.warning(
            "No trained model found for %s under %s — using untrained weights (demo).",
            analysis_type,
            _MODELS_DIR,
        )

    model = model.to(device)
    model.eval()

    _model_cache[analysis_type] = (model, device)
    return model, device


# ---------------------------------------------------------------------------
# Sentinel-2 normalisation statistics (matches preprocessing.py)
# Band order: [Red, Green, Blue, NIR]
# ---------------------------------------------------------------------------
_S2_MEAN = np.array([943.0, 1069.0, 981.0, 2734.0], dtype=np.float64)
_S2_STD  = np.array([590.0,  547.0, 498.0, 1246.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# NDVI helper (works for >=4 bands; returns zeros for RGB-only)
# ---------------------------------------------------------------------------

def _compute_ndvi_stats(image: np.ndarray) -> dict[str, float]:
    """
    Compute NDVI min/mean/max from image array.

    Expects (C, H, W) with C >= 4 where band order is [Red, Green, Blue, NIR].
    Automatically detects and reverses Sentinel-2 z-score normalisation
    (values in roughly [-5, 5]) before computing NDVI.
    Returns zeros if fewer than 4 bands.
    """
    if image.ndim == 2:
        return {"NDVI_min": 0.0, "NDVI_mean": 0.0, "NDVI_max": 0.0}

    # Normalise to (C, H, W)
    if image.ndim == 3 and image.shape[2] < image.shape[0]:
        image = np.transpose(image, (2, 0, 1))

    n_bands = image.shape[0]
    if n_bands < 4:
        return {"NDVI_min": 0.0, "NDVI_mean": 0.0, "NDVI_max": 0.0}

    # Band order: Red=0, Green=1, Blue=2, NIR=3
    red = image[0].astype(np.float64)
    nir = image[3].astype(np.float64)

    # If data looks like z-score normalised input (values in [-10, 10])
    # denormalise back to raw Sentinel-2 DN before computing NDVI.
    if red.max() <= 10.0 and nir.max() <= 10.0:
        red = red * _S2_STD[0] + _S2_MEAN[0]
        nir = nir * _S2_STD[3] + _S2_MEAN[3]

    denom = nir + red + 1e-8
    ndvi = (nir - red) / denom

    return {
        "NDVI_min":  round(float(np.nanmin(ndvi)),  4),
        "NDVI_mean": round(float(np.nanmean(ndvi)), 4),
        "NDVI_max":  round(float(np.nanmax(ndvi)),  4),
    }


def _synthetic_ndvi_stats(bbox: Optional[list[float]]) -> dict[str, float]:
    """
    Compute NDVI from a synthetic but physically realistic Sentinel-2 scene.

    Used as a fallback when GEE credentials are unavailable.
    The bbox is used to seed the RNG so the same region always returns
    the same values. Band statistics match typical tropical/temperate forest.
    """
    seed = 42
    if bbox:
        seed = int(abs(sum(v * 1000 * (i + 1) for i, v in enumerate(bbox)))) % (2 ** 31)
    rng = np.random.default_rng(seed)

    # Typical Sentinel-2 L2A forest reflectance (DN, 0-10000 scale)
    # Red ~600-1200, NIR ~2500-5000
    red = rng.normal(900.0,  350.0, (256, 256)).clip(50.0,  5000.0)
    nir = rng.normal(3800.0, 900.0, (256, 256)).clip(100.0, 9000.0)

    denom = nir + red + 1e-8
    ndvi = (nir - red) / denom

    return {
        "NDVI_min":  round(float(np.nanmin(ndvi)),  4),
        "NDVI_mean": round(float(np.nanmean(ndvi)), 4),
        "NDVI_max":  round(float(np.nanmax(ndvi)),  4),
    }


# ---------------------------------------------------------------------------
# Core inference on a numpy array
# ---------------------------------------------------------------------------

def run_inference(
    image: np.ndarray,
    *,
    bbox: Optional[list[float]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    analysis_type: str = "deforestation",
    enable_carbon:bool = False, # Feature flag for carbon loss estimation (Issue #14)
) -> dict[str, Any]:
    """
    Run full inference pipeline on a (C, H, W) numpy image.

    Returns dict with keys: region, ndvi_stats, inference.
    """
    # Normalise to (C, H, W)
    if image.ndim == 3 and image.shape[2] < image.shape[0]:
        image = np.transpose(image, (2, 0, 1))

    ndvi_stats = _compute_ndvi_stats(image)

    model, device = _load_model(analysis_type)
    n_channels = model.n_channels
    n_classes = model.n_classes

    # Prepare tensor — model expects (N, n_channels, H, W)
    c, h, w = image.shape
    if c < n_channels:
        # Pad missing channels with zeros
        pad = np.zeros((n_channels - c, h, w), dtype=image.dtype)
        image = np.concatenate([image, pad], axis=0)
    elif c > n_channels:
        image = image[:n_channels]

    # Use torch.FloatTensor via tolist() to avoid numpy<->torch interop issues
    tensor = torch.FloatTensor(image.astype(np.float32).tolist()).unsqueeze(0)  # (1, C, H, W)
    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)
        predictions = torch.argmax(output, dim=1)  # (1, H, W)
        probabilities = torch.softmax(output, dim=1)  # (1, n_classes, H, W)

    total_pixels = int(predictions.numel())
    max_probs = probabilities.max(dim=1).values
    mean_confidence = float(max_probs.mean().item())

    # Build per-class pixel counts
    class_pixels: dict[str, int] = {}
    class_percentages: dict[str, float] = {}
    for cls in range(n_classes):
        count = int((predictions == cls).sum().item())
        pct = (count / total_pixels) * 100 if total_pixels else 0.0
        class_pixels[f"class_{cls}_pixels"] = count
        class_percentages[f"class_{cls}_percentage"] = round(pct, 4)

    # Add friendly keys for known 2-class deforestation output (backward compat)
    inference: dict[str, Any] = {
        "image_size": [h, w],
        "num_classes": n_classes,
        "mean_confidence": round(mean_confidence, 4),
        **class_pixels,
        **class_percentages,
    }
    if n_classes == 2:
        inference["forest_pixels"] = class_pixels.get("class_1_pixels", 0)
        inference["non_forest_pixels"] = class_pixels.get("class_0_pixels", 0)
        inference["forest_percentage"] = class_percentages.get("class_1_percentage", 0.0)
    
    # === Integrate Carbon Computation (Issue #14) ===
    # Post-processing step for deforestation analysis to estimate carbon tonnes and hectares lost.
    #Begin fix issue #14

    carbon_estimation = None
    if enable_carbon and analysis_type == "deforestation":
        try:
            from climatevision.analytics.carbon import CarbonEstimator

            estimator = CarbonEstimator(
                forest_type="tropical_moist",
                pixel_size_m=10.0
            )

            mask_np = (predictions == 1).squeeze(0).cpu().numpy()
            conf_np = max_probs.squeeze(0).cpu().numpy()

            carbon_result = estimator.estimate_from_mask(
                deforestation_mask=mask_np,
                confidence_map=conf_np
            )

            carbon_estimation = {
                "hectares_lost": carbon_result.hectares,
                "carbon_tonnes": carbon_result.carbon_tonnes,
                "co2_equivalent": carbon_result.co2_equivalent,
                "confidence_interval": {
                    "lower": carbon_result.ci_lower,
                    "upper": carbon_result.ci_upper,
                    "uncertainty_pct": carbon_result.uncertainty_pct
                }
            }
            logger.info(f"Carbon estimation successful: {carbon_result.carbon_tonnes} tonnes")
            
        except Exception as e:
            # Graceful degradation: If carbon estimation fails, log the error and return None 
            # to prevent the main API inference pipeline from crashing.
            logger.warning(f"Failed to estimate carbon stock: {e}")
            carbon_estimation = None
    #End fix issue #14

    region: dict[str, Any] = {}
    if bbox is not None:
        region["bbox"] = bbox
    if start_date and end_date:
        region["date_range"] = f"{start_date} to {end_date}"

    return {
        "region": region,
        "ndvi_stats": ndvi_stats,
        "inference": inference,
        "carbon_estimation": carbon_estimation,
        "is_synthetic": False,

    }


# ---------------------------------------------------------------------------
# File-based inference (upload path)
# ---------------------------------------------------------------------------

def run_inference_from_file(
    path: str,
    *,
    bbox: Optional[list[float]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    analysis_type: str = "deforestation",
    enable_carbon: bool = False,  # Pass-through flag for carbon estimation
) -> dict[str, Any]:
    """
    Load an image file (GeoTIFF or PNG/JPEG) and run inference.
    """
    image = _load_image_file(path)
    result = run_inference(
        image,
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        analysis_type=analysis_type,
        enable_carbon=enable_carbon,  # Forwarding flag to the core pipeline
    )
    result.setdefault("input", {})["file"] = path
    return result


def _load_image_file(path: str) -> np.ndarray:
    """
    Load image as (C, H, W) numpy array.
    Tries rasterio first (GeoTIFF), falls back to Pillow.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    # Try rasterio for geospatial formats
    if suffix in {".tif", ".tiff", ".geotiff"}:
        try:
            import rasterio

            with rasterio.open(path) as src:
                image = src.read()  # (C, H, W)
            return image.astype(np.float32)
        except Exception:
            logger.warning("rasterio failed for %s, trying Pillow", path)

    # Pillow fallback for PNG, JPEG, etc.
    from PIL import Image

    pil_img = Image.open(path)
    arr = np.array(pil_img)  # (H, W, C) or (H, W)

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]  # (1, H, W)
    else:
        arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)

    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# GEE-based inference (bbox path) — lazy import, safe fallback
# ---------------------------------------------------------------------------

def run_inference_from_gee(
    *,
    bbox: Optional[list[float]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    analysis_type: str = "deforestation",
    enable_carbon: bool = False, # Pass-through flag for carbon estimation
) -> dict[str, Any]:
    """
    Query Google Earth Engine for a real Sentinel-2 tile and run inference.

    Falls back to synthetic NDVI stats and a synthetic tile if GEE is
    unavailable or returns no images.
    """
    ndvi_stats: Optional[dict[str, Any]] = None
    gee_count: int = 0

    if bbox and start_date and end_date:
        ndvi_stats, gee_count = _try_gee_ndvi(bbox, start_date, end_date)

    # --- Attempt to download a real tile from GEE ---
    try:
        from climatevision.data import download_tile_for_analysis, apply_scl_cloud_mask

        tile_path, metadata = download_tile_for_analysis(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            analysis_type=analysis_type,
        )

        image = _load_image_file(str(tile_path))

        # If SCL band is present (last band), apply cloud mask and drop it
        n_bands_expected = len(get_bands_for_analysis(analysis_type))
        if image.shape[0] == n_bands_expected + 1:
            scl_band = image[-1].astype(np.uint8)
            image = image[:-1]
            image = apply_scl_cloud_mask(image, scl_band)

        result = run_inference(
            image,
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            analysis_type=analysis_type,
            enable_carbon=enable_carbon,  # issue #14
        )
        result["metadata"] = metadata
        result["is_synthetic"] = metadata.get("is_synthetic", False)

        # Override NDVI with GEE-derived stats if we got them; else keep computed
        if ndvi_stats is not None:
            result["ndvi_stats"] = ndvi_stats
        elif metadata.get("is_synthetic"):
            result["ndvi_stats"] = _synthetic_ndvi_stats(bbox)

        if gee_count:
            result["region"]["images_available"] = gee_count

        return result

    except Exception as exc:
        logger.warning("Real tile inference failed (%s). Using fallback.", exc)

    # --- Fallback: template result with synthetic stats ---
    result = run_inference(
        np.zeros((4, 256, 256), dtype=np.float32),
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        analysis_type=analysis_type,
        enable_carbon=enable_carbon,  # Maintain user preference even in fallback mode
    )

    if ndvi_stats is None:
        ndvi_stats = _synthetic_ndvi_stats(bbox)
    result["ndvi_stats"] = ndvi_stats

    region = result.get("region", {})
    if gee_count:
        region["images_available"] = gee_count
    result["region"] = region
    result["is_synthetic"] = True
    result["metadata"] = {"is_synthetic": True, "fallback_reason": "gee_tile_download_failed"}

    return result


def _try_gee_ndvi(
    bbox: list[float], start_date: str, end_date: str
) -> tuple[Optional[dict[str, Any]], int]:
    """Attempt GEE NDVI query. Returns (ndvi_stats_or_None, image_count)."""
    try:
        import ee  # lazy import
        import os

        project     = os.getenv("GEE_PROJECT_ID")
        svc_account = os.getenv("GEE_SERVICE_ACCOUNT")
        key_file    = os.getenv("GEE_SERVICE_ACCOUNT_KEY")

        # Resolve relative key path against project root
        if key_file and not os.path.isabs(key_file):
            key_file = str(_PROJECT_ROOT / key_file)

        if svc_account and key_file and os.path.exists(key_file):
            credentials = ee.ServiceAccountCredentials(svc_account, key_file)
            ee.Initialize(credentials)
        elif project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()

        geometry = ee.Geometry.Rectangle(bbox)
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .select(["B4", "B3", "B2", "B8"])
        )

        count = collection.size().getInfo()

        median = collection.median()
        nir = median.select("B8")
        red = median.select("B4")
        ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")

        stats = ndvi.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True),
            geometry=geometry,
            scale=100,
            maxPixels=int(1e9),
        ).getInfo()

        return stats, count

    except Exception as exc:
        logger.warning("GEE query failed (%s). Using fallback.", exc)
        return None, 0


def _load_cached_ndvi() -> dict[str, float]:
    """Load NDVI from outputs/inference_results.json if it exists, else zeros."""
    cached = _OUTPUTS_DIR / "inference_results.json"
    if cached.exists():
        try:
            data = json.loads(cached.read_text(encoding="utf-8"))
            return data.get("ndvi_stats", {"NDVI_min": 0.0, "NDVI_mean": 0.0, "NDVI_max": 0.0})
        except Exception:
            pass
    return {"NDVI_min": 0.0, "NDVI_mean": 0.0, "NDVI_max": 0.0}
