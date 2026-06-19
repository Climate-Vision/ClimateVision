"""
SAR flood-detection inference pipeline (bbox -> 3-class flood result).

Orchestrates the full path the API uses for flooding requests:
    1. Download a Sentinel-1 VV/VH tile for the bbox/date range (synthetic fallback).
    2. Download the JRC Global Surface Water occurrence layer -> permanent-water
       reference (skipped when GEE is unavailable; the split is then unresolved).
    3. Run FloodingSARAnalysis (ensemble + permanent/flood classifier).
    4. Return a result dict in the same shape as run_inference_from_gee.

All heavy/geo dependencies are imported lazily so importing this module is cheap.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from climatevision.analysis.flooding_sar import FloodingSARAnalysis
from climatevision.analysis.flood_classification import permanent_water_from_occurrence

logger = logging.getLogger(__name__)


def run_flood_inference_from_gee(
    *,
    bbox: Optional[list[float]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    occurrence_threshold_pct: float = 50.0,
    analysis_type: str = "flooding_sar",
) -> dict[str, Any]:
    """Run SAR flood detection for a bbox/date range. Never raises for missing
    data -- falls back to a synthetic SAR scene tagged is_synthetic=True."""
    from climatevision.data import (
        download_sar_tile,
        download_permanent_water_occurrence,
    )

    sar_path, sar_meta = download_sar_tile(
        bbox=bbox, start_date=start_date, end_date=end_date,
    )
    image = _read_tile(str(sar_path))  # (2, H, W) VV/VH

    permanent_ref = _load_permanent_reference(
        bbox, image.shape[-2:], occurrence_threshold_pct, download_permanent_water_occurrence
    )

    analysis = FloodingSARAnalysis(permanent_water_ref=permanent_ref)
    date_range = f"{start_date} to {end_date}" if start_date and end_date else None
    result = analysis.analyze(image=image, bbox=bbox, date_range=date_range)

    payload = result.to_dict()
    payload["analysis_type"] = analysis_type
    is_synthetic = bool(sar_meta.get("is_synthetic", False))
    payload["is_synthetic"] = is_synthetic

    # Downstream road-impact assessment (NGO/gov actionability).
    road_impact = _road_impact(result.mask, bbox) if result.mask is not None and bbox else {
        "affected_road_km": None, "road_impact_available": False, "reason": "no_mask_or_bbox",
    }
    if road_impact.get("affected_road_km") is not None:
        payload.setdefault("inference", {})["affected_road_km"] = road_impact["affected_road_km"]

    distinguished = bool(payload.get("inference", {}).get("permanent_flood_distinguished", False))
    payload["metadata"] = {
        "sar": sar_meta,
        "permanent_water_reference": permanent_ref is not None,
        "occurrence_threshold_pct": occurrence_threshold_pct,
        "road_impact": road_impact,
    }
    # Provenance/QA block: what a decision-maker needs to judge trust at a glance.
    payload["quality"] = {
        "is_synthetic": is_synthetic,
        "permanent_flood_distinguished": distinguished,
        "permanent_water_reference_source": "jrc_gsw_occurrence" if permanent_ref is not None else "none",
        "road_impact_available": bool(road_impact.get("road_impact_available")),
        "validated_against_benchmark": False,  # set true only after a real eval run
        "notes": _quality_notes(is_synthetic, distinguished, permanent_ref is not None),
    }
    return payload


def _quality_notes(is_synthetic: bool, distinguished: bool, has_ref: bool) -> list[str]:
    notes: list[str] = []
    if is_synthetic:
        notes.append("Result is from a SYNTHETIC fallback scene (GEE unavailable) -- not a real observation.")
    if not distinguished:
        notes.append("No permanent-water reference or pre-event scene: detected water reported as flooded; "
                     "permanent vs flood NOT separated.")
    elif has_ref:
        notes.append("Flood/permanent split derived from JRC Global Surface Water occurrence.")
    notes.append("Detector accuracy not yet validated against a labeled benchmark (run scripts/eval_flood.py).")
    return notes


def _road_impact(flood_mask: np.ndarray, bbox: list[float]) -> dict[str, Any]:
    """Affected road length via OSM, degrading honestly when osmnx is absent."""
    try:
        import osmnx  # noqa: F401
    except ImportError:
        return {"affected_road_km": None, "road_impact_available": False, "reason": "osmnx_not_installed"}
    try:
        from climatevision.impact.osm_roads import assess_flood_impact
        out = assess_flood_impact(flood_mask, bbox)
        return {"affected_road_km": out.get("affected_road_km", 0.0), "road_impact_available": True}
    except Exception as exc:  # network/OSM failure
        logger.warning("Road-impact assessment failed: %s", exc)
        return {"affected_road_km": None, "road_impact_available": False, "reason": str(exc)}


def _load_permanent_reference(
    bbox, target_hw, threshold_pct, downloader,
) -> Optional[np.ndarray]:
    """Fetch JRC GSW occurrence and turn it into a permanent-water mask aligned
    to the SAR grid. Returns None when GEE is unavailable (no guessing)."""
    if bbox is None:
        return None
    try:
        occ_path, occ_meta = downloader(bbox=bbox)
    except Exception as exc:  # network/credential failure -> no reference
        logger.warning("Permanent-water reference fetch failed (%s); split unresolved.", exc)
        return None
    if occ_path is None:
        return None

    occ = _read_tile(str(occ_path))
    occ = occ[0] if occ.ndim == 3 else occ
    occ = _resample_nearest(occ, target_hw)
    return permanent_water_from_occurrence(occ, threshold_pct=threshold_pct)


def _read_tile(path: str) -> np.ndarray:
    """Read a GeoTIFF as a float32 array (C, H, W) or (H, W)."""
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("rasterio is required to read flood tiles") from exc
    with rasterio.open(path) as ds:
        return ds.read().astype(np.float32)


def _resample_nearest(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resample a 2-D array to target (H, W) without SciPy."""
    th, tw = target_hw
    h, w = arr.shape
    if (h, w) == (th, tw):
        return arr
    ys = (np.linspace(0, h - 1, th)).round().astype(int)
    xs = (np.linspace(0, w - 1, tw)).round().astype(int)
    return arr[np.ix_(ys, xs)]
