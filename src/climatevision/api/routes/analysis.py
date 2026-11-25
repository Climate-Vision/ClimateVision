"""
Analysis type endpoints: the catalogue of supported climate analyses.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()


SUPPORTED_ANALYSIS_TYPES: list[dict[str, Any]] = [
    {
        "name": "deforestation",
        "display_name": "Deforestation Detection",
        "description": "Monitor forest coverage and detect deforestation events",
        "enabled": True,
        "bands": ["B04", "B03", "B02", "B08"],
        "classes": ["non_forest", "forest"],
    },
    {
        "name": "ice_melting",
        "display_name": "Arctic Ice Melting",
        "description": "Monitor sea ice extent and melting patterns in polar regions",
        "enabled": True,
        "bands": ["B02", "B03", "B04", "B11"],
        "classes": ["sea_ice", "open_water", "land"],
    },
    {
        "name": "flooding",
        "display_name": "Flood Detection",
        "description": "Detect and monitor flooding events and affected areas",
        "enabled": True,
        "bands": ["B03", "B08", "B11"],
        "classes": ["water", "flooded", "dry_land"],
    },
    {
        "name": "flooding_sar",
        "display_name": "Flood Detection (SAR)",
        "description": "All-weather flood detection from Sentinel-1 VV/VH using a physics-based ensemble",
        "enabled": True,
        "bands": ["VV", "VH"],
        "classes": ["dry_land", "permanent_water", "flooded"],
    },
    {
        "name": "drought",
        "display_name": "Drought Monitoring",
        "description": "Monitor vegetation stress and drought conditions",
        "enabled": False,
        "bands": ["B04", "B08", "B11", "B12"],
        "classes": ["normal", "stressed", "severe_drought"],
    },
    {
        "name": "wildfire",
        "display_name": "Wildfire Detection",
        "description": "Detect active fires and burned areas",
        "enabled": False,
        "bands": ["B04", "B08", "B11", "B12"],
        "classes": ["unburned", "burned", "active_fire"],
    },
]


@router.get("/api/analysis-types")
def list_analysis_types(enabled_only: bool = True) -> list[dict[str, Any]]:
    """List available analysis types."""
    if enabled_only:
        return [t for t in SUPPORTED_ANALYSIS_TYPES if t["enabled"]]
    return SUPPORTED_ANALYSIS_TYPES


@router.get("/api/analysis-types/{analysis_type}")
def get_analysis_type(analysis_type: str) -> dict[str, Any]:
    """Get details for a specific analysis type."""
    for t in SUPPORTED_ANALYSIS_TYPES:
        if t["name"] == analysis_type:
            return t
    raise HTTPException(
        status_code=404, detail=f"Analysis type '{analysis_type}' not found"
    )
