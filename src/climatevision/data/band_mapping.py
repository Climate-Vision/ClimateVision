"""
Analysis-specific Sentinel-2 band mapping utilities.

Provides a single source of truth for which spectral bands each
climate analysis type requires, derived from config.yaml.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

# Full Sentinel-2 L2A 13-band stack in canonical order
SENTINEL2_BAND_ORDER = [
    "B01", "B02", "B03", "B04",
    "B05", "B06", "B07", "B08",
    "B8A", "B09", "B10", "B11", "B12",
]

# Scene Classification Layer (SCL) is not part of the 13 reflectance bands
# but is essential for cloud masking.
SCL_BAND = "SCL"


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    """Load the master config.yaml once and cache it."""
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_bands_for_analysis(analysis_type: str) -> list[str]:
    """
    Return the Sentinel-2 band names required for *analysis_type*.

    The bands are read from ``config.yaml`` and are guaranteed to be
    returned in the same order they are declared there.
    """
    cfg = _load_config()
    analysis_cfg = cfg.get("analysis_types", {}).get(analysis_type, {})
    bands = analysis_cfg.get("bands", ["B04", "B03", "B02", "B08"])
    return list(bands)


def get_bands_for_analysis_with_scl(analysis_type: str) -> list[str]:
    """
    Return required bands plus the SCL band for cloud masking.

    If SCL is already in the band list it is not duplicated.
    """
    bands = get_bands_for_analysis(analysis_type)
    if SCL_BAND not in bands:
        bands = bands + [SCL_BAND]
    return bands


def get_band_indices(band_names: list[str]) -> list[int]:
    """
    Map Sentinel-2 band names to zero-based indices in the 13-band stack.

    Raises:
        ValueError: If a band name is not recognised.
    """
    indices = []
    for b in band_names:
        if b == SCL_BAND:
            # SCL does not belong to the 13 reflectance bands;
            # callers that need an index in a multi-band array should
            # append it separately and compute len(reflectance_bands).
            raise ValueError(
                f"SCL is not part of the 13-band reflectance stack. "
                f"Append it manually after resolving reflectance indices."
            )
        if b not in SENTINEL2_BAND_ORDER:
            raise ValueError(f"Unknown Sentinel-2 band: {b}")
        indices.append(SENTINEL2_BAND_ORDER.index(b))
    return indices


def is_analysis_enabled(analysis_type: str) -> bool:
    """Return True if the analysis type is enabled in config.yaml."""
    cfg = _load_config()
    analysis_cfg = cfg.get("analysis_types", {}).get(analysis_type, {})
    return bool(analysis_cfg.get("enabled", False))


def list_enabled_analysis_types() -> list[str]:
    """Return all analysis type names that are currently enabled."""
    cfg = _load_config()
    return [
        name
        for name, analysis_cfg in cfg.get("analysis_types", {}).items()
        if analysis_cfg.get("enabled", False)
    ]


def get_model_config(analysis_type: str) -> dict[str, Any]:
    """
    Return the ``model`` subsection for an analysis type.

    This contains keys such as ``architecture``, ``in_channels``,
    and ``num_classes``.
    """
    cfg = _load_config()
    analysis_cfg = cfg.get("analysis_types", {}).get(analysis_type, {})
    return dict(analysis_cfg.get("model", {}))
