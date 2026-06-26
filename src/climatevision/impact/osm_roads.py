"""
OpenStreetMap road network integration for flood impact assessment.

Downloads highways within a bounding box, rasterizes them to match the
flood prediction mask, and computes affected road length.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OSM road download
# ---------------------------------------------------------------------------

def download_roads(
    bbox: list[float],
    road_types: list[str] | None = None,
) -> Any:
    """
    Download road network from OpenStreetMap within a bounding box.

    Args:
        bbox: [west, south, east, north] in WGS84.
        road_types: OSM highway types to include. Defaults to main roads.

    Returns:
        GeoDataFrame with road LineStrings.

    Raises:
        ImportError: If osmnx is not installed.
    """
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError(
            "osmnx is required for OSM road download. "
            "Install: pip install osmnx"
        )

    if road_types is None:
        road_types = [
            "motorway", "trunk", "primary", "secondary", "tertiary",
            "residential", "unclassified", "road",
        ]

    west, south, east, north = bbox
    gdf = ox.features.features_from_bbox(
        (west, south, east, north),
        tags={"highway": road_types},
    )

    if gdf.empty:
        logger.warning("No roads found in bbox %s", bbox)
        return gdf

    gdf = gdf[gdf.geometry.type == "LineString"].copy()
    return gdf


def download_buildings(
    bbox: list[float],
) -> Any:
    """
    Download building footprints from OpenStreetMap.

    Args:
        bbox: [west, south, east, north] in WGS84.

    Returns:
        GeoDataFrame with building Polygons.
    """
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError(
            "osmnx is required for OSM building download. "
            "Install: pip install osmnx"
        )

    west, south, east, north = bbox
    gdf = ox.features.features_from_bbox(
        (west, south, east, north),
        tags={"building": True},
    )

    if gdf.empty:
        logger.warning("No buildings found in bbox %s", bbox)
        return gdf

    gdf = gdf[gdf.geometry.type == "Polygon"].copy()
    return gdf


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def rasterize_roads(
    roads_gdf: Any,
    raster_shape: tuple[int, int],
    transform: Any,
) -> np.ndarray:
    """
    Rasterize road LineStrings to a binary mask.

    Args:
        roads_gdf: GeoDataFrame with LineString geometries.
        raster_shape: (height, width) of output mask.
        transform: Affine transform (from rasterio.DatasetReader.transform).

    Returns:
        (H, W) uint8 binary mask, 1=road.
    """
    try:
        import rasterio.features
    except ImportError:
        raise ImportError("rasterio is required for rasterization")

    if roads_gdf.empty:
        return np.zeros(raster_shape, dtype=np.uint8)

    shapes = ((geom, 1) for geom in roads_gdf.geometry)
    mask = rasterio.features.rasterize(
        shapes,
        out_shape=raster_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask


# ---------------------------------------------------------------------------
# Impact calculation
# ---------------------------------------------------------------------------

def calculate_affected_road_km(
    flood_mask: np.ndarray,
    road_mask: np.ndarray,
    pixel_size_m: float = 10.0,
) -> float:
    """
    Compute total length of roads inundated by flood.

    Args:
        flood_mask: (H, W) binary mask, 1=flooded.
        road_mask: (H, W) binary mask, 1=road.
        pixel_size_m: Spatial resolution in metres per pixel.

    Returns:
        Affected road length in kilometres.
    """
    flooded_roads = (flood_mask > 0) & (road_mask > 0)
    pixel_count = int(flooded_roads.sum())
    km = pixel_count * pixel_size_m / 1000.0
    return round(km, 3)


# ---------------------------------------------------------------------------
# High-level impact assessment
# ---------------------------------------------------------------------------

def assess_flood_impact(
    flood_mask: np.ndarray,
    bbox: list[float],
    pixel_size_m: float = 10.0,
) -> dict[str, Any]:
    """
    Full flood impact assessment for a given bbox.

    Args:
        flood_mask: (H, W) integer prediction mask.
        bbox: [west, south, east, north] in WGS84.
        pixel_size_m: Spatial resolution.

    Returns:
        Dict with affected_road_km and raw masks.
    """
    try:
        import rasterio.transform
    except ImportError:
        raise ImportError("rasterio is required for impact assessment")

    h, w = flood_mask.shape
    transform = rasterio.transform.from_bounds(
        bbox[0], bbox[1], bbox[2], bbox[3], w, h
    )

    binary_flood = (flood_mask == 2).astype(np.uint8)  # class 2 = flooded

    try:
        roads_gdf = download_roads(bbox)
        road_mask = rasterize_roads(roads_gdf, (h, w), transform)
        affected_road_km = calculate_affected_road_km(binary_flood, road_mask, pixel_size_m)
    except Exception as exc:
        logger.warning("Road impact assessment failed: %s", exc)
        affected_road_km = 0.0

    return {
        "affected_road_km": affected_road_km,
        "bbox": bbox,
        "pixel_size_m": pixel_size_m,
    }
