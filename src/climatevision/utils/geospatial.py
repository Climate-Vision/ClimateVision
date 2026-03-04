"""
Geospatial utilities for coordinate transformations and area calculations
"""
import numpy as np
from typing import Tuple, List, Optional
import rasterio
from rasterio.transform import from_bounds
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import box, Polygon


def latlon_to_pixel(
    lat: float,
    lon: float,
    transform: rasterio.Affine
) -> Tuple[int, int]:
    """
    Convert latitude/longitude to pixel coordinates
    
    Args:
        lat: Latitude
        lon: Longitude
        transform: Rasterio affine transform
    
    Returns:
        (row, col) pixel coordinates
    """
    col, row = ~transform * (lon, lat)
    return int(row), int(col)


def pixel_to_latlon(
    row: int,
    col: int,
    transform: rasterio.Affine
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to latitude/longitude
    
    Args:
        row: Pixel row
        col: Pixel column
        transform: Rasterio affine transform
    
    Returns:
        (latitude, longitude)
    """
    lon, lat = transform * (col, row)
    return lat, lon


def calculate_pixel_area_ha(transform: rasterio.Affine, crs: str = "EPSG:4326") -> float:
    """
    Calculate area of a single pixel in hectares
    
    Args:
        transform: Rasterio affine transform
        crs: Coordinate reference system
    
    Returns:
        Area in hectares
    """
    # Get pixel size in degrees or meters
    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)
    
    if "4326" in crs or "WGS84" in crs:
        # Convert from degrees to meters (approximate at equator)
        # 1 degree ≈ 111,320 meters
        pixel_width_m = pixel_width * 111320
        pixel_height_m = pixel_height * 111320
    else:
        # Assume already in meters
        pixel_width_m = pixel_width
        pixel_height_m = pixel_height
    
    # Calculate area in square meters and convert to hectares
    area_m2 = pixel_width_m * pixel_height_m
    area_ha = area_m2 / 10000  # 1 hectare = 10,000 m²
    
    return area_ha


def calculate_deforestation_area(
    prediction_mask: np.ndarray,
    transform: rasterio.Affine,
    crs: str = "EPSG:4326",
    deforestation_class: int = 1
) -> float:
    """
    Calculate total deforestation area in hectares
    
    Args:
        prediction_mask: Binary or multi-class prediction mask
        transform: Rasterio affine transform
        crs: Coordinate reference system
        deforestation_class: Class index representing deforestation
    
    Returns:
        Deforestation area in hectares
    """
    pixel_area_ha = calculate_pixel_area_ha(transform, crs)
    num_deforested_pixels = np.sum(prediction_mask == deforestation_class)
    total_area_ha = num_deforested_pixels * pixel_area_ha
    
    return total_area_ha


def create_bounding_box(
    center_lat: float,
    center_lon: float,
    width_km: float,
    height_km: float
) -> Tuple[float, float, float, float]:
    """
    Create bounding box around a center point
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        width_km: Width in kilometers
        height_km: Height in kilometers
    
    Returns:
        (min_lon, min_lat, max_lon, max_lat) bounding box
    """
    # Approximate conversion: 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 111 * cos(latitude) km
    lat_offset = (height_km / 2) / 111
    lon_offset = (width_km / 2) / (111 * np.cos(np.radians(center_lat)))
    
    min_lat = center_lat - lat_offset
    max_lat = center_lat + lat_offset
    min_lon = center_lon - lon_offset
    max_lon = center_lon + lon_offset
    
    return (min_lon, min_lat, max_lon, max_lat)


def reproject_bbox(
    bbox: Tuple[float, float, float, float],
    src_crs: str = "EPSG:4326",
    dst_crs: str = "EPSG:3857"
) -> Tuple[float, float, float, float]:
    """
    Reproject bounding box to different CRS
    
    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        src_crs: Source coordinate reference system
        dst_crs: Destination coordinate reference system
    
    Returns:
        Reprojected bounding box
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Transform corners
    min_x, min_y = transformer.transform(min_lon, min_lat)
    max_x, max_y = transformer.transform(max_lon, max_lat)
    
    return (min_x, min_y, max_x, max_y)


def extract_roi(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Extract region of interest from image
    
    Args:
        image: Input image (H, W) or (C, H, W)
        bbox: (min_row, min_col, max_row, max_col)
    
    Returns:
        Cropped image
    """
    min_row, min_col, max_row, max_col = bbox
    
    if image.ndim == 3:
        return image[:, min_row:max_row, min_col:max_col]
    else:
        return image[min_row:max_row, min_col:max_col]


def create_geotiff_metadata(
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
    crs: str = "EPSG:4326",
    num_bands: int = 1,
    dtype: str = "uint8"
) -> dict:
    """
    Create metadata for writing GeoTIFF files
    
    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        width: Image width in pixels
        height: Image height in pixels
        crs: Coordinate reference system
        num_bands: Number of bands
        dtype: Data type
    
    Returns:
        Rasterio metadata dictionary
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
    
    metadata = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': num_bands,
        'dtype': dtype,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw',
    }
    
    return metadata


def calculate_carbon_loss(
    deforestation_area_ha: float,
    biomass_density_t_per_ha: float = 150.0,
    carbon_fraction: float = 0.47
) -> float:
    """
    Estimate carbon loss from deforestation
    
    Args:
        deforestation_area_ha: Deforested area in hectares
        biomass_density_t_per_ha: Biomass density in tons per hectare
        carbon_fraction: Fraction of biomass that is carbon (typically ~0.47)
    
    Returns:
        Carbon loss in tons CO2 equivalent
    """
    # Calculate biomass loss
    biomass_loss_t = deforestation_area_ha * biomass_density_t_per_ha
    
    # Calculate carbon loss
    carbon_loss_t = biomass_loss_t * carbon_fraction
    
    # Convert to CO2 equivalent (multiply by 44/12 for CO2/C ratio)
    co2_loss_t = carbon_loss_t * (44 / 12)
    
    return co2_loss_t


def get_tile_bounds(
    image_shape: Tuple[int, int],
    tile_size: int = 256,
    overlap: int = 32
) -> List[Tuple[int, int, int, int]]:
    """
    Calculate tile boundaries for processing large images
    
    Args:
        image_shape: (height, width) of the image
        tile_size: Size of each tile
        overlap: Overlap between tiles to avoid edge effects
    
    Returns:
        List of (min_row, min_col, max_row, max_col) tuples
    """
    height, width = image_shape
    stride = tile_size - overlap
    
    tiles = []
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            min_row = i
            min_col = j
            max_row = min(i + tile_size, height)
            max_col = min(j + tile_size, width)
            
            tiles.append((min_row, min_col, max_row, max_col))
    
    return tiles


def polygon_to_mask(
    polygon: Polygon,
    transform: rasterio.Affine,
    shape: Tuple[int, int]
) -> np.ndarray:
    """
    Convert polygon to binary mask
    
    Args:
        polygon: Shapely polygon
        transform: Rasterio affine transform
        shape: (height, width) of output mask
    
    Returns:
        Binary mask
    """
    from rasterio.features import geometry_mask
    
    mask = geometry_mask(
        [polygon],
        transform=transform,
        invert=True,
        out_shape=shape
    )
    
    return mask.astype(np.uint8)
