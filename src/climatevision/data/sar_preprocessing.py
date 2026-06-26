"""
Sentinel-1 SAR preprocessing for flood detection.

Handles speckle filtering, terrain flattening, and backscatter conversion
for C-band VV/VH imagery from COPERNICUS/S1_GRD.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Speckle filtering
# ---------------------------------------------------------------------------

class RefinedLeeSpeckleFilter:
    """
    Refined Lee adaptive speckle filter for SAR imagery.

    Uses local statistics (mean, variance) within a sliding window to
    adaptively smooth homogeneous regions while preserving edges.

    Reference:
        Lee, J.-S. (1981). Speckle analysis and smoothing of synthetic
        aperture radar images. Computer Graphics and Image Processing.
    """

    def __init__(self, window_size: int = 7, num_looks: float = 1.0):
        assert window_size % 2 == 1, "window_size must be odd"
        self.window_size = window_size
        self.half = window_size // 2
        self.num_looks = num_looks
        self.cu = 1.0 / np.sqrt(num_looks)  # theoretical speckle std/mean

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply filter to a (H, W) or (C, H, W) array.

        Args:
            image: Linear intensity or amplitude image (NOT dB).

        Returns:
            Filtered image with same shape.
        """
        if image.ndim == 2:
            return self._filter_band(image)
        elif image.ndim == 3:
            return np.stack([self._filter_band(image[i]) for i in range(image.shape[0])], axis=0)
        else:
            raise ValueError(f"image must be 2-D or 3-D, got shape {image.shape}")

    def _filter_band(self, band: np.ndarray) -> np.ndarray:
        from scipy.ndimage import uniform_filter

        band = band.astype(np.float64)
        h, w = band.shape

        # Local mean and mean-of-squares
        mean = uniform_filter(band, size=self.window_size, mode="reflect")
        mean_sq = uniform_filter(band ** 2, size=self.window_size, mode="reflect")
        var = mean_sq - mean ** 2
        var = np.clip(var, 0, None)

        std = np.sqrt(var)
        cv = std / (mean + 1e-8)  # coefficient of variation

        # Refined Lee weights
        # Three cases: homogeneous, heterogeneous, point target
        cu2 = self.cu ** 2
        cmax2 = 2.0 * cu2  # upper threshold for heterogeneous region

        weight = np.zeros_like(band)
        homogeneous = cv <= self.cu
        heterogeneous = (cv > self.cu) & (cv < np.sqrt(cmax2))
        point_target = cv >= np.sqrt(cmax2)

        # Homogeneous: full filtering
        weight[homogeneous] = 1.0
        # Heterogeneous: adaptive weight
        weight[heterogeneous] = (cu2 * (cv[heterogeneous] ** 2 - cu2)) / (
            cv[heterogeneous] ** 2 * (cmax2 - cu2) + 1e-8
        )
        # Point target: no filtering
        weight[point_target] = 0.0

        filtered = mean + weight * (band - mean)
        return filtered.astype(np.float32)


# ---------------------------------------------------------------------------
# Backscatter conversion
# ---------------------------------------------------------------------------

def linear_to_db(image: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Convert linear intensity/amplitude to decibel scale."""
    return 10.0 * np.log10(np.clip(image, eps, None))


def db_to_linear(image_db: np.ndarray) -> np.ndarray:
    """Convert decibel scale back to linear intensity."""
    return 10.0 ** (image_db / 10.0)


# ---------------------------------------------------------------------------
# Terrain masking
# ---------------------------------------------------------------------------

def apply_slope_mask(
    sar_image: np.ndarray,
    dem_slope: np.ndarray,
    max_slope_deg: float = 15.0,
) -> np.ndarray:
    """
    Mask steep slopes where SAR layover/shadow corrupts flood detection.

    Args:
        sar_image: (C, H, W) or (H, W) SAR image.
        dem_slope: (H, W) slope in degrees from DEM.
        max_slope_deg: Pixels with slope > this are masked to NaN.

    Returns:
        Masked SAR image.
    """
    steep = dem_slope > max_slope_deg
    masked = sar_image.copy()
    if masked.ndim == 3:
        masked[:, steep] = np.nan
    else:
        masked[steep] = np.nan
    return masked


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_sar(
    image: np.ndarray,
    apply_filter: bool = True,
    to_db: bool = True,
    dem_slope: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Full SAR preprocessing pipeline.

    Args:
        image: (C, H, W) array with VV/VH in linear intensity.
        apply_filter: Apply Refined Lee speckle filter.
        to_db: Convert output to decibel scale.
        dem_slope: Optional (H, W) slope mask.

    Returns:
        Preprocessed (C, H, W) array.
    """
    out = image.astype(np.float32)

    if apply_filter:
        flt = RefinedLeeSpeckleFilter(window_size=7)
        out = flt(out)

    if to_db:
        out = linear_to_db(out)

    if dem_slope is not None:
        out = apply_slope_mask(out, dem_slope)

    return out
