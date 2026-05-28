"""
Sentinel-2 band normalization and preprocessing.

Sentinel-2 L2A surface reflectance is stored as uint16 in range [0, 10000].
Normalizers are band-agnostic and adapt to any number of input channels.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Sentinel-2 statistics — used as fallback when no dataset stats exist
# Band order MUST match the actual input bands (not hardcoded to RGB+NIR)
# ---------------------------------------------------------------------------


def _default_stats(num_bands: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sensible default mean/std/p2/p98 for `num_bands`."""
    mean = np.full(num_bands, 1500.0, dtype=np.float32)
    std = np.full(num_bands, 800.0, dtype=np.float32)
    p2 = np.full(num_bands, 0.0, dtype=np.float32)
    p98 = np.full(num_bands, 4000.0, dtype=np.float32)
    return mean, std, p2, p98


# ---------------------------------------------------------------------------
# Sentinel-2 Normalizer (band-agnostic)
# ---------------------------------------------------------------------------

class Sentinel2Normalizer:
    """
    Normalize a multi-band Sentinel-2 image to zero-mean / unit-variance float32.

    Two modes:
      - 'standard': use pre-computed or default global statistics (fast).
      - 'dataset':  use statistics supplied via `fit()` (accurate per dataset).
    """

    def __init__(self, mode: str = "standard", num_bands: int = 4):
        assert mode in ("standard", "dataset")
        self.mode = mode
        self.num_bands = num_bands
        self.mean, self.std, self.p2, self.p98 = _default_stats(num_bands)
        self._fitted = (mode == "standard")

    def fit(self, images: list[np.ndarray]) -> "Sentinel2Normalizer":
        """Compute statistics from a list of (C, H, W) arrays."""
        if not images:
            raise ValueError("Cannot fit on empty image list")

        all_pixels: list[np.ndarray] = []
        for img in images:
            c, h, w = img.shape
            all_pixels.append(img.reshape(c, -1))
        stacked = np.concatenate(all_pixels, axis=1)  # (C, N)

        self.num_bands = stacked.shape[0]
        self.mean = stacked.mean(axis=1).astype(np.float32)
        self.std = stacked.std(axis=1).astype(np.float32) + 1e-6
        self.p2 = np.percentile(stacked, 2, axis=1).astype(np.float32)
        self.p98 = np.percentile(stacked, 98, axis=1).astype(np.float32)
        self._fitted = True
        return self

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize a (C, H, W) uint16 or float32 array to float32.
        Returns values roughly in [-3, 3].
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before normalizing in 'dataset' mode.")

        img = image.astype(np.float32)
        c = img.shape[0]

        # Resize stats if band count mismatch (e.g., 3-band vs 5-band)
        if c != self.num_bands:
            if c > self.num_bands:
                # Pad stats with defaults for extra bands
                extra = c - self.num_bands
                self.mean = np.concatenate([self.mean, np.full(extra, 1500.0, dtype=np.float32)])
                self.std = np.concatenate([self.std, np.full(extra, 800.0, dtype=np.float32)])
                self.p2 = np.concatenate([self.p2, np.full(extra, 0.0, dtype=np.float32)])
                self.p98 = np.concatenate([self.p98, np.full(extra, 4000.0, dtype=np.float32)])
            else:
                self.mean = self.mean[:c]
                self.std = self.std[:c]
                self.p2 = self.p2[:c]
                self.p98 = self.p98[:c]
            self.num_bands = c

        # 1. Clip outliers band-wise
        for b in range(c):
            img[b] = np.clip(img[b], self.p2[b], self.p98[b])

        # 2. Standardize
        for b in range(c):
            img[b] = (img[b] - self.mean[b]) / self.std[b]

        return img

    def save(self, path: str | Path) -> None:
        data = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "p2": self.p2.tolist(),
            "p98": self.p98.tolist(),
            "mode": self.mode,
            "num_bands": self.num_bands,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Sentinel2Normalizer":
        data = json.loads(Path(path).read_text())
        obj = cls(mode=data["mode"], num_bands=data.get("num_bands", 4))
        obj.mean = np.array(data["mean"], dtype=np.float32)
        obj.std = np.array(data["std"], dtype=np.float32)
        obj.p2 = np.array(data["p2"], dtype=np.float32)
        obj.p98 = np.array(data["p98"], dtype=np.float32)
        obj._fitted = True
        return obj


# ---------------------------------------------------------------------------
# Cloud masking
# ---------------------------------------------------------------------------

def apply_scl_cloud_mask(
    image: np.ndarray,
    scl_band: np.ndarray,
    clear_labels: Optional[list[int]] = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Mask cloudy pixels in a multi-band image using the Sentinel-2 SCL band.

    Args:
        image: Array of shape (C, H, W).
        scl_band: Array of shape (H, W) containing Scene Classification Layer values.
        clear_labels: SCL codes considered clear. If None, uses a safe default
            of [4, 5, 6] (vegetation, bare soil, water).
        fill_value: Value to replace cloudy pixels with.

    Returns:
        Cloud-masked image with the same shape as *image*.
    """
    if clear_labels is None:
        clear_labels = [4, 5, 6]

    if image.ndim != 3:
        raise ValueError(f"image must be 3-D (C, H, W), got shape {image.shape}")
    if scl_band.shape != image.shape[1:]:
        raise ValueError(
            f"scl_band shape {scl_band.shape} must match image spatial dimensions "
            f"{image.shape[1:]}"
        )

    clear_mask = np.isin(scl_band, clear_labels)
    masked = image.copy()
    masked[:, ~clear_mask] = fill_value
    return masked


# ---------------------------------------------------------------------------
# Band resampling
# ---------------------------------------------------------------------------

def resample_20m_to_10m(
    band_20m: np.ndarray,
    target_shape: tuple[int, int],
    order: int = 1,
) -> np.ndarray:
    """
    Resample a 20m band to 10m resolution using bilinear interpolation.

    Args:
        band_20m: Array of shape (H, W) or (C, H, W).
        target_shape: Desired (H, W) at 10m resolution (2× the 20m dimensions).
        order: 0=nearest, 1=bilinear, 3=bicubic. Default 1 (bilinear).

    Returns:
        Resampled array with the same number of dimensions as input.
    """
    try:
        from scipy.ndimage import zoom
    except ImportError:
        raise ImportError("scipy is required for band resampling. Install: pip install scipy")

    if band_20m.ndim == 2:
        h, w = band_20m.shape
        zoom_factors = (target_shape[0] / h, target_shape[1] / w)
        return zoom(band_20m, zoom_factors, order=order, mode="reflect")
    elif band_20m.ndim == 3:
        c, h, w = band_20m.shape
        zoom_factors = (1.0, target_shape[0] / h, target_shape[1] / w)
        return zoom(band_20m, zoom_factors, order=order, mode="reflect")
    else:
        raise ValueError(f"band_20m must be 2-D or 3-D, got shape {band_20m.shape}")


# ---------------------------------------------------------------------------
# Dataset statistics helper
# ---------------------------------------------------------------------------

def compute_dataset_stats(
    image_dir: str | Path,
    max_samples: int = 500,
) -> dict[str, list[float]]:
    """
    Compute per-channel mean/std from GeoTIFF images in a directory.
    Returns a dict suitable for logging or saving as JSON.
    """
    import rasterio

    image_dir = Path(image_dir)
    paths = sorted(image_dir.glob("*.tif"))[:max_samples]
    if not paths:
        raise FileNotFoundError(f"No .tif files found in {image_dir}")

    all_pixels: list[np.ndarray] = []
    for p in paths:
        with rasterio.open(p) as src:
            img = src.read()  # (C, H, W)
        all_pixels.append(img.reshape(img.shape[0], -1))

    stacked = np.concatenate(all_pixels, axis=1).astype(np.float32)  # (C, N)
    return {
        "mean": stacked.mean(axis=1).tolist(),
        "std": stacked.std(axis=1).tolist(),
        "min": stacked.min(axis=1).tolist(),
        "max": stacked.max(axis=1).tolist(),
    }
