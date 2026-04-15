"""
Sentinel-2 band normalization and preprocessing.

Sentinel-2 L2A surface reflectance is stored as uint16 in range [0, 10000].
We normalize each band to float32 using robust per-channel statistics derived
from a large sample of Amazon/Congo forest and non-forest pixels.

Reference band order expected throughout this project:
  index 0 → B04 Red        (~665 nm)
  index 1 → B03 Green      (~560 nm)
  index 2 → B02 Blue       (~490 nm)
  index 3 → B08 NIR        (~842 nm)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel-2 L2A statistics computed from 50 k Amazon/Congo patches
# Values are surface reflectance ×10000, band order [R, G, B, NIR]
# ---------------------------------------------------------------------------
_S2_MEAN = np.array([943.0, 1069.0, 981.0, 2734.0], dtype=np.float32)
_S2_STD  = np.array([590.0,  547.0, 498.0, 1246.0], dtype=np.float32)

# Robust (2nd–98th percentile) clip bounds to suppress sensor artefacts
_S2_P2   = np.array([   0.0,   10.0,   0.0,  100.0], dtype=np.float32)
_S2_P98  = np.array([2500.0, 2500.0, 2200.0, 8000.0], dtype=np.float32)


class Sentinel2Normalizer:
    """
    Normalize a 4-band Sentinel-2 image to zero-mean / unit-variance float32.

    Two modes:
      - 'standard': use pre-computed global statistics (default, fast).
      - 'dataset':  use statistics supplied via `fit()` (accurate per dataset).
    """

    def __init__(self, mode: str = "standard"):
        assert mode in ("standard", "dataset")
        self.mode = mode
        self.mean: np.ndarray = _S2_MEAN.copy()
        self.std: np.ndarray  = _S2_STD.copy()
        self.p2: np.ndarray   = _S2_P2.copy()
        self.p98: np.ndarray  = _S2_P98.copy()
        self._fitted = (mode == "standard")

    # ------------------------------------------------------------------
    def fit(self, images: list[np.ndarray]) -> "Sentinel2Normalizer":
        """Compute statistics from a list of (4, H, W) arrays."""
        all_pixels: list[np.ndarray] = []
        for img in images:
            c, h, w = img.shape
            all_pixels.append(img.reshape(c, -1))
        stacked = np.concatenate(all_pixels, axis=1)  # (4, N)

        self.mean = stacked.mean(axis=1).astype(np.float32)
        self.std  = stacked.std(axis=1).astype(np.float32) + 1e-6
        self.p2   = np.percentile(stacked, 2, axis=1).astype(np.float32)
        self.p98  = np.percentile(stacked, 98, axis=1).astype(np.float32)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize a (4, H, W) uint16 or float32 array to float32.
        Returns values roughly in [-3, 3].
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before normalizing in 'dataset' mode.")

        img = image.astype(np.float32)

        # 1. Clip outliers band-wise
        for b in range(min(4, img.shape[0])):
            img[b] = np.clip(img[b], self.p2[b], self.p98[b])

        # 2. Standardize
        for b in range(min(4, img.shape[0])):
            img[b] = (img[b] - self.mean[b]) / self.std[b]

        return img

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        data = {
            "mean": self.mean.tolist(),
            "std":  self.std.tolist(),
            "p2":   self.p2.tolist(),
            "p98":  self.p98.tolist(),
            "mode": self.mode,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Sentinel2Normalizer":
        data = json.loads(Path(path).read_text())
        obj = cls(mode=data["mode"])
        obj.mean = np.array(data["mean"], dtype=np.float32)
        obj.std  = np.array(data["std"],  dtype=np.float32)
        obj.p2   = np.array(data["p2"],   dtype=np.float32)
        obj.p98  = np.array(data["p98"],  dtype=np.float32)
        obj._fitted = True
        return obj


# ---------------------------------------------------------------------------
# Dataset statistics helper
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
        clear_labels: SCL codes considered clear. Defaults to vegetation, bare soil,
            water, and snow (``[4, 5, 6, 11]``).
        fill_value: Value to replace cloudy pixels with.

    Returns:
        Cloud-masked image with the same shape as *image*.
    """
    if clear_labels is None:
        clear_labels = [4, 5, 6, 11]

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
        "std":  stacked.std(axis=1).tolist(),
        "min":  stacked.min(axis=1).tolist(),
        "max":  stacked.max(axis=1).tolist(),
    }
