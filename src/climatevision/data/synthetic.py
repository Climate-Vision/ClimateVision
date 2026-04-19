"""
Synthetic Sentinel-2 forest patch generator.

Produces realistic 4-band (R, G, B, NIR) imagery with corresponding binary
forest masks using fractal Perlin-noise patterns that capture the spatial
autocorrelation of real tropical forest boundaries.

Statistics match Sentinel-2 L2A surface reflectance (scaled 0–10000):

              Red (B04)   Green (B03)  Blue (B02)  NIR (B08)
  Forest      ~400–900    ~700–1100    ~500–900    ~3000–7000
  Non-forest  ~700–2000   ~800–1500    ~700–1300   ~1000–3000

Usage:
    generate_synthetic_dataset(
        output_dir="data",
        n_train=800,
        n_val=100,
        n_test=100,
        patch_size=256,
    )
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Perlin-noise helpers
# ---------------------------------------------------------------------------

def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def _gradient(h: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Dot product of gradient vector and distance vector."""
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.float32)
    g = vectors[h % 4]
    return g[..., 0] * x + g[..., 1] * y


def _perlin2d(shape: Tuple[int, int], scale: float, rng: np.random.Generator) -> np.ndarray:
    """2D Perlin noise in [-1, 1]."""
    h, w = shape
    x = np.linspace(0, scale, w, endpoint=False)
    y = np.linspace(0, scale, h, endpoint=False)
    xg, yg = np.meshgrid(x, y)

    xi = xg.astype(int)
    yi = yg.astype(int)
    xf = xg - xi
    yf = yg - yi

    u = _fade(xf)
    v = _fade(yf)

    # Random permutation table
    p = rng.permutation(256).astype(np.int32)
    p = np.stack([p, p]).flatten()  # extend

    aa = p[p[xi    ] + yi    ]
    ab = p[p[xi    ] + yi + 1]
    ba = p[p[xi + 1] + yi    ]
    bb = p[p[xi + 1] + yi + 1]

    x0 = _lerp(_gradient(aa, xf,     yf    ),
               _gradient(ba, xf - 1, yf    ), u)
    x1 = _lerp(_gradient(ab, xf,     yf - 1),
               _gradient(bb, xf - 1, yf - 1), u)
    return _lerp(x0, x1, v)


def _fractal_noise(
    shape: Tuple[int, int],
    rng: np.random.Generator,
    octaves: int = 6,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
    base_scale: float = 4.0,
) -> np.ndarray:
    """Fractal (fBm) noise — sum of Perlin octaves."""
    noise = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    scale = base_scale
    for _ in range(octaves):
        noise += amplitude * _perlin2d(shape, scale, rng)
        total_amp += amplitude
        amplitude *= persistence
        scale *= lacunarity
    return noise / total_amp


# ---------------------------------------------------------------------------
# Patch generation
# ---------------------------------------------------------------------------

def _generate_patch(
    rng: np.random.Generator,
    patch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        image: (4, H, W) float32 Sentinel-2 reflectance ×10000
        mask:  (H, W)    uint8  binary (0=non-forest, 1=forest)
    """
    H = W = patch_size

    # 1. Forest mask via fractal noise threshold
    noise = _fractal_noise((H, W), rng, octaves=6, base_scale=rng.uniform(3, 8))
    # Vary forest fraction: real Amazon has ~60-90% forest, cleared areas <30%
    forest_frac = rng.uniform(0.15, 0.90)
    threshold = np.percentile(noise, (1 - forest_frac) * 100)
    mask = (noise >= threshold).astype(np.uint8)  # 1=forest

    # 2. Add secondary noise for forest texture variation
    texture = _fractal_noise((H, W), rng, octaves=4, base_scale=2.0)

    # 3. Build 4-band reflectance image
    image = np.zeros((4, H, W), dtype=np.float32)
    f = mask.astype(np.float32)        # 1 where forest
    nf = 1.0 - f                       # 1 where non-forest

    # Band-specific forest / non-forest reflectance ranges (mean ± noise)
    # Red (B04)
    image[0] = (
        f  * (rng.normal(600, 80, (H, W))  + texture * 150)
      + nf * (rng.normal(1300, 200, (H, W)) + texture * 300)
    )
    # Green (B03)
    image[1] = (
        f  * (rng.normal(900, 80, (H, W))  + texture * 120)
      + nf * (rng.normal(1200, 150, (H, W)) + texture * 200)
    )
    # Blue (B02)
    image[2] = (
        f  * (rng.normal(700, 60, (H, W))  + texture * 80)
      + nf * (rng.normal(1000, 130, (H, W)) + texture * 150)
    )
    # NIR (B08) — strongest discriminator
    image[3] = (
        f  * (rng.normal(4500, 600, (H, W)) + texture * 800)
      + nf * (rng.normal(1800, 400, (H, W)) + texture * 400)
    )

    # Clip to realistic Sentinel-2 range
    image = np.clip(image, 0, 10000)

    # Occasionally add a cloud-like occlusion (random bright rectangle)
    if rng.random() < 0.12:
        r0 = rng.integers(0, H // 2)
        c0 = rng.integers(0, W // 2)
        rh = rng.integers(20, H // 3)
        rw = rng.integers(20, W // 3)
        cloud_val = rng.uniform(8000, 10000)
        image[:, r0:r0+rh, c0:c0+rw] = cloud_val

    return image.astype(np.float32), mask


# ---------------------------------------------------------------------------
# GeoTIFF writer (rasterio required; falls back to numpy .npy)
# ---------------------------------------------------------------------------

def _write_geotiff(path: Path, data: np.ndarray) -> None:
    """Write (C, H, W) or (H, W) array as GeoTIFF."""
    try:
        import rasterio
        from rasterio.transform import from_bounds

        if data.ndim == 2:
            data = data[np.newaxis]

        c, h, w = data.shape
        transform = from_bounds(0, 0, 1, 1, w, h)
        dtype = "float32" if data.dtype == np.float32 else "uint8"

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=c,
            dtype=dtype,
            crs="EPSG:4326",
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(data)
    except ImportError:
        # Fallback: save as .npy (dataset loader handles this)
        npy_path = path.with_suffix(".npy")
        np.save(npy_path, data)
        logger.warning("rasterio not available; saved as %s", npy_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    output_dir: str | Path = "data",
    n_train: int = 800,
    n_val: int = 100,
    n_test: int = 100,
    patch_size: int = 256,
    seed: int = 42,
) -> None:
    """
    Generate synthetic forest segmentation dataset.

    Output layout:
        <output_dir>/
          train/images/*.tif  train/masks/*.tif
          val/images/*.tif    val/masks/*.tif
          test/images/*.tif   test/masks/*.tif

    Args:
        output_dir:  Root directory to write data into.
        n_train:     Number of training patches.
        n_val:       Number of validation patches.
        n_test:      Number of test patches.
        patch_size:  Spatial size of each patch (pixels).
        seed:        Random seed for reproducibility.
    """
    output_dir = Path(output_dir)
    rng = np.random.default_rng(seed)

    splits = {"train": n_train, "val": n_val, "test": n_test}
    total = sum(splits.values())
    generated = 0

    for split, n in splits.items():
        img_dir  = output_dir / split / "images"
        mask_dir = output_dir / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating %d %s patches …", n, split)

        for i in range(n):
            image, mask = _generate_patch(rng, patch_size)
            stem = f"patch_{i:05d}"
            _write_geotiff(img_dir  / f"{stem}.tif", image)
            _write_geotiff(mask_dir / f"{stem}.tif", mask[np.newaxis].astype(np.float32))
            generated += 1

            if generated % 100 == 0:
                pct = generated / total * 100
                logger.info("  %d / %d patches  (%.0f%%)", generated, total, pct)

    logger.info(
        "Dataset generation complete: %d train, %d val, %d test patches → %s",
        n_train, n_val, n_test, output_dir,
    )
