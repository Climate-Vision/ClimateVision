"""
PyTorch Dataset for forest segmentation from Sentinel-2 GeoTIFF imagery.

Expected directory layout (configurable):
  <root>/
    train/
      images/   *.tif   — 4-band (R, G, B, NIR) float32 / uint16
      masks/    *.tif   — uint8 binary (0=non-forest, 1=forest)
    val/
      images/
      masks/
    test/
      images/
      masks/

Naming convention: image and mask files share the same stem, e.g.
  images/patch_00042.tif  ↔  masks/patch_00042.tif
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level image I/O (rasterio with Pillow fallback)
# ---------------------------------------------------------------------------

def _load_tif(path: Path) -> np.ndarray:
    """Return (C, H, W) float32 array."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            return src.read().astype(np.float32)
    except Exception:
        from PIL import Image
        arr = np.array(Image.open(path)).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis]          # (1, H, W)
        else:
            arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
        return arr


def _load_mask(path: Path) -> np.ndarray:
    """Return (H, W) uint8 array with values {0, 1}."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            mask = src.read(1)
    except Exception:
        from PIL import Image
        mask = np.array(Image.open(path).convert("L"))
    return (mask > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# ForestDataset
# ---------------------------------------------------------------------------

class ForestDataset(Dataset):
    """
    Sentinel-2 forest/non-forest segmentation dataset.

    Args:
        root:        Path containing `images/` and `masks/` sub-directories.
        transform:   albumentations Compose transform (applied to image+mask).
        normalizer:  Sentinel2Normalizer instance (applied after transform).
        image_size:  Spatial size. Images are padded/cropped if needed.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        normalizer: Optional[Callable] = None,
        image_size: int = 256,
        n_channels: int = 4,
    ):
        self.root = Path(root)
        self.transform = transform
        self.normalizer = normalizer
        self.image_size = image_size
        self.n_channels = n_channels

        image_dir = self.root / "images"
        mask_dir  = self.root / "masks"

        stems = sorted(p.stem for p in image_dir.glob("*.tif"))
        self.samples: list[tuple[Path, Path]] = []
        for stem in stems:
            img_path  = image_dir / f"{stem}.tif"
            mask_path = mask_dir  / f"{stem}.tif"
            if mask_path.exists():
                self.samples.append((img_path, mask_path))
            else:
                logger.warning("No mask for %s — skipped.", stem)

        if not self.samples:
            raise FileNotFoundError(
                f"No image/mask pairs found in {self.root}. "
                "Run `python scripts/prepare_data.py` first."
            )
        logger.info("ForestDataset: %d samples from %s", len(self.samples), self.root)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        image = _load_tif(img_path)   # (C, H, W) float32
        mask  = _load_mask(mask_path) # (H, W)    uint8

        # Ensure exactly n_channels bands (pad with zeros if fewer, truncate if more)
        c, h, w = image.shape
        if c < self.n_channels:
            pad = np.zeros((self.n_channels - c, h, w), dtype=np.float32)
            image = np.concatenate([image, pad], axis=0)
        elif c > self.n_channels:
            image = image[:self.n_channels]

        # Ensure spatial size — pad if smaller, random crop via transform
        if h < self.image_size or w < self.image_size:
            image, mask = self._pad(image, mask)

        # albumentations expects (H, W, C)
        image_hwc = np.transpose(image, (1, 2, 0))
        if self.transform is not None:
            result    = self.transform(image=image_hwc, mask=mask)
            image_hwc = result["image"]
            mask      = result["mask"]
        image = np.transpose(image_hwc, (2, 0, 1))  # back to (C, H, W)

        # Normalize to float32 zero-mean / unit-variance
        if self.normalizer is not None:
            image = self.normalizer(image)
        else:
            # Minimal default: divide by 10000 (Sentinel-2 L2A scale)
            image = image / 10000.0

        return (
            torch.tensor(image.copy(), dtype=torch.float32),
            torch.tensor(mask.astype(np.int64).copy(), dtype=torch.int64),
        )

    # ------------------------------------------------------------------
    def _pad(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        c, h, w = image.shape
        ph = max(0, self.image_size - h)
        pw = max(0, self.image_size - w)
        image = np.pad(image, ((0, 0), (0, ph), (0, pw)), mode="reflect")
        mask  = np.pad(mask,  ((0, ph), (0, pw)),          mode="reflect")
        return image, mask

    # ------------------------------------------------------------------
    def compute_class_weights(self) -> torch.Tensor:
        """
        Return [w_non_forest, w_forest] inverse-frequency weights.
        Processes a random subset of 200 samples for speed.
        """
        rng   = np.random.default_rng(42)
        idxs  = rng.choice(len(self.samples), min(200, len(self.samples)), replace=False)
        counts = np.zeros(2, dtype=np.float64)
        for i in idxs:
            _, mask_path = self.samples[i]
            mask = _load_mask(mask_path).flatten()
            counts[0] += (mask == 0).sum()
            counts[1] += (mask == 1).sum()
        total = counts.sum()
        weights = total / (2.0 * counts + 1e-6)
        logger.info(
            "Class weights → non-forest: %.3f  forest: %.3f", weights[0], weights[1]
        )
        return torch.tensor(weights, dtype=torch.float32)

    # ------------------------------------------------------------------
    def make_sampler(self) -> WeightedRandomSampler:
        """
        Weighted sampler that over-samples patches rich in forest pixels.
        This accelerates learning of the minority class.
        """
        sample_weights: list[float] = []
        rng = np.random.default_rng(0)
        for _, mask_path in self.samples:
            mask = _load_mask(mask_path)
            forest_frac = mask.mean()
            # Weight ∝ forest fraction (clamped so fully non-forest patches
            # still appear occasionally)
            sample_weights.append(max(float(forest_frac), 0.05))

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 256,
    normalizer: Optional[Callable] = None,
    pin_memory: bool = True,
    use_weighted_sampler: bool = True,
    n_channels: int = 4,
) -> dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from a data directory.

    Args:
        data_dir:             Root directory containing train/, val/, test/.
        batch_size:           Samples per batch.
        num_workers:          DataLoader worker processes.
        image_size:           Spatial size after cropping.
        normalizer:           Sentinel2Normalizer instance.
        pin_memory:           Pin CPU tensors for faster GPU transfer.
        use_weighted_sampler: Over-sample forest-rich patches during training.

    Returns:
        dict with keys 'train', 'val', 'test'.
    """
    from .augmentation import get_train_transforms, get_val_transforms

    data_dir = Path(data_dir)
    loaders: dict[str, DataLoader] = {}

    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            logger.warning("Split directory %s not found — skipped.", split_dir)
            continue

        is_train = split == "train"
        transform = get_train_transforms(image_size) if is_train else get_val_transforms(image_size)

        dataset = ForestDataset(
            root=split_dir,
            transform=transform,
            normalizer=normalizer,
            image_size=image_size,
            n_channels=n_channels,
        )

        sampler = None
        shuffle = is_train
        if is_train and use_weighted_sampler:
            sampler = dataset.make_sampler()
            shuffle = False  # sampler is mutually exclusive with shuffle

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=is_train,
            persistent_workers=(num_workers > 0),
        )

    return loaders
