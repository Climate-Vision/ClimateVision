"""
Custom data transforms for ClimateVision pipeline.

Satellite imagery-specific augmentations and preprocessing.
"""
from __future__ import annotations

import logging
from typing import Tuple, Optional, Callable, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None):
        for t in self.transforms:
            if mask is not None:
                image, mask = t(image, mask)
            else:
                image = t(image)
        return (image, mask) if mask is not None else image


class RandomHorizontalFlip:
    """Randomly flip image horizontally."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None):
        if np.random.random() < self.p:
            image = np.flip(image, axis=-1).copy()
            if mask is not None:
                mask = np.flip(mask, axis=-1).copy()
        return (image, mask) if mask is not None else image


class RandomVerticalFlip:
    """Randomly flip image vertically."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None):
        if np.random.random() < self.p:
            image = np.flip(image, axis=-2).copy()
            if mask is not None:
                mask = np.flip(mask, axis=-2).copy()
        return (image, mask) if mask is not None else image


class RandomRotate90:
    """Randomly rotate image by 90 degree increments."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None):
        if np.random.random() < self.p:
            k = np.random.randint(1, 4)  # 90, 180, or 270 degrees
            image = np.rot90(image, k, axes=(-2, -1)).copy()
            if mask is not None:
                mask = np.rot90(mask, k, axes=(-2, -1)).copy()
        return (image, mask) if mask is not None else image


class RandomBrightnessContrast:
    """Randomly adjust brightness and contrast."""

    def __init__(
        self,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        p: float = 0.5,
    ):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.p = p

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None):
        if np.random.random() < self.p:
            alpha = 1.0 + np.random.uniform(-self.contrast_limit, self.contrast_limit)
            beta = np.random.uniform(-self.brightness_limit, self.brightness_limit)
            image = np.clip(alpha * image + beta, 0, 1)
        return (image, mask) if mask is not None else image


class RandomGaussianNoise:
    """Add random Gaussian noise."""

    def __init__(self, var_limit: Tuple[float, float] = (0.001, 0.01), p: float = 0.5):
        self.var_limit = var_limit
        self.p = p

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None):
        if np.random.random() < self.p:
            var = np.random.uniform(*self.var_limit)
            noise = np.random.normal(0, var ** 0.5, image.shape)
            image = np.clip(image + noise, 0, 1)
        return (image, mask) if mask is not None else image


class Normalize:
    """Normalize image with mean and std."""

    def __init__(
        self,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406, 0.5),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225, 0.25),
    ):
        self.mean = np.array(mean).reshape(-1, 1, 1)
        self.std = np.array(std).reshape(-1, 1, 1)

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None):
        # Handle different number of channels
        mean = self.mean[:image.shape[0]]
        std = self.std[:image.shape[0]]
        image = (image - mean) / std
        return (image, mask) if mask is not None else image


class Denormalize:
    """Reverse normalization for visualization."""

    def __init__(
        self,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406, 0.5),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225, 0.25),
    ):
        self.mean = np.array(mean).reshape(-1, 1, 1)
        self.std = np.array(std).reshape(-1, 1, 1)

    def __call__(self, image: np.ndarray):
        mean = self.mean[:image.shape[0]]
        std = self.std[:image.shape[0]]
        return image * std + mean


class ToFloat:
    """Convert image to float32 and scale to [0, 1]."""

    def __init__(self, max_value: float = 255.0):
        self.max_value = max_value

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None):
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / self.max_value
        return (image, mask) if mask is not None else image


class CloudMask:
    """Apply cloud masking to satellite imagery."""

    def __init__(self, cloud_threshold: float = 0.3, fill_value: float = 0.0):
        self.cloud_threshold = cloud_threshold
        self.fill_value = fill_value

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None):
        # Simple brightness-based cloud detection
        if image.shape[0] >= 3:
            brightness = np.mean(image[:3], axis=0)
            cloud_mask = brightness > self.cloud_threshold
            image = image.copy()
            image[:, cloud_mask] = self.fill_value
        return (image, mask) if mask is not None else image


def get_training_transforms() -> Compose:
    """Get default training transforms."""
    return Compose([
        ToFloat(),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        RandomBrightnessContrast(p=0.3),
        RandomGaussianNoise(p=0.2),
        Normalize(),
    ])


def get_validation_transforms() -> Compose:
    """Get default validation transforms."""
    return Compose([
        ToFloat(),
        Normalize(),
    ])
