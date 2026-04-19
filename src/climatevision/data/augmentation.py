"""
Data augmentation pipeline for Sentinel-2 satellite imagery.

Compatible with albumentations >= 2.0 (always_apply removed, use p=1.0).
"""
from __future__ import annotations

import albumentations as A
import numpy as np


def get_train_transforms(image_size: int = 256) -> A.Compose:
    return A.Compose(
        [
            # --- Geometry ---
            A.RandomCrop(height=image_size, width=image_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.3),

            # Elastic / grid distortion simulates terrain warp
            A.OneOf(
                [
                    A.ElasticTransform(alpha=120, sigma=6, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                    A.OpticalDistortion(distort_limit=0.2, p=1.0),
                ],
                p=0.3,
            ),

            # Coarse dropout simulates cloud / cloud-shadow occlusion
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill_value=0,
                p=0.3,
            ),

            # --- Radiometric / spectral ---
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.4),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ],
                p=0.2,
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        ],
        additional_targets={"mask": "mask"},
    )


def get_val_transforms(image_size: int = 256) -> A.Compose:
    return A.Compose(
        [
            A.CenterCrop(height=image_size, width=image_size, p=1.0),
        ],
        additional_targets={"mask": "mask"},
    )


# TTA transforms — constructed lazily to avoid module-level side effects
def _build_tta_transforms() -> list:
    return [
        A.Compose([]),
        A.Compose([A.HorizontalFlip(p=1.0)]),
        A.Compose([A.VerticalFlip(p=1.0)]),
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]),
        A.Compose([A.RandomRotate90(p=1.0)]),
    ]


TTA_TRANSFORMS = None  # Loaded on first use via get_tta_transforms()


def get_tta_transforms() -> list:
    global TTA_TRANSFORMS
    if TTA_TRANSFORMS is None:
        TTA_TRANSFORMS = _build_tta_transforms()
    return TTA_TRANSFORMS


TTA_INVERSE = [
    lambda x: x,
    lambda x: np.flip(x, axis=-1).copy(),
    lambda x: np.flip(x, axis=-2).copy(),
    lambda x: np.flip(np.flip(x, axis=-1), axis=-2).copy(),
    lambda x: np.rot90(x, k=-1, axes=(-2, -1)).copy(),
]
