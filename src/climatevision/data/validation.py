"""
Data validation utilities for ClimateVision pipeline.

Validates input data integrity before training or inference.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


def validate_image_shape(
    image: np.ndarray,
    expected_channels: int = 4,
    expected_size: Tuple[int, int] = (256, 256),
) -> bool:
    """
    Validate that an image has the expected shape.

    Args:
        image: Input image array (C, H, W) or (H, W, C)
        expected_channels: Expected number of channels (default 4 for Sentinel-2 RGBNIR)
        expected_size: Expected (height, width)

    Returns:
        True if valid

    Raises:
        DataValidationError: If validation fails
    """
    if image.ndim != 3:
        raise DataValidationError(f"Expected 3D array, got {image.ndim}D")

    # Handle both (C, H, W) and (H, W, C) formats
    if image.shape[0] == expected_channels:
        h, w = image.shape[1], image.shape[2]
    elif image.shape[2] == expected_channels:
        h, w = image.shape[0], image.shape[1]
    else:
        raise DataValidationError(
            f"Expected {expected_channels} channels, got shape {image.shape}"
        )

    if (h, w) != expected_size:
        raise DataValidationError(
            f"Expected size {expected_size}, got ({h}, {w})"
        )

    return True


def validate_mask_shape(
    mask: np.ndarray,
    expected_size: Tuple[int, int] = (256, 256),
) -> bool:
    """
    Validate that a segmentation mask has the expected shape.

    Args:
        mask: Input mask array (H, W) or (1, H, W)
        expected_size: Expected (height, width)

    Returns:
        True if valid

    Raises:
        DataValidationError: If validation fails
    """
    if mask.ndim == 3:
        if mask.shape[0] != 1:
            raise DataValidationError(f"Expected single-channel mask, got {mask.shape}")
        h, w = mask.shape[1], mask.shape[2]
    elif mask.ndim == 2:
        h, w = mask.shape
    else:
        raise DataValidationError(f"Expected 2D or 3D mask, got {mask.ndim}D")

    if (h, w) != expected_size:
        raise DataValidationError(
            f"Expected mask size {expected_size}, got ({h}, {w})"
        )

    return True


def validate_value_range(
    array: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0,
    name: str = "array",
) -> bool:
    """
    Validate that array values are within expected range.

    Args:
        array: Input array
        min_val: Minimum expected value
        max_val: Maximum expected value
        name: Name for error messages

    Returns:
        True if valid

    Raises:
        DataValidationError: If validation fails
    """
    actual_min = float(np.min(array))
    actual_max = float(np.max(array))

    if actual_min < min_val or actual_max > max_val:
        raise DataValidationError(
            f"{name} values out of range [{min_val}, {max_val}]: "
            f"got [{actual_min:.4f}, {actual_max:.4f}]"
        )

    return True


def validate_no_nan(array: np.ndarray, name: str = "array") -> bool:
    """
    Validate that array contains no NaN values.

    Args:
        array: Input array
        name: Name for error messages

    Returns:
        True if valid

    Raises:
        DataValidationError: If NaN values found
    """
    nan_count = np.isnan(array).sum()
    if nan_count > 0:
        raise DataValidationError(
            f"{name} contains {nan_count} NaN values"
        )
    return True


def validate_dataset_split(
    data_dir: Path,
    required_splits: List[str] = ["train", "val"],
) -> bool:
    """
    Validate that a dataset directory has the required splits.

    Args:
        data_dir: Path to dataset directory
        required_splits: List of required split names

    Returns:
        True if valid

    Raises:
        DataValidationError: If splits are missing
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise DataValidationError(f"Dataset directory not found: {data_dir}")

    missing = []
    for split in required_splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            missing.append(split)

    if missing:
        raise DataValidationError(
            f"Missing dataset splits: {missing} in {data_dir}"
        )

    return True


def validate_sample(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    expected_channels: int = 4,
    expected_size: Tuple[int, int] = (256, 256),
) -> bool:
    """
    Validate a complete sample (image and optional mask).

    Args:
        image: Input image array
        mask: Optional segmentation mask
        expected_channels: Expected number of image channels
        expected_size: Expected (height, width)

    Returns:
        True if valid

    Raises:
        DataValidationError: If validation fails
    """
    validate_image_shape(image, expected_channels, expected_size)
    validate_no_nan(image, "image")

    if mask is not None:
        validate_mask_shape(mask, expected_size)
        validate_no_nan(mask, "mask")

    logger.debug("Sample validation passed")
    return True
