"""
Data quality assessment for ClimateVision pipeline.

Computes quality metrics for satellite imagery datasets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Data quality assessment report."""
    total_samples: int
    valid_samples: int
    invalid_samples: int
    cloud_coverage_mean: float
    cloud_coverage_std: float
    missing_data_ratio: float
    class_distribution: Dict[int, float]
    issues: List[str]

    @property
    def quality_score(self) -> float:
        """Compute overall quality score (0-100)."""
        if self.total_samples == 0:
            return 0.0

        valid_ratio = self.valid_samples / self.total_samples
        cloud_penalty = min(self.cloud_coverage_mean / 100, 1.0)
        missing_penalty = min(self.missing_data_ratio, 1.0)

        score = valid_ratio * 100 * (1 - cloud_penalty * 0.3) * (1 - missing_penalty * 0.5)
        return round(score, 2)

    def is_acceptable(self, min_score: float = 70.0) -> bool:
        """Check if quality meets minimum threshold."""
        return self.quality_score >= min_score


def estimate_cloud_coverage(image: np.ndarray, threshold: float = 0.3) -> float:
    """
    Estimate cloud coverage percentage from Sentinel-2 image.

    Uses a simple brightness threshold on the visible bands.
    For production, consider using the SCL band or cloud masks.

    Args:
        image: Sentinel-2 image (C, H, W) with RGB in first 3 channels
        threshold: Brightness threshold for cloud detection

    Returns:
        Cloud coverage percentage (0-100)
    """
    if image.ndim != 3:
        return 0.0

    # Use mean of RGB bands
    if image.shape[0] >= 3:
        rgb_mean = np.mean(image[:3], axis=0)
    else:
        rgb_mean = image[0]

    # Normalize to 0-1 if needed
    if rgb_mean.max() > 1.0:
        rgb_mean = rgb_mean / rgb_mean.max()

    cloud_pixels = np.sum(rgb_mean > threshold)
    total_pixels = rgb_mean.size

    return (cloud_pixels / total_pixels) * 100


def compute_class_distribution(mask: np.ndarray) -> Dict[int, float]:
    """
    Compute class distribution in a segmentation mask.

    Args:
        mask: Segmentation mask (H, W) or (1, H, W)

    Returns:
        Dictionary mapping class ID to percentage
    """
    if mask.ndim == 3:
        mask = mask.squeeze(0)

    unique, counts = np.unique(mask, return_counts=True)
    total = counts.sum()

    return {int(cls): round(count / total * 100, 2) for cls, count in zip(unique, counts)}


def check_data_completeness(image: np.ndarray) -> float:
    """
    Check for missing/invalid data in image.

    Args:
        image: Input image array

    Returns:
        Ratio of missing/invalid pixels (0-1)
    """
    total_pixels = image.size

    # Count NaN values
    nan_count = np.isnan(image).sum()

    # Count zero values (potential no-data)
    zero_count = (image == 0).sum()

    # Count extreme values
    if image.max() > 0:
        extreme_count = ((image < -1) | (image > 10)).sum()
    else:
        extreme_count = 0

    invalid_count = nan_count + extreme_count
    return invalid_count / total_pixels


def assess_dataset_quality(
    data_dir: Path,
    split: str = "train",
    sample_size: Optional[int] = None,
) -> QualityReport:
    """
    Assess quality of a dataset split.

    Args:
        data_dir: Path to dataset directory
        split: Split to assess (train/val/test)
        sample_size: Number of samples to check (None = all)

    Returns:
        QualityReport with quality metrics
    """
    data_dir = Path(data_dir)
    split_dir = data_dir / split
    images_dir = split_dir / "images"
    masks_dir = split_dir / "masks"

    issues = []

    if not split_dir.exists():
        return QualityReport(
            total_samples=0,
            valid_samples=0,
            invalid_samples=0,
            cloud_coverage_mean=0.0,
            cloud_coverage_std=0.0,
            missing_data_ratio=1.0,
            class_distribution={},
            issues=[f"Split directory not found: {split_dir}"],
        )

    # Find image files
    image_files = list(images_dir.glob("*.npy")) if images_dir.exists() else []

    if not image_files:
        image_files = list(images_dir.glob("*.tif")) if images_dir.exists() else []

    if not image_files:
        issues.append(f"No image files found in {images_dir}")
        return QualityReport(
            total_samples=0,
            valid_samples=0,
            invalid_samples=0,
            cloud_coverage_mean=0.0,
            cloud_coverage_std=0.0,
            missing_data_ratio=1.0,
            class_distribution={},
            issues=issues,
        )

    # Sample if needed
    if sample_size and sample_size < len(image_files):
        rng = np.random.default_rng(42)
        image_files = list(rng.choice(image_files, size=sample_size, replace=False))

    total = len(image_files)
    valid = 0
    invalid = 0
    cloud_coverages = []
    missing_ratios = []
    all_class_dist: Dict[int, List[float]] = {}

    for img_path in image_files:
        try:
            image = np.load(img_path)

            # Check for mask
            mask_path = masks_dir / img_path.name
            mask = np.load(mask_path) if mask_path.exists() else None

            # Compute metrics
            cloud_cov = estimate_cloud_coverage(image)
            cloud_coverages.append(cloud_cov)

            missing = check_data_completeness(image)
            missing_ratios.append(missing)

            if mask is not None:
                class_dist = compute_class_distribution(mask)
                for cls, pct in class_dist.items():
                    if cls not in all_class_dist:
                        all_class_dist[cls] = []
                    all_class_dist[cls].append(pct)

            valid += 1

        except Exception as e:
            invalid += 1
            logger.warning(f"Error processing {img_path}: {e}")

    # Aggregate metrics
    avg_class_dist = {
        cls: round(np.mean(pcts), 2) for cls, pcts in all_class_dist.items()
    }

    return QualityReport(
        total_samples=total,
        valid_samples=valid,
        invalid_samples=invalid,
        cloud_coverage_mean=round(np.mean(cloud_coverages), 2) if cloud_coverages else 0.0,
        cloud_coverage_std=round(np.std(cloud_coverages), 2) if cloud_coverages else 0.0,
        missing_data_ratio=round(np.mean(missing_ratios), 4) if missing_ratios else 0.0,
        class_distribution=avg_class_dist,
        issues=issues,
    )
