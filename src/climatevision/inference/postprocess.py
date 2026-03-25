"""
Post-processing utilities for inference pipeline.

Provides confidence thresholding, output filtering, and anomaly detection
for model predictions before they are returned to users.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PostProcessConfig:
    """Configuration for post-processing operations."""
    confidence_threshold: float = 0.5
    min_region_pixels: int = 100
    anomaly_std_threshold: float = 3.0
    smooth_kernel_size: int = 3
    apply_morphological_ops: bool = True


@dataclass
class PostProcessResult:
    """Result from post-processing operations."""
    mask: np.ndarray
    confidence_map: np.ndarray
    filtered_pixels: int
    anomaly_detected: bool
    anomaly_regions: list[dict[str, Any]]
    quality_score: float


def apply_confidence_threshold(
    predictions: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Filter predictions below confidence threshold.

    Args:
        predictions: Model prediction mask (H, W) or (H, W, C)
        confidence: Confidence scores (H, W)
        threshold: Minimum confidence to keep prediction

    Returns:
        Filtered prediction mask with low-confidence pixels zeroed
    """
    mask = confidence >= threshold
    filtered = predictions.copy()

    if filtered.ndim == 2:
        filtered[~mask] = 0
    else:
        filtered[~mask, :] = 0

    filtered_count = (~mask).sum()
    logger.debug(f"Filtered {filtered_count} pixels below threshold {threshold}")

    return filtered


def remove_small_regions(
    mask: np.ndarray,
    min_pixels: int = 100
) -> np.ndarray:
    """
    Remove small isolated regions from segmentation mask.

    Args:
        mask: Binary segmentation mask (H, W)
        min_pixels: Minimum region size to keep

    Returns:
        Cleaned mask with small regions removed
    """
    try:
        from scipy import ndimage
    except ImportError:
        logger.warning("scipy not available, skipping small region removal")
        return mask

    labeled, num_features = ndimage.label(mask)

    cleaned = np.zeros_like(mask)
    for i in range(1, num_features + 1):
        region = labeled == i
        if region.sum() >= min_pixels:
            cleaned[region] = mask[region]

    removed = num_features - len(np.unique(ndimage.label(cleaned)[0])) + 1
    logger.debug(f"Removed {removed} regions smaller than {min_pixels} pixels")

    return cleaned


def detect_anomalies(
    predictions: np.ndarray,
    confidence: np.ndarray,
    std_threshold: float = 3.0
) -> tuple[bool, list[dict[str, Any]]]:
    """
    Detect anomalous predictions that may indicate model issues.

    Args:
        predictions: Model predictions
        confidence: Confidence scores
        std_threshold: Number of standard deviations for anomaly detection

    Returns:
        Tuple of (anomaly_detected, list of anomaly regions)
    """
    anomalies = []

    mean_conf = confidence.mean()
    std_conf = confidence.std()

    if std_conf > 0:
        z_scores = np.abs((confidence - mean_conf) / std_conf)
        anomaly_mask = z_scores > std_threshold

        if anomaly_mask.any():
            anomaly_indices = np.where(anomaly_mask)
            anomalies.append({
                "type": "confidence_outlier",
                "count": int(anomaly_mask.sum()),
                "mean_confidence": float(mean_conf),
                "std_confidence": float(std_conf),
                "threshold": std_threshold
            })

    # Check for suspiciously uniform predictions
    unique_values = len(np.unique(predictions))
    if unique_values == 1 and predictions.size > 1000:
        anomalies.append({
            "type": "uniform_prediction",
            "message": "Model returned uniform predictions across entire region"
        })

    return len(anomalies) > 0, anomalies


def compute_quality_score(
    confidence: np.ndarray,
    filtered_ratio: float,
    anomaly_detected: bool
) -> float:
    """
    Compute overall quality score for the prediction.

    Args:
        confidence: Confidence map
        filtered_ratio: Ratio of pixels filtered out
        anomaly_detected: Whether anomalies were detected

    Returns:
        Quality score between 0 and 1
    """
    mean_confidence = float(confidence.mean())
    confidence_score = min(mean_confidence, 1.0)

    filter_penalty = filtered_ratio * 0.3
    anomaly_penalty = 0.2 if anomaly_detected else 0.0

    quality = max(0.0, confidence_score - filter_penalty - anomaly_penalty)
    return round(quality, 4)


def postprocess_predictions(
    predictions: np.ndarray,
    confidence: np.ndarray,
    config: Optional[PostProcessConfig] = None
) -> PostProcessResult:
    """
    Apply full post-processing pipeline to model predictions.

    Args:
        predictions: Raw model predictions (H, W) or (H, W, C)
        confidence: Confidence scores (H, W)
        config: Post-processing configuration

    Returns:
        PostProcessResult with filtered predictions and quality metrics
    """
    if config is None:
        config = PostProcessConfig()

    original_pixels = predictions.size

    # Apply confidence threshold
    filtered = apply_confidence_threshold(
        predictions, confidence, config.confidence_threshold
    )

    # Remove small regions for binary masks
    if filtered.ndim == 2:
        filtered = remove_small_regions(filtered, config.min_region_pixels)

    # Detect anomalies
    anomaly_detected, anomaly_regions = detect_anomalies(
        predictions, confidence, config.anomaly_std_threshold
    )

    # Compute metrics
    filtered_pixels = int((filtered == 0).sum())
    filtered_ratio = filtered_pixels / max(original_pixels, 1)

    quality_score = compute_quality_score(
        confidence, filtered_ratio, anomaly_detected
    )

    if anomaly_detected:
        logger.warning(f"Anomalies detected in prediction: {anomaly_regions}")

    return PostProcessResult(
        mask=filtered,
        confidence_map=confidence,
        filtered_pixels=filtered_pixels,
        anomaly_detected=anomaly_detected,
        anomaly_regions=anomaly_regions,
        quality_score=quality_score
    )
