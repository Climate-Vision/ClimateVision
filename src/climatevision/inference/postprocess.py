"""
Inference post-processing utilities.

Provides:
- Confidence thresholding and small-region removal
- Sliding-window tiling for large-region inference
- Anomaly detection and quality scoring
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence thresholding
# ---------------------------------------------------------------------------

def apply_confidence_threshold(
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Mask out predictions where max class probability is below threshold.

    Args:
        probabilities: (C, H, W) softmax probabilities.
        threshold: Minimum confidence to keep a prediction.

    Returns:
        (H, W) integer mask with -1 for low-confidence pixels.
    """
    max_prob = probabilities.max(axis=0)
    predictions = probabilities.argmax(axis=0)
    predictions[max_prob < threshold] = -1
    return predictions


# ---------------------------------------------------------------------------
# Small-region removal
# ---------------------------------------------------------------------------

def remove_small_regions(
    mask: np.ndarray,
    min_size: int = 50,
    connectivity: int = 1,
) -> np.ndarray:
    """
    Remove connected components smaller than min_size pixels.

    Args:
        mask: (H, W) binary or integer label mask.
        min_size: Minimum component size to retain.
        connectivity: 1 for 4-connectivity, 2 for 8-connectivity.

    Returns:
        Cleaned mask with small regions removed (set to 0).
    """
    try:
        from scipy import ndimage
    except ImportError:
        logger.warning("scipy not available; skipping small-region removal")
        return mask

    cleaned = mask.copy()
    unique_labels = np.unique(mask)
    for label in unique_labels:
        if label == 0:
            continue
        binary = (mask == label).astype(np.uint8)
        labeled, num_features = ndimage.label(binary, structure=ndimage.generate_binary_structure(2, connectivity))
        component_sizes = ndimage.sum(binary, labeled, index=range(1, num_features + 1))
        too_small = component_sizes < min_size
        remove_mask = too_small[labeled - 1]
        cleaned[(mask == label) & remove_mask] = 0

    return cleaned


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    prediction: np.ndarray,
    expected_classes: list[int],
) -> dict[str, Any]:
    """
    Flag anomalous predictions (unexpected class labels, extreme coverage).

    Args:
        prediction: (H, W) integer mask.
        expected_classes: List of valid class indices.

    Returns:
        Dict with anomaly flags and descriptions.
    """
    anomalies = []
    unique = np.unique(prediction)
    unexpected = set(unique) - set(expected_classes)
    if unexpected:
        anomalies.append(f"Unexpected class labels: {unexpected}")

    total = prediction.size
    for cls in expected_classes:
        pct = (prediction == cls).sum() / total * 100
        if pct > 95:
            anomalies.append(f"Class {cls} dominates ({pct:.1f}% — possible model collapse)")

    return {
        "is_anomalous": len(anomalies) > 0,
        "anomalies": anomalies,
    }


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def compute_quality_score(
    prediction: np.ndarray,
    confidence: np.ndarray,
) -> float:
    """
    Compute an overall quality score [0, 1] for a prediction.

    Based on mean confidence and class balance.
    """
    mean_conf = float(confidence.mean())
    total = prediction.size
    class_balance = 1.0
    unique, counts = np.unique(prediction, return_counts=True)
    if len(unique) > 1:
        max_pct = counts.max() / total
        class_balance = 1.0 - max(0, max_pct - 0.8) / 0.2  # penalize if one class > 80%

    return round(mean_conf * 0.7 + class_balance * 0.3, 4)


# ---------------------------------------------------------------------------
# Sliding-window tiling for large regions
# ---------------------------------------------------------------------------

class SlidingWindowTiler:
    """
    Tile large images into overlapping patches for inference,
    then stitch predictions back with soft-voting in overlap regions.
    """

    def __init__(
        self,
        tile_size: int = 256,
        overlap: int = 32,
    ):
        assert overlap < tile_size, "overlap must be smaller than tile_size"
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap

    def generate_tiles(
        self,
        image: np.ndarray,
    ) -> list[tuple[int, int, np.ndarray]]:
        """
        Generate (row, col, tile) tuples from a (C, H, W) image.

        row and col are the top-left coordinates in the original image.
        """
        c, h, w = image.shape
        tiles = []
        for y in range(0, h - self.tile_size + 1, self.stride):
            for x in range(0, w - self.tile_size + 1, self.stride):
                tile = image[:, y:y + self.tile_size, x:x + self.tile_size]
                tiles.append((y, x, tile))

        # Handle edge cases: last row/col if not perfectly divisible
        if h % self.stride != 0 and h > self.tile_size:
            y = h - self.tile_size
            for x in range(0, w - self.tile_size + 1, self.stride):
                tile = image[:, y:y + self.tile_size, x:x + self.tile_size]
                tiles.append((y, x, tile))
        if w % self.stride != 0 and w > self.tile_size:
            x = w - self.tile_size
            for y in range(0, h - self.tile_size + 1, self.stride):
                tile = image[:, y:y + self.tile_size, x:x + self.tile_size]
                tiles.append((y, x, tile))
        if h % self.stride != 0 and w % self.stride != 0 and h > self.tile_size and w > self.tile_size:
            y = h - self.tile_size
            x = w - self.tile_size
            tile = image[:, y:y + self.tile_size, x:x + self.tile_size]
            tiles.append((y, x, tile))

        return tiles

    def stitch_predictions(
        self,
        tiles: list[tuple[int, int, np.ndarray]],
        image_shape: tuple[int, int, int],
        num_classes: int,
    ) -> np.ndarray:
        """
        Stitch per-tile predictions into a full-size prediction mask.

        Args:
            tiles: List of (row, col, pred_mask) where pred_mask is (H, W) int.
            image_shape: (C, H, W) original image shape.
            num_classes: Number of classes.

        Returns:
            (H, W) stitched prediction mask.
        """
        c, h, w = image_shape
        vote_counts = np.zeros((num_classes, h, w), dtype=np.float32)

        for y, x, pred in tiles:
            th, tw = pred.shape
            for cls in range(num_classes):
                vote_counts[cls, y:y + th, x:x + tw] += (pred == cls).astype(np.float32)

        # Class with most votes wins
        stitched = vote_counts.argmax(axis=0)
        return stitched

    def stitch_probabilities(
        self,
        tiles: list[tuple[int, int, np.ndarray]],
        image_shape: tuple[int, int, int],
    ) -> np.ndarray:
        """
        Stitch per-tile probability maps with averaging in overlap regions.

        Args:
            tiles: List of (row, col, probs) where probs is (C, H, W) float.
            image_shape: (C, H, W) original image shape.

        Returns:
            (num_classes, H, W) averaged probability map.
        """
        c, h, w = image_shape
        num_classes = tiles[0][2].shape[0]
        prob_sum = np.zeros((num_classes, h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)

        for y, x, probs in tiles:
            _, th, tw = probs.shape
            prob_sum[:, y:y + th, x:x + tw] += probs
            count[y:y + th, x:x + tw] += 1.0

        # Avoid divide by zero
        count = np.clip(count, 1e-8, None)
        averaged = prob_sum / count[np.newaxis, :, :]
        return averaged
