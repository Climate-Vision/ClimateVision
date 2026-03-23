"""
Data sampling utilities for ClimateVision pipeline.

Provides stratified and balanced sampling for training datasets.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import random

import numpy as np

logger = logging.getLogger(__name__)


def stratified_split(
    samples: List[Path],
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split samples into train/val/test while maintaining class distribution.

    Args:
        samples: List of sample file paths
        labels: Corresponding class labels
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    rng = random.Random(seed)

    # Group samples by class
    class_samples: Dict[int, List[Path]] = {}
    for sample, label in zip(samples, labels):
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(sample)

    train_set: List[Path] = []
    val_set: List[Path] = []
    test_set: List[Path] = []

    # Split each class proportionally
    for label, class_paths in class_samples.items():
        rng.shuffle(class_paths)
        n = len(class_paths)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_set.extend(class_paths[:train_end])
        val_set.extend(class_paths[train_end:val_end])
        test_set.extend(class_paths[val_end:])

    # Shuffle final sets
    rng.shuffle(train_set)
    rng.shuffle(val_set)
    rng.shuffle(test_set)

    logger.info(
        f"Stratified split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}"
    )

    return train_set, val_set, test_set


def balanced_sampler(
    samples: List[Path],
    labels: List[int],
    target_per_class: Optional[int] = None,
    seed: int = 42,
) -> List[Path]:
    """
    Create a balanced sample set with equal representation per class.

    Args:
        samples: List of sample file paths
        labels: Corresponding class labels
        target_per_class: Target samples per class (None = use min class count)
        seed: Random seed for reproducibility

    Returns:
        Balanced list of sample paths
    """
    rng = random.Random(seed)

    # Group by class
    class_samples: Dict[int, List[Path]] = {}
    for sample, label in zip(samples, labels):
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(sample)

    # Determine target count
    if target_per_class is None:
        target_per_class = min(len(v) for v in class_samples.values())

    balanced: List[Path] = []

    for label, class_paths in class_samples.items():
        if len(class_paths) >= target_per_class:
            selected = rng.sample(class_paths, target_per_class)
        else:
            # Oversample with replacement
            selected = rng.choices(class_paths, k=target_per_class)
        balanced.extend(selected)

    rng.shuffle(balanced)

    logger.info(
        f"Balanced sampling: {len(balanced)} total, {target_per_class} per class"
    )

    return balanced


def weighted_sampler_weights(labels: List[int]) -> List[float]:
    """
    Compute sample weights for weighted random sampling.

    Inverse class frequency weighting for handling imbalanced datasets.

    Args:
        labels: List of class labels

    Returns:
        List of weights (one per sample)
    """
    label_counts: Dict[int, int] = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    total = len(labels)
    n_classes = len(label_counts)

    # Inverse frequency weighting
    class_weights = {
        label: total / (n_classes * count)
        for label, count in label_counts.items()
    }

    weights = [class_weights[label] for label in labels]

    logger.info(f"Computed weights for {n_classes} classes")

    return weights


def random_subset(
    samples: List[Path],
    fraction: float = 0.1,
    seed: int = 42,
) -> List[Path]:
    """
    Select a random subset of samples.

    Useful for debugging or quick experiments.

    Args:
        samples: List of sample paths
        fraction: Fraction to select (0-1)
        seed: Random seed

    Returns:
        Subset of samples
    """
    rng = random.Random(seed)
    k = max(1, int(len(samples) * fraction))
    subset = rng.sample(samples, k)

    logger.info(f"Selected {len(subset)}/{len(samples)} samples ({fraction*100:.1f}%)")

    return subset


def kfold_split(
    samples: List[Path],
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[Path], List[Path]]]:
    """
    Generate k-fold cross-validation splits.

    Args:
        samples: List of sample paths
        n_folds: Number of folds
        seed: Random seed

    Returns:
        List of (train, val) tuples for each fold
    """
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)

    fold_size = len(shuffled) // n_folds
    folds = []

    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(shuffled)

        val_fold = shuffled[start:end]
        train_fold = shuffled[:start] + shuffled[end:]

        folds.append((train_fold, val_fold))

    logger.info(f"Created {n_folds}-fold splits")

    return folds
