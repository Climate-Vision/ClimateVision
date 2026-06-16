"""
Separate flood water from permanent water.

A single post-event image cannot tell you whether detected water is a flood or a
lake/river that is always there -- the backscatter (or water index) looks the
same. Distinguishing the two requires a *reference* for where water normally is.
This module provides the two standard, defensible ways to get that reference:

  1. Reference subtraction (`classify_with_reference`)
       Overlay a permanent-water layer (e.g. JRC Global Surface Water occurrence
       >= ~50%, or any pre-computed permanent mask). Water that coincides with the
       reference is permanent; water outside it is flood.

  2. Change detection (`classify_with_change`)
       Run water detection on a pre-event scene as well. Water present in both
       pre and post is permanent; water that appears only post-event is flood.

Output classes match FloodingAnalysis.output_classes:
    0 = dry_land, 1 = permanent_water, 2 = flooded

Design choice: when NO reference is available, these functions raise rather than
fabricate a permanent/flood split. A guessed distinction on a disaster-response
product is worse than an honest "water, source unknown".
"""
from __future__ import annotations

import numpy as np

DRY_LAND = 0
PERMANENT_WATER = 1
FLOODED = 2


def classify_with_reference(
    water_mask: np.ndarray,
    permanent_water_ref: np.ndarray,
) -> np.ndarray:
    """Classify detected water against a permanent-water reference.

    Args:
        water_mask: (H, W) binary mask of water detected in the post-event scene
            (1 = water). Typically `EnsembleFloodPipeline.detect(...)["ensemble_mask"]`.
        permanent_water_ref: (H, W) binary mask, 1 where water is normally present
            (e.g. from `permanent_water_from_occurrence`). Must match water_mask shape.

    Returns:
        (H, W) int array: 0=dry, 1=permanent_water, 2=flooded.
    """
    water = _as_bool(water_mask)
    perm = _as_bool(permanent_water_ref)
    if water.shape != perm.shape:
        raise ValueError(
            f"water_mask {water.shape} and permanent_water_ref {perm.shape} must match. "
            "Reproject/resample the reference to the scene grid first."
        )

    out = np.full(water.shape, DRY_LAND, dtype=np.int32)
    out[water & perm] = PERMANENT_WATER
    out[water & ~perm] = FLOODED
    return out


def classify_with_change(
    pre_water_mask: np.ndarray,
    post_water_mask: np.ndarray,
) -> np.ndarray:
    """Classify water by pre/post change detection.

    Water present before *and* after the event is treated as permanent; water
    that appears only after the event is flood. Water that was present before but
    not after (receding / dried out) is returned as dry land.

    Args:
        pre_water_mask: (H, W) binary water mask from the pre-event scene.
        post_water_mask: (H, W) binary water mask from the post-event scene.

    Returns:
        (H, W) int array: 0=dry, 1=permanent_water, 2=flooded.
    """
    pre = _as_bool(pre_water_mask)
    post = _as_bool(post_water_mask)
    if pre.shape != post.shape:
        raise ValueError(
            f"pre_water_mask {pre.shape} and post_water_mask {post.shape} must match. "
            "Co-register the pre/post scenes first."
        )

    out = np.full(post.shape, DRY_LAND, dtype=np.int32)
    out[post & pre] = PERMANENT_WATER
    out[post & ~pre] = FLOODED
    return out


def permanent_water_from_occurrence(
    occurrence: np.ndarray,
    threshold_pct: float = 50.0,
) -> np.ndarray:
    """Derive a permanent-water mask from a surface-water occurrence layer.

    JRC Global Surface Water ("JRC/GSW1_4/GlobalSurfaceWater", band "occurrence")
    gives, per pixel, the % of observations in which water was present (0-100).
    Pixels at or above `threshold_pct` are treated as permanent water.

    Args:
        occurrence: (H, W) array of occurrence percentages in [0, 100].
        threshold_pct: occurrence at/above which a pixel counts as permanent.

    Returns:
        (H, W) uint8 binary permanent-water mask.
    """
    occ = np.asarray(occurrence, dtype=np.float32)
    return (occ >= threshold_pct).astype(np.uint8)


def _as_bool(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    return arr > 0
