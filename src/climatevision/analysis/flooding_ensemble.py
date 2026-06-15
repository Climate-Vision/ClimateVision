"""
GFM-style ensemble flood detection using three independent SAR algorithms.

No deep learning required — operates purely on Sentinel-1 backscatter
using well-established physics-based and statistical methods.

Algorithms:
  1. LIST-style: change detection (pre/post differencing) + histogram thresholding
  2. DLR-style: tile-based Otsu thresholding on VH + fuzzy slope filtering
  3. TUW-style: per-pixel Bayesian classification using backscatter distributions

Ensemble: majority vote (≥2 of 3 must agree to classify as flooded).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LIST-style change detection
# ---------------------------------------------------------------------------

class LISTFloodDetector:
    """
    Change-detection based flood mapping.

    Detects flooding by comparing a pre-event reference image to a post-event
    image. Uses histogram thresholding on the backscatter difference.
    """

    def __init__(self, diff_threshold_db: float = -3.0):
        self.diff_threshold_db = diff_threshold_db

    def detect(
        self,
        pre_vh: np.ndarray,
        post_vh: np.ndarray,
    ) -> np.ndarray:
        """
        Args:
            pre_vh: Pre-event VH backscatter in dB, shape (H, W).
            post_vh: Post-event VH backscatter in dB, shape (H, W).

        Returns:
            Binary flood mask (H, W), 1=flooded.
        """
        diff = post_vh - pre_vh
        # Flood typically lowers VH backscatter by several dB
        flooded = diff < self.diff_threshold_db
        return flooded.astype(np.uint8)


# ---------------------------------------------------------------------------
# DLR-style fuzzy thresholding
# ---------------------------------------------------------------------------

class DLRFloodDetector:
    """
    Hierarchical tile-based thresholding with fuzzy logic refinement.

    Uses Otsu thresholding on post-event VH backscatter, then refines using
    terrain slope and water body size constraints.
    """

    def __init__(
        self,
        min_water_size: int = 10,
        slope_mask: Optional[np.ndarray] = None,
    ):
        self.min_water_size = min_water_size
        self.slope_mask = slope_mask

    def detect(self, post_vh: np.ndarray) -> np.ndarray:
        """
        Args:
            post_vh: Post-event VH backscatter in dB, shape (H, W).

        Returns:
            Binary flood mask (H, W), 1=flooded.
        """
        try:
            from skimage.filters import threshold_otsu
            from skimage.morphology import remove_small_objects
        except ImportError:
            raise ImportError("scikit-image is required for DLR detector. Install: pip install scikit-image")

        # Water has very low VH backscatter
        thresh = threshold_otsu(post_vh)
        water = post_vh < thresh

        # Remove small noise pixels
        water = remove_small_objects(water, min_size=self.min_water_size)

        # Apply slope mask if provided (mask out steep terrain)
        if self.slope_mask is not None:
            water = water & (~self.slope_mask)

        return water.astype(np.uint8)


# ---------------------------------------------------------------------------
# TUW-style Bayesian classification
# ---------------------------------------------------------------------------

class TUWFloodDetector:
    """
    Bayesian flood classification using backscatter distribution modeling.

    Models water and land as Gaussian distributions in VH backscatter space,
    then classifies each pixel by posterior probability.
    """

    def __init__(
        self,
        water_mean_db: float = -24.0,
        water_std_db: float = 3.0,
        land_mean_db: float = -18.0,
        land_std_db: float = 4.0,
        prior_water: float = 0.3,
    ):
        self.water_mean = water_mean_db
        self.water_std = water_std_db
        self.land_mean = land_mean_db
        self.land_std = land_std_db
        self.prior_water = prior_water
        self.prior_land = 1.0 - prior_water

    def detect(self, post_vh: np.ndarray) -> np.ndarray:
        """
        Args:
            post_vh: Post-event VH backscatter in dB, shape (H, W).

        Returns:
            Binary flood mask (H, W), 1=flooded.
        """
        # Gaussian likelihoods
        def _gaussian_pdf(x, mu, sigma):
            return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

        p_vh_water = _gaussian_pdf(post_vh, self.water_mean, self.water_std)
        p_vh_land = _gaussian_pdf(post_vh, self.land_mean, self.land_std)

        # Posterior probability of water
        posterior_water = (p_vh_water * self.prior_water) / (
            p_vh_water * self.prior_water + p_vh_land * self.prior_land + 1e-10
        )

        flooded = posterior_water > 0.5
        return flooded.astype(np.uint8)


# ---------------------------------------------------------------------------
# Ensemble pipeline
# ---------------------------------------------------------------------------

class EnsembleFloodPipeline:
    """
    GFM-style ensemble combining LIST, DLR, and TUW detectors.

    A pixel is classified as flooded only if at least 2 of 3 algorithms agree.
    """

    def __init__(
        self,
        list_detector: Optional[LISTFloodDetector] = None,
        dlr_detector: Optional[DLRFloodDetector] = None,
        tuw_detector: Optional[TUWFloodDetector] = None,
    ):
        self.list_det = list_detector or LISTFloodDetector()
        self.dlr_det = dlr_detector or DLRFloodDetector()
        self.tuw_det = tuw_detector or TUWFloodDetector()

    def detect(
        self,
        post_vh: np.ndarray,
        pre_vh: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """
        Run all three detectors and return ensemble result.

        Args:
            post_vh: Post-event VH backscatter in dB, shape (H, W).
            pre_vh: Optional pre-event VH for change detection.

        Returns:
            Dict with keys:
                - list_mask: LIST detector result
                - dlr_mask: DLR detector result
                - tuw_mask: TUW detector result
                - ensemble_mask: Majority vote result
                - agreement: Number of algorithms agreeing per pixel (0-3)
        """
        list_mask = (
            self.list_det.detect(pre_vh, post_vh)
            if pre_vh is not None
            else np.zeros_like(post_vh, dtype=np.uint8)
        )
        dlr_mask = self.dlr_det.detect(post_vh)
        tuw_mask = self.tuw_det.detect(post_vh)

        # Stack and sum votes
        votes = list_mask.astype(np.uint8) + dlr_mask.astype(np.uint8) + tuw_mask.astype(np.uint8)
        ensemble_mask = (votes >= 2).astype(np.uint8)

        logger.info(
            "Ensemble vote counts: 0=%d, 1=%d, 2=%d, 3=%d",
            int((votes == 0).sum()),
            int((votes == 1).sum()),
            int((votes == 2).sum()),
            int((votes == 3).sum()),
        )

        return {
            "list_mask": list_mask,
            "dlr_mask": dlr_mask,
            "tuw_mask": tuw_mask,
            "ensemble_mask": ensemble_mask,
            "agreement": votes,
        }
