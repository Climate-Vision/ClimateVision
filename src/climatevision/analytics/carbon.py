"""
Carbon Stock Estimation Module

Converts deforestation predictions into carbon loss estimates using
allometric equations and biomass-to-carbon conversion factors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ===== Constants =====

# Carbon fraction of dry biomass (IPCC default)
CARBON_FRACTION = 0.47

# Above-ground biomass density by forest type (tonnes/hectare)
AGB_DENSITY = {
    "tropical_moist": 300.0,
    "tropical_dry": 130.0,
    "temperate_broadleaf": 180.0,
    "temperate_conifer": 200.0,
    "boreal": 90.0,
    "mangrove": 250.0,
}

# Root-to-shoot ratios for below-ground biomass estimation
ROOT_SHOOT_RATIO = {
    "tropical_moist": 0.24,
    "tropical_dry": 0.28,
    "temperate_broadleaf": 0.26,
    "temperate_conifer": 0.29,
    "boreal": 0.32,
    "mangrove": 0.49,
}

# Regional adjustment factors based on forest condition
REGIONAL_FACTORS = {
    "amazon": 1.15,
    "congo": 1.10,
    "southeast_asia": 1.05,
    "default": 1.00,
}

ForestType = Literal[
    "tropical_moist", "tropical_dry", "temperate_broadleaf",
    "temperate_conifer", "boreal", "mangrove"
]


@dataclass
class CarbonEstimate:
    """Result of carbon stock estimation."""
    hectares: float
    biomass_tonnes: float
    carbon_tonnes: float
    co2_equivalent: float
    ci_lower: float
    ci_upper: float
    forest_type: str
    region: str
    uncertainty_pct: float
    metadata: dict[str, Any] = field(default_factory=dict)


class CarbonEstimator:
    """
    Estimates carbon stock and carbon loss from deforestation masks.

    Uses allometric equations and IPCC guidelines for biomass-to-carbon
    conversion with Monte Carlo uncertainty quantification.
    """

    def __init__(
        self,
        forest_type: ForestType = "tropical_moist",
        region: str = "default",
        pixel_size_m: float = 10.0,
        uncertainty_samples: int = 1000
    ):
        self.forest_type = forest_type
        self.region = region
        self.pixel_size_m = pixel_size_m
        self.uncertainty_samples = uncertainty_samples

        self.agb_density = AGB_DENSITY.get(forest_type, 200.0)
        self.root_shoot = ROOT_SHOOT_RATIO.get(forest_type, 0.26)
        self.regional_factor = REGIONAL_FACTORS.get(region, 1.0)

    def estimate_from_mask(
        self,
        deforestation_mask: np.ndarray,
        confidence_map: Optional[np.ndarray] = None
    ) -> CarbonEstimate:
        """
        Estimate carbon loss from a deforestation mask.

        Args:
            deforestation_mask: Binary mask where 1 = deforested pixels
            confidence_map: Optional confidence scores per pixel

        Returns:
            CarbonEstimate with carbon loss and uncertainty bounds
        """
        deforested_pixels = int(deforestation_mask.sum())
        pixel_area_ha = (self.pixel_size_m ** 2) / 10000
        hectares = deforested_pixels * pixel_area_ha

        if hectares == 0:
            return CarbonEstimate(
                hectares=0, biomass_tonnes=0, carbon_tonnes=0,
                co2_equivalent=0, ci_lower=0, ci_upper=0,
                forest_type=self.forest_type, region=self.region,
                uncertainty_pct=0
            )

        # Calculate total biomass (above + below ground)
        agb = hectares * self.agb_density * self.regional_factor
        bgb = agb * self.root_shoot
        total_biomass = agb + bgb

        # Convert to carbon
        carbon_tonnes = total_biomass * CARBON_FRACTION

        # Convert to CO2 equivalent (molecular weight ratio)
        co2_equivalent = carbon_tonnes * (44 / 12)

        # Monte Carlo uncertainty estimation
        ci_lower, ci_upper, uncertainty_pct = self._estimate_uncertainty(
            hectares, confidence_map
        )

        return CarbonEstimate(
            hectares=round(hectares, 2),
            biomass_tonnes=round(total_biomass, 2),
            carbon_tonnes=round(carbon_tonnes, 2),
            co2_equivalent=round(co2_equivalent, 2),
            ci_lower=round(ci_lower, 2),
            ci_upper=round(ci_upper, 2),
            forest_type=self.forest_type,
            region=self.region,
            uncertainty_pct=round(uncertainty_pct, 1),
            metadata={
                "deforested_pixels": deforested_pixels,
                "pixel_size_m": self.pixel_size_m,
                "agb_density": self.agb_density,
                "root_shoot_ratio": self.root_shoot,
                "regional_factor": self.regional_factor,
            }
        )

    def _estimate_uncertainty(
        self,
        hectares: float,
        confidence_map: Optional[np.ndarray]
    ) -> tuple[float, float, float]:
        """
        Estimate uncertainty bounds using Monte Carlo simulation.

        Returns:
            Tuple of (ci_lower, ci_upper, uncertainty_percentage)
        """
        # Base uncertainty from IPCC (20-30% for biomass estimates)
        base_uncertainty = 0.25

        # Adjust for confidence if available
        if confidence_map is not None and confidence_map.size > 0:
            mean_confidence = float(confidence_map.mean())
            confidence_factor = 1.5 - mean_confidence  # Higher conf = lower uncertainty
        else:
            confidence_factor = 1.0

        total_uncertainty = base_uncertainty * confidence_factor

        # Generate samples
        rng = np.random.default_rng(42)
        samples = []

        for _ in range(self.uncertainty_samples):
            # Perturb parameters
            agb_sample = self.agb_density * rng.normal(1.0, 0.15)
            rs_sample = self.root_shoot * rng.normal(1.0, 0.10)
            area_sample = hectares * rng.normal(1.0, total_uncertainty)

            biomass = area_sample * agb_sample * (1 + rs_sample) * self.regional_factor
            carbon = biomass * CARBON_FRACTION
            co2 = carbon * (44 / 12)
            samples.append(co2)

        samples = np.array(samples)
        ci_lower = float(np.percentile(samples, 5))
        ci_upper = float(np.percentile(samples, 95))
        mean_val = float(samples.mean())

        uncertainty_pct = ((ci_upper - ci_lower) / (2 * mean_val)) * 100 if mean_val > 0 else 0

        return ci_lower, ci_upper, uncertainty_pct


def estimate_carbon(
    mask_path: str,
    region: str = "amazon",
    forest_type: ForestType = "tropical_moist",
    pixel_size_m: float = 10.0
) -> dict[str, Any]:
    """
    Convenience function to estimate carbon from a mask file.

    Args:
        mask_path: Path to deforestation mask (GeoTIFF)
        region: Geographic region for adjustment factors
        forest_type: Type of forest for biomass density
        pixel_size_m: Pixel size in meters

    Returns:
        Dictionary with carbon estimation results
    """
    try:
        import rasterio
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
    except ImportError:
        logger.warning("rasterio not available, loading as numpy")
        mask = np.load(mask_path.replace('.tif', '.npy'))

    estimator = CarbonEstimator(
        forest_type=forest_type,
        region=region,
        pixel_size_m=pixel_size_m
    )

    result = estimator.estimate_from_mask(mask)

    return {
        "hectares": result.hectares,
        "carbon_tonnes": result.carbon_tonnes,
        "co2_equivalent": result.co2_equivalent,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "forest_type": result.forest_type,
        "region": result.region,
        "uncertainty_pct": result.uncertainty_pct,
    }


def estimate_carbon_loss(
    deforested_pixels: int,
    pixel_size_m: float = 10.0,
    forest_type: ForestType = "tropical_moist",
    region: str = "default"
) -> dict[str, float]:
    """
    Quick carbon loss estimate from pixel count.

    Args:
        deforested_pixels: Number of deforested pixels
        pixel_size_m: Pixel size in meters
        forest_type: Type of forest
        region: Geographic region

    Returns:
        Dictionary with hectares, carbon_tonnes, and co2_equivalent
    """
    pixel_area_ha = (pixel_size_m ** 2) / 10000
    hectares = deforested_pixels * pixel_area_ha

    agb = hectares * AGB_DENSITY.get(forest_type, 200.0)
    agb *= REGIONAL_FACTORS.get(region, 1.0)
    bgb = agb * ROOT_SHOOT_RATIO.get(forest_type, 0.26)

    carbon = (agb + bgb) * CARBON_FRACTION
    co2 = carbon * (44 / 12)

    return {
        "hectares": round(hectares, 2),
        "carbon_tonnes": round(carbon, 2),
        "co2_equivalent": round(co2, 2),
    }
