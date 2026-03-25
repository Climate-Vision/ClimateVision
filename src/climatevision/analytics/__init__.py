"""
ClimateVision Analytics Module

Provides carbon estimation, statistical analysis, validation, and reporting
for translating model predictions into environmental impact metrics.
"""

from climatevision.analytics.carbon import (
    CarbonEstimator,
    estimate_carbon,
    estimate_carbon_loss,
)

__all__ = [
    "CarbonEstimator",
    "estimate_carbon",
    "estimate_carbon_loss",
]
