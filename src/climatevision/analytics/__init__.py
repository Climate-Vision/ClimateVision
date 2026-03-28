"""
ClimateVision Analytics Module

Provides carbon estimation, statistical analysis, validation, and reporting
for translating model predictions into environmental impact metrics.
"""

from climatevision.analytics.carbon import (
    CarbonEstimator,
    CarbonEstimate,
    estimate_carbon,
    estimate_carbon_loss,
)

from climatevision.analytics.validation import (
    GroundTruthValidator,
    ValidationMetrics,
    ValidationReport,
    validate_predictions,
    compare_model_versions,
)

from climatevision.analytics.statistics import (
    t_test_two_sample,
    mann_whitney_u,
    linear_trend_analysis,
    ab_test_models,
    bootstrap_confidence_interval,
    HypothesisTestResult,
    TrendAnalysisResult,
    ABTestResult,
)

from climatevision.analytics.reporting import (
    ReportGenerator,
    ImpactReport,
    RegionalMetrics,
    generate_report,
)

__all__ = [
    # Carbon estimation
    "CarbonEstimator",
    "CarbonEstimate",
    "estimate_carbon",
    "estimate_carbon_loss",
    # Validation
    "GroundTruthValidator",
    "ValidationMetrics",
    "ValidationReport",
    "validate_predictions",
    "compare_model_versions",
    # Statistics
    "t_test_two_sample",
    "mann_whitney_u",
    "linear_trend_analysis",
    "ab_test_models",
    "bootstrap_confidence_interval",
    "HypothesisTestResult",
    "TrendAnalysisResult",
    "ABTestResult",
    # Reporting
    "ReportGenerator",
    "ImpactReport",
    "RegionalMetrics",
    "generate_report",
]
