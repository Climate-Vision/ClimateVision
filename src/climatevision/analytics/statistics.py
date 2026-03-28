"""
Statistical Testing and Analysis Module

Provides hypothesis testing, trend analysis, and A/B testing
capabilities for model comparison and environmental data analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HypothesisTestResult:
    """Result from a statistical hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    reject_null: bool
    confidence_level: float
    effect_size: Optional[float]
    interpretation: str


@dataclass
class TrendAnalysisResult:
    """Result from time series trend analysis."""
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    trend_direction: Literal["increasing", "decreasing", "stable"]
    percent_change: float
    confidence_interval: tuple[float, float]


@dataclass
class ABTestResult:
    """Result from A/B model comparison test."""
    model_a_mean: float
    model_b_mean: float
    difference: float
    p_value: float
    significant: bool
    better_model: Literal["A", "B", "no_difference"]
    confidence_interval: tuple[float, float]
    sample_size: int


def t_test_two_sample(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    alpha: float = 0.05
) -> HypothesisTestResult:
    """
    Perform two-sample t-test for comparing model predictions.

    Args:
        sample_a: First sample array
        sample_b: Second sample array
        alpha: Significance level

    Returns:
        HypothesisTestResult with test statistics
    """
    n_a, n_b = len(sample_a), len(sample_b)
    mean_a, mean_b = sample_a.mean(), sample_b.mean()
    var_a, var_b = sample_a.var(ddof=1), sample_b.var(ddof=1)

    # Pooled standard error
    se = np.sqrt(var_a / n_a + var_b / n_b)
    t_stat = (mean_a - mean_b) / se if se > 0 else 0

    # Degrees of freedom (Welch's approximation)
    if var_a > 0 and var_b > 0:
        df = ((var_a / n_a + var_b / n_b) ** 2 /
              ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)))
    else:
        df = n_a + n_b - 2

    # Approximate p-value using normal distribution for large samples
    p_value = 2 * (1 - _norm_cdf(abs(t_stat)))

    # Cohen's d effect size
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    effect_size = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

    reject_null = p_value < alpha

    interpretation = (
        f"Significant difference detected (p={p_value:.4f})" if reject_null
        else f"No significant difference (p={p_value:.4f})"
    )

    return HypothesisTestResult(
        test_name="Two-Sample T-Test (Welch's)",
        statistic=round(t_stat, 4),
        p_value=round(p_value, 4),
        reject_null=reject_null,
        confidence_level=1 - alpha,
        effect_size=round(effect_size, 4),
        interpretation=interpretation
    )


def mann_whitney_u(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    alpha: float = 0.05
) -> HypothesisTestResult:
    """
    Non-parametric Mann-Whitney U test for comparing distributions.

    Useful when data may not be normally distributed.
    """
    n_a, n_b = len(sample_a), len(sample_b)

    # Combine and rank
    combined = np.concatenate([sample_a, sample_b])
    ranks = np.argsort(np.argsort(combined)) + 1

    # Sum of ranks for sample A
    r_a = ranks[:n_a].sum()

    # U statistic
    u_a = r_a - n_a * (n_a + 1) / 2
    u_b = n_a * n_b - u_a
    u_stat = min(u_a, u_b)

    # Normal approximation for large samples
    mu = n_a * n_b / 2
    sigma = np.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)
    z = (u_stat - mu) / sigma if sigma > 0 else 0

    p_value = 2 * (1 - _norm_cdf(abs(z)))
    reject_null = p_value < alpha

    # Effect size (rank-biserial correlation)
    effect_size = 1 - (2 * u_stat) / (n_a * n_b)

    return HypothesisTestResult(
        test_name="Mann-Whitney U Test",
        statistic=round(u_stat, 4),
        p_value=round(p_value, 4),
        reject_null=reject_null,
        confidence_level=1 - alpha,
        effect_size=round(effect_size, 4),
        interpretation=f"{'Significant' if reject_null else 'No significant'} difference in distributions"
    )


def linear_trend_analysis(
    values: np.ndarray,
    time_points: Optional[np.ndarray] = None
) -> TrendAnalysisResult:
    """
    Analyze linear trend in time series data.

    Args:
        values: Array of values over time
        time_points: Optional time indices (default: 0 to n-1)

    Returns:
        TrendAnalysisResult with slope, significance, and direction
    """
    n = len(values)
    if time_points is None:
        time_points = np.arange(n)

    # Linear regression via least squares
    x_mean = time_points.mean()
    y_mean = values.mean()

    ss_xy = ((time_points - x_mean) * (values - y_mean)).sum()
    ss_xx = ((time_points - x_mean) ** 2).sum()

    slope = ss_xy / ss_xx if ss_xx > 0 else 0
    intercept = y_mean - slope * x_mean

    # Predictions and residuals
    predictions = slope * time_points + intercept
    ss_res = ((values - predictions) ** 2).sum()
    ss_tot = ((values - y_mean) ** 2).sum()

    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Standard error and t-statistic
    if n > 2 and ss_xx > 0:
        se_slope = np.sqrt(ss_res / (n - 2) / ss_xx)
        t_stat = slope / se_slope if se_slope > 0 else 0
        p_value = 2 * (1 - _norm_cdf(abs(t_stat)))
    else:
        p_value = 1.0

    # Trend direction
    if p_value < 0.05:
        trend_direction = "increasing" if slope > 0 else "decreasing"
    else:
        trend_direction = "stable"

    # Percent change over the series
    start_val = slope * time_points[0] + intercept
    end_val = slope * time_points[-1] + intercept
    percent_change = ((end_val - start_val) / abs(start_val) * 100) if start_val != 0 else 0

    # 95% CI for slope
    if n > 2 and ss_xx > 0:
        ci_margin = 1.96 * se_slope
        ci = (slope - ci_margin, slope + ci_margin)
    else:
        ci = (slope, slope)

    return TrendAnalysisResult(
        slope=round(slope, 6),
        intercept=round(intercept, 4),
        r_squared=round(r_squared, 4),
        p_value=round(p_value, 4),
        trend_direction=trend_direction,
        percent_change=round(percent_change, 2),
        confidence_interval=(round(ci[0], 6), round(ci[1], 6))
    )


def ab_test_models(
    metrics_a: np.ndarray,
    metrics_b: np.ndarray,
    metric_name: str = "IoU",
    alpha: float = 0.05
) -> ABTestResult:
    """
    A/B test comparing two model versions on the same dataset.

    Args:
        metrics_a: Metrics from model A (e.g., IoU scores per image)
        metrics_b: Metrics from model B
        metric_name: Name of metric for reporting
        alpha: Significance level

    Returns:
        ABTestResult with comparison statistics
    """
    mean_a = float(metrics_a.mean())
    mean_b = float(metrics_b.mean())
    diff = mean_a - mean_b

    # Paired t-test for matched samples
    if len(metrics_a) == len(metrics_b):
        differences = metrics_a - metrics_b
        n = len(differences)
        d_mean = differences.mean()
        d_std = differences.std(ddof=1)

        se = d_std / np.sqrt(n) if n > 0 else 0
        t_stat = d_mean / se if se > 0 else 0
        p_value = 2 * (1 - _norm_cdf(abs(t_stat)))

        ci_margin = 1.96 * se
        ci = (d_mean - ci_margin, d_mean + ci_margin)
    else:
        # Unpaired test
        result = t_test_two_sample(metrics_a, metrics_b, alpha)
        p_value = result.p_value
        se = abs(diff) / abs(result.statistic) if result.statistic != 0 else 0
        ci = (diff - 1.96 * se, diff + 1.96 * se)

    significant = p_value < alpha

    if not significant:
        better = "no_difference"
    else:
        better = "A" if diff > 0 else "B"

    return ABTestResult(
        model_a_mean=round(mean_a, 4),
        model_b_mean=round(mean_b, 4),
        difference=round(diff, 4),
        p_value=round(p_value, 4),
        significant=significant,
        better_model=better,
        confidence_interval=(round(ci[0], 4), round(ci[1], 4)),
        sample_size=len(metrics_a)
    )


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func: callable = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for any statistic.

    Args:
        data: Input data array
        statistic_func: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(42)
    n = len(data)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    point_estimate = float(statistic_func(data))

    return point_estimate, ci_lower, ci_upper


def _norm_cdf(x: float) -> float:
    """Approximation of standard normal CDF."""
    return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
