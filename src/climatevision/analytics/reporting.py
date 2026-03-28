"""
Impact Reporting Module

Generates environmental impact reports from model predictions,
including carbon loss estimates, trend analysis, and KPI dashboards.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegionalMetrics:
    """Environmental metrics for a specific region."""
    region: str
    period: str
    total_hectares_analyzed: float
    deforested_hectares: float
    deforestation_rate_pct: float
    carbon_tonnes_lost: float
    co2_equivalent: float
    confidence_interval: tuple[float, float]
    trend_direction: str
    yoy_change_pct: Optional[float] = None


@dataclass
class ImpactReport:
    """Complete environmental impact report."""
    report_id: str
    generated_at: datetime
    region: str
    period: str
    metrics: RegionalMetrics
    validation_summary: dict[str, Any]
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization."""
        data = asdict(self)
        data["generated_at"] = self.generated_at.isoformat()
        return data

    def to_json(self) -> str:
        """Serialize report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ReportGenerator:
    """
    Generates impact reports from analysis results.

    Combines carbon estimation, validation metrics, and trend analysis
    into stakeholder-ready reports.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_regional_report(
        self,
        region: str,
        period: str,
        carbon_result: dict[str, Any],
        validation_metrics: Optional[dict[str, Any]] = None,
        historical_data: Optional[list[dict]] = None
    ) -> ImpactReport:
        """
        Generate an impact report for a specific region and period.

        Args:
            region: Geographic region name
            period: Time period (e.g., "2023-Q2")
            carbon_result: Results from carbon estimation
            validation_metrics: Optional validation results
            historical_data: Optional historical data for trend analysis

        Returns:
            ImpactReport with all metrics and recommendations
        """
        report_id = f"{region}_{period}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        # Calculate regional metrics
        metrics = self._compute_regional_metrics(
            region, period, carbon_result, historical_data
        )

        # Summarize validation
        validation_summary = self._summarize_validation(validation_metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            metrics, validation_summary
        )

        report = ImpactReport(
            report_id=report_id,
            generated_at=datetime.utcnow(),
            region=region,
            period=period,
            metrics=metrics,
            validation_summary=validation_summary,
            recommendations=recommendations,
            metadata={
                "model_version": "1.0.0",
                "data_sources": ["sentinel-2", "landsat-8"],
            }
        )

        logger.info(f"Generated impact report: {report_id}")
        return report

    def _compute_regional_metrics(
        self,
        region: str,
        period: str,
        carbon_result: dict[str, Any],
        historical_data: Optional[list[dict]]
    ) -> RegionalMetrics:
        """Compute environmental metrics for the region."""
        hectares = carbon_result.get("hectares", 0)
        carbon_tonnes = carbon_result.get("carbon_tonnes", 0)
        co2_eq = carbon_result.get("co2_equivalent", 0)
        ci_lower = carbon_result.get("ci_lower", 0)
        ci_upper = carbon_result.get("ci_upper", 0)

        # Estimate total analyzed area (assume 10x deforested area as context)
        total_hectares = max(hectares * 10, 1000)
        deforestation_rate = (hectares / total_hectares) * 100 if total_hectares > 0 else 0

        # Trend analysis if historical data available
        trend_direction = "stable"
        yoy_change = None

        if historical_data and len(historical_data) >= 2:
            values = [h.get("carbon_tonnes", 0) for h in historical_data]
            if len(values) >= 2 and values[-2] > 0:
                yoy_change = ((values[-1] - values[-2]) / values[-2]) * 100
                if yoy_change > 5:
                    trend_direction = "increasing"
                elif yoy_change < -5:
                    trend_direction = "decreasing"

        return RegionalMetrics(
            region=region,
            period=period,
            total_hectares_analyzed=round(total_hectares, 2),
            deforested_hectares=round(hectares, 2),
            deforestation_rate_pct=round(deforestation_rate, 2),
            carbon_tonnes_lost=round(carbon_tonnes, 2),
            co2_equivalent=round(co2_eq, 2),
            confidence_interval=(round(ci_lower, 2), round(ci_upper, 2)),
            trend_direction=trend_direction,
            yoy_change_pct=round(yoy_change, 2) if yoy_change else None
        )

    def _summarize_validation(
        self,
        validation_metrics: Optional[dict[str, Any]]
    ) -> dict[str, Any]:
        """Summarize validation results for the report."""
        if not validation_metrics:
            return {"status": "not_validated", "message": "No ground truth validation performed"}

        return {
            "status": "validated",
            "iou": validation_metrics.get("iou"),
            "f1": validation_metrics.get("f1"),
            "precision": validation_metrics.get("precision"),
            "recall": validation_metrics.get("recall"),
            "passed_threshold": validation_metrics.get("passed", False),
        }

    def _generate_recommendations(
        self,
        metrics: RegionalMetrics,
        validation_summary: dict[str, Any]
    ) -> list[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []

        # High deforestation rate
        if metrics.deforestation_rate_pct > 5:
            recommendations.append(
                f"CRITICAL: Deforestation rate of {metrics.deforestation_rate_pct:.1f}% "
                "detected. Recommend immediate field investigation."
            )

        # Increasing trend
        if metrics.trend_direction == "increasing":
            recommendations.append(
                "Deforestation trend is INCREASING. Consider enhanced monitoring frequency."
            )

        # High carbon loss
        if metrics.carbon_tonnes_lost > 10000:
            recommendations.append(
                f"Significant carbon loss of {metrics.carbon_tonnes_lost:,.0f} tonnes detected. "
                "Consider alerting relevant conservation authorities."
            )

        # Validation issues
        if validation_summary.get("iou") and validation_summary["iou"] < 0.6:
            recommendations.append(
                "Model accuracy below threshold. Results should be verified with ground truth."
            )

        if not recommendations:
            recommendations.append(
                "Metrics within normal ranges. Continue standard monitoring."
            )

        return recommendations

    def save_report(self, report: ImpactReport, format: str = "json") -> Path:
        """
        Save report to file.

        Args:
            report: The report to save
            format: Output format ("json" or "html")

        Returns:
            Path to saved report file
        """
        filename = f"{report.report_id}_impact_report.{format}"
        filepath = self.output_dir / filename

        if format == "json":
            with open(filepath, "w") as f:
                f.write(report.to_json())
        elif format == "html":
            html_content = self._render_html_report(report)
            with open(filepath, "w") as f:
                f.write(html_content)

        logger.info(f"Saved report to {filepath}")
        return filepath

    def _render_html_report(self, report: ImpactReport) -> str:
        """Render report as HTML."""
        metrics = report.metrics
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Impact Report - {report.region} {report.period}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2e7d32; }}
        .metric {{ margin: 10px 0; padding: 10px; background: #f5f5f5; }}
        .critical {{ color: #d32f2f; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Environmental Impact Report</h1>
    <p><strong>Region:</strong> {report.region}</p>
    <p><strong>Period:</strong> {report.period}</p>
    <p><strong>Generated:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}</p>

    <h2>Key Metrics</h2>
    <div class="metric">
        <strong>Deforested Area:</strong> {metrics.deforested_hectares:,.2f} hectares
    </div>
    <div class="metric">
        <strong>Carbon Loss:</strong> {metrics.carbon_tonnes_lost:,.2f} tonnes CO2e
    </div>
    <div class="metric">
        <strong>Confidence Interval:</strong> {metrics.confidence_interval[0]:,.0f} - {metrics.confidence_interval[1]:,.0f} tonnes
    </div>
    <div class="metric">
        <strong>Trend:</strong> {metrics.trend_direction}
    </div>

    <h2>Recommendations</h2>
    <ul>
        {''.join(f'<li>{r}</li>' for r in report.recommendations)}
    </ul>
</body>
</html>"""


def generate_report(
    region: str,
    period: str,
    carbon_result: dict[str, Any],
    validation_metrics: Optional[dict[str, Any]] = None,
    output_dir: str = "outputs/reports/"
) -> Path:
    """
    Convenience function to generate and save an impact report.

    Args:
        region: Geographic region
        period: Time period
        carbon_result: Carbon estimation results
        validation_metrics: Optional validation metrics
        output_dir: Output directory for reports

    Returns:
        Path to saved report file
    """
    generator = ReportGenerator(output_dir)
    report = generator.generate_regional_report(
        region=region,
        period=period,
        carbon_result=carbon_result,
        validation_metrics=validation_metrics
    )
    return generator.save_report(report, format="json")
