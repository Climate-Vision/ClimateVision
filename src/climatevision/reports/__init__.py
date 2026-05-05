"""
ClimateVision Reports Module

LLM-backed natural-language reporting on top of model predictions and
analytics outputs. Stakeholder reports combine carbon analytics, SHAP
explanations, and fairness metadata into a single readable narrative.
"""

from .llm_reporter import (
    ImpactReport,
    LLMReporter,
    ReportContext,
    generate_impact_report,
    render_template,
)

__all__ = [
    "ImpactReport",
    "LLMReporter",
    "ReportContext",
    "generate_impact_report",
    "render_template",
]
