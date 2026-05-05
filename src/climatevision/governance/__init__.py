"""
ClimateVision Governance Module

Provides responsible AI capabilities:
- SHAP-based explainability for segmentation predictions
- Regional bias and fairness auditing
- Anomaly detection for inference inputs/outputs
- Model audit trails and version tracking
"""

from .explainability import (
    explain_prediction,
    generate_shap_heatmap,
    get_band_contributions,
    SHAPExplainer,
)
from .anomaly_detector import (
    AnomalyDetector,
    AnomalyResult,
    PredictionFeatures,
    detect_anomaly,
    extract_features,
    write_anomaly_report,
)
from .audit_logger import (
    AuditEntry,
    AuditLogger,
    log_prediction,
)

__all__ = [
    "explain_prediction",
    "generate_shap_heatmap",
    "get_band_contributions",
    "SHAPExplainer",
    "AnomalyDetector",
    "AnomalyResult",
    "PredictionFeatures",
    "detect_anomaly",
    "extract_features",
    "write_anomaly_report",
    "AuditEntry",
    "AuditLogger",
    "log_prediction",
]
