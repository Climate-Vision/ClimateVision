"""
ClimateVision Governance Module

Provides responsible AI capabilities:
- SHAP-based explainability for segmentation predictions
- Regional bias and fairness auditing
- Calibration metrics for confidence reliability
- Anomaly detection for inference inputs/outputs
- Model audit trails and version tracking
- Datasheets for training datasets (Gebru et al., 2018)
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
from .model_card import (
    ModelCard,
    build_model_card,
    generate as generate_model_card,
    render_markdown,
    write_model_card,
)
from .bias_audit import (
    run_bias_audit,
    BiasAuditor,
    BiasReport,
    RegionMetrics,
    check_fairness_gate,
    SUPPORTED_REGIONS,
)
from .calibration import (
    CalibrationReport,
    ReliabilityBin,
    brier_score,
    evaluate_calibration,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
    write_calibration_report,
)
from .datasheet import (
    Datasheet,
    build_datasheet,
    generate as generate_datasheet,
    render_markdown as render_datasheet_markdown,
    write_datasheet,
)

__all__ = [
    # Explainability
    "explain_prediction",
    "generate_shap_heatmap",
    "get_band_contributions",
    "SHAPExplainer",
    # Anomaly detection
    "AnomalyDetector",
    "AnomalyResult",
    "PredictionFeatures",
    "detect_anomaly",
    "extract_features",
    "write_anomaly_report",
    # Audit logging
    "AuditEntry",
    "AuditLogger",
    "log_prediction",
    # Model card
    "ModelCard",
    "build_model_card",
    "generate_model_card",
    "render_markdown",
    "write_model_card",
    # Bias audit
    "run_bias_audit",
    "BiasAuditor",
    "BiasReport",
    "RegionMetrics",
    "check_fairness_gate",
    "SUPPORTED_REGIONS",
    # Calibration
    "CalibrationReport",
    "ReliabilityBin",
    "brier_score",
    "evaluate_calibration",
    "expected_calibration_error",
    "maximum_calibration_error",
    "reliability_bins",
    "write_calibration_report",
    # Datasheet
    "Datasheet",
    "build_datasheet",
    "generate_datasheet",
    "render_datasheet_markdown",
    "write_datasheet",
]
