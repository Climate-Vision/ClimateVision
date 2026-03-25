"""
Ground Truth Validation Framework

Compares model predictions against reference datasets to compute
accuracy metrics and validate model performance across regions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Metrics from ground truth validation."""
    iou: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    kappa: float
    confusion_matrix: dict[str, int]
    per_class_iou: dict[str, float]
    total_pixels: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report with metrics and analysis."""
    metrics: ValidationMetrics
    region: str
    ground_truth_source: str
    prediction_source: str
    passed_threshold: bool
    threshold_used: float
    recommendations: list[str]


class GroundTruthValidator:
    """
    Validates model predictions against ground truth reference data.

    Supports validation against Global Forest Watch, forest inventory data,
    and other reference datasets.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        f1_threshold: float = 0.6,
        classes: Optional[list[str]] = None
    ):
        self.iou_threshold = iou_threshold
        self.f1_threshold = f1_threshold
        self.classes = classes or ["background", "deforested"]

    def validate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        region: str = "unknown",
        gt_source: str = "unknown"
    ) -> ValidationReport:
        """
        Validate predictions against ground truth.

        Args:
            predictions: Model prediction mask (H, W)
            ground_truth: Ground truth mask (H, W)
            region: Region name for reporting
            gt_source: Source of ground truth data

        Returns:
            ValidationReport with metrics and recommendations
        """
        if predictions.shape != ground_truth.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs "
                f"ground_truth {ground_truth.shape}"
            )

        metrics = self._compute_metrics(predictions, ground_truth)

        passed = metrics.iou >= self.iou_threshold and metrics.f1 >= self.f1_threshold

        recommendations = self._generate_recommendations(metrics)

        return ValidationReport(
            metrics=metrics,
            region=region,
            ground_truth_source=gt_source,
            prediction_source="model_inference",
            passed_threshold=passed,
            threshold_used=self.iou_threshold,
            recommendations=recommendations
        )

    def _compute_metrics(
        self,
        pred: np.ndarray,
        gt: np.ndarray
    ) -> ValidationMetrics:
        """Compute all validation metrics."""
        pred_binary = (pred > 0).astype(np.int32)
        gt_binary = (gt > 0).astype(np.int32)

        # Confusion matrix components
        tp = int(((pred_binary == 1) & (gt_binary == 1)).sum())
        tn = int(((pred_binary == 0) & (gt_binary == 0)).sum())
        fp = int(((pred_binary == 1) & (gt_binary == 0)).sum())
        fn = int(((pred_binary == 0) & (gt_binary == 1)).sum())

        total = tp + tn + fp + fn

        # Core metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0

        # IoU (Jaccard index)
        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else 0.0

        # Cohen's Kappa
        po = accuracy
        pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (total ** 2) if total > 0 else 0
        kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

        # Per-class IoU
        per_class_iou = {}
        for i, class_name in enumerate(self.classes):
            class_pred = pred_binary == i
            class_gt = gt_binary == i
            class_intersection = (class_pred & class_gt).sum()
            class_union = (class_pred | class_gt).sum()
            per_class_iou[class_name] = float(class_intersection / class_union) if class_union > 0 else 0.0

        return ValidationMetrics(
            iou=round(iou, 4),
            f1=round(f1, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            accuracy=round(accuracy, 4),
            kappa=round(kappa, 4),
            confusion_matrix={"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            per_class_iou=per_class_iou,
            total_pixels=total
        )

    def _generate_recommendations(self, metrics: ValidationMetrics) -> list[str]:
        """Generate recommendations based on validation metrics."""
        recommendations = []

        if metrics.precision < 0.6:
            recommendations.append(
                "High false positive rate detected. Consider increasing "
                "confidence threshold or adding negative samples to training."
            )

        if metrics.recall < 0.6:
            recommendations.append(
                "Low recall indicates missed detections. Review training data "
                "for underrepresented deforestation patterns."
            )

        if metrics.iou < self.iou_threshold:
            recommendations.append(
                f"IoU ({metrics.iou:.3f}) below threshold ({self.iou_threshold}). "
                "Model may need retraining or calibration for this region."
            )

        if metrics.kappa < 0.4:
            recommendations.append(
                "Low Kappa score suggests predictions are not much better than "
                "random. Review training data quality and model architecture."
            )

        if not recommendations:
            recommendations.append("Model performance meets quality thresholds.")

        return recommendations


def validate_predictions(
    pred_mask: str,
    ground_truth: str,
    iou_threshold: float = 0.5
) -> dict[str, Any]:
    """
    Convenience function to validate prediction mask against ground truth.

    Args:
        pred_mask: Path to prediction mask file
        ground_truth: Path to ground truth mask file
        iou_threshold: Minimum IoU to pass validation

    Returns:
        Dictionary with validation metrics
    """
    try:
        import rasterio
        with rasterio.open(pred_mask) as src:
            pred = src.read(1)
        with rasterio.open(ground_truth) as src:
            gt = src.read(1)
    except ImportError:
        logger.warning("rasterio not available, attempting numpy load")
        pred = np.load(pred_mask.replace('.tif', '.npy'))
        gt = np.load(ground_truth.replace('.tif', '.npy'))

    validator = GroundTruthValidator(iou_threshold=iou_threshold)
    report = validator.validate(pred, gt)

    return {
        "iou": report.metrics.iou,
        "f1": report.metrics.f1,
        "precision": report.metrics.precision,
        "recall": report.metrics.recall,
        "accuracy": report.metrics.accuracy,
        "kappa": report.metrics.kappa,
        "passed": report.passed_threshold,
        "recommendations": report.recommendations,
    }


def compare_model_versions(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    ground_truth: np.ndarray
) -> dict[str, Any]:
    """
    Compare two model versions against the same ground truth.

    Args:
        predictions_a: Predictions from model A
        predictions_b: Predictions from model B
        ground_truth: Ground truth mask

    Returns:
        Comparison metrics for both models
    """
    validator = GroundTruthValidator()

    report_a = validator.validate(predictions_a, ground_truth)
    report_b = validator.validate(predictions_b, ground_truth)

    iou_diff = report_a.metrics.iou - report_b.metrics.iou
    f1_diff = report_a.metrics.f1 - report_b.metrics.f1

    return {
        "model_a": {
            "iou": report_a.metrics.iou,
            "f1": report_a.metrics.f1,
            "precision": report_a.metrics.precision,
            "recall": report_a.metrics.recall,
        },
        "model_b": {
            "iou": report_b.metrics.iou,
            "f1": report_b.metrics.f1,
            "precision": report_b.metrics.precision,
            "recall": report_b.metrics.recall,
        },
        "comparison": {
            "iou_improvement": round(iou_diff, 4),
            "f1_improvement": round(f1_diff, 4),
            "better_model": "A" if iou_diff > 0 else "B" if iou_diff < 0 else "Equal",
        }
    }
