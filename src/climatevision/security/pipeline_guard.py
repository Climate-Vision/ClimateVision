"""
Inference Pipeline Security Guard for ClimateVision.

Detects adversarial inputs and validates model outputs:
- Statistical anomaly detection on input images
- Confidence threshold enforcement
- Output distribution validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly detection check."""

    is_anomalous: bool
    anomaly_score: float
    anomaly_type: Optional[str] = None
    details: Optional[dict[str, Any]] = None
    recommendation: str = ""


@dataclass
class OutputValidation:
    """Result of model output validation."""

    is_valid: bool
    confidence: float
    issues: list[str]
    recommendation: str = ""


class InputAnomalyDetector:
    """
    Detects anomalous inputs that may indicate adversarial attacks.

    Uses statistical analysis to identify:
    - Unusual pixel distributions
    - Out-of-range values
    - Suspicious patterns (noise, uniform regions)
    """

    def __init__(
        self,
        min_pixel_value: float = -10.0,
        max_pixel_value: float = 10.0,
        min_std: float = 0.01,
        max_std: float = 5.0,
        uniform_threshold: float = 0.95,
    ):
        self.min_pixel_value = min_pixel_value
        self.max_pixel_value = max_pixel_value
        self.min_std = min_std
        self.max_std = max_std
        self.uniform_threshold = uniform_threshold

    def detect(self, image: np.ndarray) -> AnomalyResult:
        """
        Analyze image for anomalies.

        Args:
            image: Input image array (C, H, W) or (H, W, C) or (H, W)

        Returns:
            AnomalyResult with detection details
        """
        # Normalize shape
        if image.ndim == 2:
            image = image[np.newaxis, :, :]
        elif image.ndim == 3 and image.shape[2] < image.shape[0]:
            image = np.transpose(image, (2, 0, 1))

        issues = []
        anomaly_score = 0.0

        # Check 1: Value range
        min_val = float(np.min(image))
        max_val = float(np.max(image))

        if min_val < self.min_pixel_value or max_val > self.max_pixel_value:
            issues.append(f"Pixel values out of range: [{min_val:.2f}, {max_val:.2f}]")
            anomaly_score += 0.3

        # Check 2: Standard deviation (too uniform or too noisy)
        std_val = float(np.std(image))

        if std_val < self.min_std:
            issues.append(f"Image too uniform (std={std_val:.4f})")
            anomaly_score += 0.4
        elif std_val > self.max_std:
            issues.append(f"Image too noisy (std={std_val:.4f})")
            anomaly_score += 0.3

        # Check 3: NaN or Inf values
        if np.any(np.isnan(image)):
            issues.append("Image contains NaN values")
            anomaly_score += 0.5
        if np.any(np.isinf(image)):
            issues.append("Image contains Inf values")
            anomaly_score += 0.5

        # Check 4: Uniform regions (potential adversarial patch)
        for c in range(image.shape[0]):
            channel = image[c]
            unique_ratio = len(np.unique(channel)) / channel.size
            if unique_ratio < (1 - self.uniform_threshold):
                issues.append(f"Channel {c} has suspicious uniform regions")
                anomaly_score += 0.2

        # Check 5: Gradient analysis (adversarial often has unusual gradients)
        gradient_x = np.abs(np.diff(image, axis=2)).mean()
        gradient_y = np.abs(np.diff(image, axis=1)).mean()

        if gradient_x < 0.001 and gradient_y < 0.001:
            issues.append("Suspiciously low gradient (constant image)")
            anomaly_score += 0.3
        elif gradient_x > 2.0 or gradient_y > 2.0:
            issues.append("Unusually high gradient (possible noise injection)")
            anomaly_score += 0.2

        # Clamp score
        anomaly_score = min(1.0, anomaly_score)
        is_anomalous = anomaly_score >= 0.5

        details = {
            "min_value": min_val,
            "max_value": max_val,
            "std": std_val,
            "gradient_x": float(gradient_x),
            "gradient_y": float(gradient_y),
            "shape": list(image.shape),
        }

        recommendation = ""
        if is_anomalous:
            recommendation = "Input flagged as potentially adversarial. Manual review recommended."

        return AnomalyResult(
            is_anomalous=is_anomalous,
            anomaly_score=anomaly_score,
            anomaly_type="statistical_anomaly" if is_anomalous else None,
            details=details,
            recommendation=recommendation,
        )


class PipelineGuard:
    """
    Guards the inference pipeline against adversarial inputs and poisoned outputs.

    Wraps model inference to validate inputs before processing
    and outputs before returning to the client.
    """

    def __init__(
        self,
        min_confidence: float = 0.3,
        max_single_class_ratio: float = 0.99,
        enable_input_check: bool = True,
        enable_output_check: bool = True,
    ):
        self.min_confidence = min_confidence
        self.max_single_class_ratio = max_single_class_ratio
        self.enable_input_check = enable_input_check
        self.enable_output_check = enable_output_check
        self.anomaly_detector = InputAnomalyDetector()

    def check_input(self, image: np.ndarray) -> AnomalyResult:
        """Check input image for anomalies."""
        if not self.enable_input_check:
            return AnomalyResult(
                is_anomalous=False,
                anomaly_score=0.0,
                recommendation="Input checking disabled",
            )
        return self.anomaly_detector.detect(image)

    def check_output(
        self,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        n_classes: int = 2,
    ) -> OutputValidation:
        """
        Validate model output.

        Args:
            predictions: Class predictions (H, W) or (N, H, W)
            probabilities: Class probabilities (N, C, H, W) if available
            n_classes: Expected number of classes

        Returns:
            OutputValidation result
        """
        if not self.enable_output_check:
            return OutputValidation(
                is_valid=True,
                confidence=1.0,
                issues=[],
                recommendation="Output checking disabled",
            )

        issues = []
        confidence = 1.0

        # Check 1: Valid class values
        unique_classes = np.unique(predictions)
        invalid_classes = [c for c in unique_classes if c < 0 or c >= n_classes]
        if invalid_classes:
            issues.append(f"Invalid class values: {invalid_classes}")
            confidence *= 0.5

        # Check 2: Class distribution (suspicious if one class dominates)
        total_pixels = predictions.size
        for cls in range(n_classes):
            ratio = np.sum(predictions == cls) / total_pixels
            if ratio > self.max_single_class_ratio:
                issues.append(
                    f"Class {cls} dominates output ({ratio:.2%}). "
                    "May indicate model failure or adversarial input."
                )
                confidence *= 0.7

        # Check 3: Probability confidence (if available)
        if probabilities is not None:
            mean_confidence = float(np.max(probabilities, axis=1).mean())
            if mean_confidence < self.min_confidence:
                issues.append(
                    f"Low prediction confidence ({mean_confidence:.2%}). "
                    "Results may be unreliable."
                )
                confidence *= 0.8

            # Check for uniform probabilities (model confusion)
            prob_std = float(np.std(probabilities))
            if prob_std < 0.1:
                issues.append("Uniform probability distribution. Model may be confused.")
                confidence *= 0.6

        # Check 4: NaN/Inf in output
        if np.any(np.isnan(predictions)):
            issues.append("NaN values in predictions")
            confidence = 0.0
        if np.any(np.isinf(predictions)):
            issues.append("Inf values in predictions")
            confidence = 0.0

        is_valid = len(issues) == 0 or confidence >= 0.5

        recommendation = ""
        if not is_valid:
            recommendation = (
                "Output validation failed. Consider rejecting this prediction "
                "or flagging for manual review."
            )
        elif issues:
            recommendation = "Output has warnings but may still be usable."

        return OutputValidation(
            is_valid=is_valid,
            confidence=confidence,
            issues=issues,
            recommendation=recommendation,
        )

    def guard_inference(
        self,
        image: np.ndarray,
        inference_fn: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run guarded inference with input and output validation.

        Args:
            image: Input image
            inference_fn: Function to call for inference
            **kwargs: Additional arguments for inference_fn

        Returns:
            Inference result with security metadata
        """
        result = {
            "input_check": None,
            "output_check": None,
            "inference_result": None,
            "blocked": False,
            "block_reason": None,
        }

        # Input check
        input_check = self.check_input(image)
        result["input_check"] = {
            "is_anomalous": input_check.is_anomalous,
            "anomaly_score": input_check.anomaly_score,
            "anomaly_type": input_check.anomaly_type,
            "details": input_check.details,
        }

        if input_check.is_anomalous:
            logger.warning(
                "Anomalous input detected (score=%.2f): %s",
                input_check.anomaly_score,
                input_check.anomaly_type,
            )
            result["blocked"] = True
            result["block_reason"] = input_check.recommendation
            return result

        # Run inference
        try:
            inference_result = inference_fn(image, **kwargs)
            result["inference_result"] = inference_result
        except Exception as e:
            logger.error("Inference failed: %s", e)
            result["blocked"] = True
            result["block_reason"] = f"Inference error: {str(e)}"
            return result

        # Output check (if we have predictions)
        if "predictions" in inference_result:
            predictions = inference_result["predictions"]
            probabilities = inference_result.get("probabilities")
            n_classes = inference_result.get("n_classes", 2)

            output_check = self.check_output(predictions, probabilities, n_classes)
            result["output_check"] = {
                "is_valid": output_check.is_valid,
                "confidence": output_check.confidence,
                "issues": output_check.issues,
            }

            if not output_check.is_valid:
                logger.warning(
                    "Output validation failed: %s",
                    output_check.issues,
                )

        return result


def detect_adversarial_input(image: np.ndarray) -> AnomalyResult:
    """
    Convenience function to detect adversarial inputs.

    Args:
        image: Input image array

    Returns:
        AnomalyResult
    """
    detector = InputAnomalyDetector()
    return detector.detect(image)


def validate_model_output(
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    n_classes: int = 2,
) -> OutputValidation:
    """
    Convenience function to validate model output.

    Args:
        predictions: Class predictions
        probabilities: Class probabilities (optional)
        n_classes: Expected number of classes

    Returns:
        OutputValidation result
    """
    guard = PipelineGuard()
    return guard.check_output(predictions, probabilities, n_classes)
