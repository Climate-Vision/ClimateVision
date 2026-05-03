"""
Anomaly detection for ClimateVision inference inputs and outputs.

Flags predictions whose confidence distributions or input statistics fall
outside historical norms, so they can be routed for human review before
reaching downstream stakeholders.

The detector combines two complementary strategies:

1. Isolation Forest over a vector of summary features extracted from each
   prediction (mean confidence, std, positive-pixel fraction, entropy).
2. Statistical bounds (z-score, IQR) computed from a rolling history of
   recent predictions, useful when not enough data exists to fit IF.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_HISTORY_PATH = _PROJECT_ROOT / "outputs" / "anomalies" / "history.jsonl"
_REPORT_DIR = _PROJECT_ROOT / "outputs" / "anomalies"

_FEATURE_NAMES = (
    "mean_confidence",
    "std_confidence",
    "positive_fraction",
    "entropy",
)


@dataclass
class PredictionFeatures:
    """Summary statistics extracted from a single prediction mask."""

    mean_confidence: float
    std_confidence: float
    positive_fraction: float
    entropy: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.mean_confidence,
                self.std_confidence,
                self.positive_fraction,
                self.entropy,
            ],
            dtype=np.float64,
        )


@dataclass
class AnomalyResult:
    is_anomaly: bool
    score: float
    method: str
    reasons: list[str]
    features: PredictionFeatures

    def to_dict(self) -> dict:
        d = asdict(self)
        d["features"] = asdict(self.features)
        return d


def extract_features(
    confidence: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> PredictionFeatures:
    """
    Compute summary statistics for an inference output.

    Args:
        confidence: Per-pixel confidence scores in [0, 1].
        mask: Optional binary mask. If omitted, derived from `confidence > threshold`.
        threshold: Decision threshold used when mask is omitted.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    if confidence.size == 0:
        raise ValueError("confidence array is empty")

    if mask is None:
        mask = confidence > threshold
    mask = np.asarray(mask).astype(bool)

    eps = 1e-9
    p = np.clip(confidence, eps, 1 - eps)
    pixel_entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))

    return PredictionFeatures(
        mean_confidence=float(confidence.mean()),
        std_confidence=float(confidence.std()),
        positive_fraction=float(mask.mean()),
        entropy=float(pixel_entropy.mean()),
    )


class AnomalyDetector:
    """
    Hybrid anomaly detector for inference outputs.

    Holds a rolling history of prediction features and exposes two
    detection paths: a fitted Isolation Forest (when enough history is
    available) and a statistical fallback based on per-feature z-scores
    and IQR fences.
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        min_history_for_iforest: int = 50,
        contamination: float = 0.05,
        history_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.min_history_for_iforest = min_history_for_iforest
        self.contamination = contamination
        self.history_path = Path(history_path) if history_path else _HISTORY_PATH
        self._history: list[PredictionFeatures] = []
        self._iforest = None

    def load_history(self) -> None:
        if not self.history_path.exists():
            return
        with self.history_path.open() as fh:
            for line in fh:
                row = json.loads(line)
                self._history.append(PredictionFeatures(**row))
        logger.info("Loaded %d historical predictions", len(self._history))

    def _persist(self, features: PredictionFeatures) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a") as fh:
            fh.write(json.dumps(asdict(features)) + "\n")

    def _fit_iforest(self) -> None:
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.warning("scikit-learn not available, falling back to statistical checks")
            self._iforest = None
            return

        X = np.stack([f.as_vector() for f in self._history])
        self._iforest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
        ).fit(X)
        logger.info("Fitted IsolationForest on %d samples", len(self._history))

    def _statistical_check(
        self, features: PredictionFeatures
    ) -> tuple[bool, float, list[str]]:
        if len(self._history) < 5:
            return False, 0.0, ["insufficient_history"]

        X = np.stack([f.as_vector() for f in self._history])
        x = features.as_vector()

        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-9
        z = np.abs((x - mean) / std)

        q1, q3 = np.percentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr

        reasons: list[str] = []
        for i, name in enumerate(_FEATURE_NAMES):
            if z[i] > self.z_threshold:
                reasons.append(f"{name}_z={z[i]:.2f}")
            if x[i] < lower[i] or x[i] > upper[i]:
                reasons.append(f"{name}_outside_iqr")

        return bool(reasons), float(z.max()), reasons

    def detect(
        self,
        confidence: np.ndarray,
        mask: Optional[np.ndarray] = None,
        record: bool = True,
    ) -> AnomalyResult:
        features = extract_features(confidence, mask=mask)

        if (
            self._iforest is None
            and len(self._history) >= self.min_history_for_iforest
        ):
            self._fit_iforest()

        if self._iforest is not None:
            score = float(self._iforest.score_samples(features.as_vector().reshape(1, -1))[0])
            is_anomaly = bool(self._iforest.predict(features.as_vector().reshape(1, -1))[0] == -1)
            method = "isolation_forest"
            reasons = ["isolation_forest_outlier"] if is_anomaly else []
        else:
            is_anomaly, score, reasons = self._statistical_check(features)
            method = "statistical"

        if record:
            self._history.append(features)
            self._persist(features)

        result = AnomalyResult(
            is_anomaly=is_anomaly,
            score=score,
            method=method,
            reasons=reasons,
            features=features,
        )

        if is_anomaly:
            logger.warning(
                "Anomaly detected (method=%s, score=%.3f, reasons=%s)",
                method,
                score,
                reasons,
            )
        return result


def detect_anomaly(
    confidence: np.ndarray,
    mask: Optional[np.ndarray] = None,
    detector: Optional[AnomalyDetector] = None,
) -> AnomalyResult:
    """Convenience wrapper that lazily constructs a detector and loads history."""
    if detector is None:
        detector = AnomalyDetector()
        detector.load_history()
    return detector.detect(confidence, mask=mask)


def write_anomaly_report(
    results: Iterable[AnomalyResult],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Persist a batch of anomaly results to a JSON report for review."""
    output_path = Path(output_path) if output_path else _REPORT_DIR / "anomaly_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.to_dict() for r in results]
    with output_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote anomaly report with %d entries to %s", len(payload), output_path)
    return output_path
