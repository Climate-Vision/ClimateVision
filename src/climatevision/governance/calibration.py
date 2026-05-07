"""
Calibration metrics for ClimateVision segmentation models.

A model that reports a confidence of 0.9 should be correct about 90% of the
time — anything else is miscalibration. For NGO-facing alerts driven by
threshold logic on confidence, miscalibration directly mistranslates into
either missed events or false alarms, so the calibration of every released
model needs to be measured alongside the headline accuracy.

This module computes the standard reliability-diagram metrics for binary
segmentation outputs:

- Reliability bins: bucket pixel predictions by confidence, record the
  observed positive-rate in each bucket against the bucket's mean confidence.
- Expected Calibration Error (ECE): support-weighted mean of the absolute gap
  between confidence and accuracy across bins.
- Maximum Calibration Error (MCE): the worst single-bin gap.
- Brier score: mean squared error between probability and binary target.

All metrics operate on flat numpy arrays so they slot into the existing
governance pipeline (model card generator, release CI gate) without
introducing a torch dependency at evaluation time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_N_BINS = 15


@dataclass
class ReliabilityBin:
    """One bucket of the reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    observed_positive_rate: float


@dataclass
class CalibrationReport:
    """Calibration evaluation summary for a single model run."""

    model_version: str
    n_samples: int
    n_bins: int
    ece: float
    mce: float
    brier_score: float
    bins: List[ReliabilityBin] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bins"] = [asdict(b) for b in self.bins]
        return d

    def is_well_calibrated(self, ece_threshold: float = 0.05) -> bool:
        """Default release-gate threshold: ECE under 5%."""
        return self.ece <= ece_threshold


def _validate_inputs(probabilities: np.ndarray, targets: np.ndarray) -> None:
    if probabilities.shape != targets.shape:
        raise ValueError(
            f"probabilities and targets must have the same shape, got "
            f"{probabilities.shape} and {targets.shape}"
        )
    if probabilities.size == 0:
        raise ValueError("probabilities array is empty")
    if probabilities.min() < 0.0 or probabilities.max() > 1.0:
        raise ValueError("probabilities must lie in [0, 1]")
    unique_targets = np.unique(targets)
    if not np.all(np.isin(unique_targets, [0, 1])):
        raise ValueError(
            f"targets must be binary {{0, 1}}, got values {unique_targets}"
        )


def reliability_bins(
    probabilities: np.ndarray,
    targets: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> List[ReliabilityBin]:
    """Bucket predictions by confidence and return per-bin reliability."""
    probs = np.asarray(probabilities, dtype=np.float64).ravel()
    tgts = np.asarray(targets, dtype=np.int32).ravel()
    _validate_inputs(probs, tgts)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: List[ReliabilityBin] = []
    for i in range(n_bins):
        lower, upper = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                ReliabilityBin(
                    lower=float(lower),
                    upper=float(upper),
                    count=0,
                    mean_confidence=0.0,
                    observed_positive_rate=0.0,
                )
            )
            continue
        bins.append(
            ReliabilityBin(
                lower=float(lower),
                upper=float(upper),
                count=count,
                mean_confidence=float(probs[mask].mean()),
                observed_positive_rate=float(tgts[mask].mean()),
            )
        )
    return bins


def expected_calibration_error(bins: List[ReliabilityBin]) -> float:
    """Support-weighted mean gap between confidence and observed accuracy."""
    total = sum(b.count for b in bins)
    if total == 0:
        return 0.0
    weighted = sum(
        (b.count / total) * abs(b.mean_confidence - b.observed_positive_rate)
        for b in bins
        if b.count > 0
    )
    return float(weighted)


def maximum_calibration_error(bins: List[ReliabilityBin]) -> float:
    """Worst single-bin gap between confidence and observed accuracy."""
    populated = [b for b in bins if b.count > 0]
    if not populated:
        return 0.0
    return float(
        max(abs(b.mean_confidence - b.observed_positive_rate) for b in populated)
    )


def brier_score(
    probabilities: np.ndarray, targets: np.ndarray
) -> float:
    """Mean squared error between probability and binary target."""
    probs = np.asarray(probabilities, dtype=np.float64).ravel()
    tgts = np.asarray(targets, dtype=np.float64).ravel()
    _validate_inputs(probs, tgts.astype(np.int32))
    return float(np.mean((probs - tgts) ** 2))


def evaluate_calibration(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    model_version: str,
    n_bins: int = DEFAULT_N_BINS,
) -> CalibrationReport:
    """Run the full calibration evaluation and return a report dataclass."""
    probs = np.asarray(probabilities, dtype=np.float64).ravel()
    tgts = np.asarray(targets, dtype=np.int32).ravel()
    _validate_inputs(probs, tgts)
    bins = reliability_bins(probs, tgts, n_bins=n_bins)
    return CalibrationReport(
        model_version=model_version,
        n_samples=int(probs.size),
        n_bins=n_bins,
        ece=expected_calibration_error(bins),
        mce=maximum_calibration_error(bins),
        brier_score=brier_score(probs, tgts),
        bins=bins,
    )


def write_calibration_report(
    report: CalibrationReport, path: Union[str, Path]
) -> Path:
    """Persist a CalibrationReport to disk as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2))
    logger.info("Wrote calibration report to %s", out)
    return out
