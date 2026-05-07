"""
Distributional drift detection for ClimateVision inputs and predictions.

The existing anomaly detector (``governance.anomaly_detector``) flags
*individual* predictions whose features fall outside historical norms.
This module is its complement: it compares the *distribution* of recent
predictions (or inputs) against a reference baseline and flags drift
even when no single prediction is anomalous.

Two well-understood non-parametric tests are exposed:

- **Population Stability Index (PSI)** — bins both windows on the
  reference's quantiles and sums (p_i - q_i) * log(p_i / q_i). The
  industry-standard rule of thumb: PSI < 0.1 stable, 0.1-0.25 moderate
  drift, > 0.25 significant drift.
- **Kolmogorov-Smirnov (KS)** — supremum of the gap between the two
  empirical CDFs, with a two-sample asymptotic p-value.

Both run on a single feature at a time. Multi-feature drift is reported
as a list of per-feature ``DriftResult`` objects so callers can decide
how to aggregate (any-feature-drifts vs. average) without baking that
policy into the detector.

Designed to plug into the prediction-history JSONL written by the
anomaly detector, so drift checks can run as a CI step over the last
N days of production predictions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PSI_BINS = 10
PSI_STABLE = 0.10
PSI_MODERATE = 0.25
DEFAULT_KS_SIGNIFICANCE = 0.05


@dataclass
class DriftResult:
    """Per-feature drift assessment."""

    feature: str
    method: str
    statistic: float
    threshold: float
    drifted: bool
    severity: str  # "stable", "moderate", "severe"
    p_value: Optional[float] = None
    n_reference: int = 0
    n_current: int = 0


@dataclass
class DriftReport:
    """Multi-feature drift report covering one window comparison."""

    reference_window: str
    current_window: str
    method: str
    results: List[DriftResult] = field(default_factory=list)

    @property
    def any_drifted(self) -> bool:
        return any(r.drifted for r in self.results)

    @property
    def severe_features(self) -> List[str]:
        return [r.feature for r in self.results if r.severity == "severe"]

    def to_dict(self) -> dict:
        return {
            "reference_window": self.reference_window,
            "current_window": self.current_window,
            "method": self.method,
            "any_drifted": self.any_drifted,
            "severe_features": self.severe_features,
            "results": [asdict(r) for r in self.results],
        }


def _as_array(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} window is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} window contains non-finite values")
    return arr


def population_stability_index(
    reference: Sequence[float],
    current: Sequence[float],
    n_bins: int = DEFAULT_PSI_BINS,
) -> float:
    """Compute PSI between a reference and a current sample.

    Bins are derived from quantiles of the reference distribution so the
    reference always has roughly equal mass per bin, which is the canonical
    PSI definition. Empty bins are floored to a small epsilon to keep the
    log finite.
    """
    ref = _as_array(reference, "reference")
    cur = _as_array(current, "current")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 2:
        # Reference is a constant — fall back to a single bin.
        edges = np.array([ref.min() - 1e-6, ref.max() + 1e-6])

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _psi_severity(psi: float) -> str:
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_MODERATE:
        return "moderate"
    return "severe"


def kolmogorov_smirnov(
    reference: Sequence[float],
    current: Sequence[float],
) -> tuple[float, float]:
    """Two-sample KS statistic + asymptotic p-value.

    Avoids importing scipy by computing the supremum gap of the empirical
    CDFs directly and using the standard Kolmogorov asymptotic series for
    the p-value.
    """
    ref = _as_array(reference, "reference")
    cur = _as_array(current, "current")

    combined = np.sort(np.concatenate([ref, cur]))
    cdf_ref = np.searchsorted(np.sort(ref), combined, side="right") / ref.size
    cdf_cur = np.searchsorted(np.sort(cur), combined, side="right") / cur.size
    statistic = float(np.max(np.abs(cdf_ref - cdf_cur)))

    n = ref.size
    m = cur.size
    en = float(np.sqrt(n * m / (n + m)))
    lam = (en + 0.12 + 0.11 / en) * statistic

    # Kolmogorov asymptotic distribution: P(K > lam) summed series.
    p = 0.0
    for k in range(1, 101):
        term = 2 * (-1) ** (k - 1) * np.exp(-2 * (lam ** 2) * (k ** 2))
        p += term
        if abs(term) < 1e-12:
            break
    p_value = float(min(max(p, 0.0), 1.0))
    return statistic, p_value


def detect_drift(
    reference: dict[str, Sequence[float]],
    current: dict[str, Sequence[float]],
    *,
    method: str = "psi",
    reference_window: str = "baseline",
    current_window: str = "current",
    psi_bins: int = DEFAULT_PSI_BINS,
    ks_significance: float = DEFAULT_KS_SIGNIFICANCE,
) -> DriftReport:
    """Per-feature drift assessment over two windows.

    ``reference`` and ``current`` are dicts mapping feature name to a 1D
    sample of values from the respective window. Features must match.
    """
    if method not in {"psi", "ks"}:
        raise ValueError(f"unknown method: {method!r}; expected 'psi' or 'ks'")

    missing = set(reference.keys()) ^ set(current.keys())
    if missing:
        raise ValueError(f"feature mismatch between windows: {missing}")

    results: List[DriftResult] = []
    for feature in reference.keys():
        ref_values = reference[feature]
        cur_values = current[feature]
        if method == "psi":
            psi = population_stability_index(ref_values, cur_values, n_bins=psi_bins)
            severity = _psi_severity(psi)
            results.append(
                DriftResult(
                    feature=feature,
                    method="psi",
                    statistic=psi,
                    threshold=PSI_MODERATE,
                    drifted=psi >= PSI_MODERATE,
                    severity=severity,
                    n_reference=len(ref_values),
                    n_current=len(cur_values),
                )
            )
        else:
            statistic, p_value = kolmogorov_smirnov(ref_values, cur_values)
            severe = p_value < (ks_significance / 5)
            severity = "severe" if severe else "moderate" if p_value < ks_significance else "stable"
            results.append(
                DriftResult(
                    feature=feature,
                    method="ks",
                    statistic=statistic,
                    threshold=ks_significance,
                    drifted=p_value < ks_significance,
                    severity=severity,
                    p_value=p_value,
                    n_reference=len(ref_values),
                    n_current=len(cur_values),
                )
            )

    return DriftReport(
        reference_window=reference_window,
        current_window=current_window,
        method=method,
        results=results,
    )


def write_drift_report(
    report: DriftReport, path: Union[str, Path]
) -> Path:
    """Persist a DriftReport to disk as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2))
    logger.info("Wrote drift report to %s", out)
    return out
