"""Tests for governance.drift_detector."""

from __future__ import annotations

import json

import numpy as np
import pytest

from climatevision.governance.drift_detector import (
    DriftReport,
    DriftResult,
    detect_drift,
    kolmogorov_smirnov,
    population_stability_index,
    write_drift_report,
)


def _normal(mean: float, std: float, n: int = 5_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, size=n)


def test_psi_zero_for_identical_samples():
    base = _normal(0, 1, seed=0)
    same = _normal(0, 1, seed=1)
    psi = population_stability_index(base, same)
    assert psi < 0.05


def test_psi_flags_shifted_distribution():
    base = _normal(0, 1, seed=0)
    shifted = _normal(2, 1, seed=1)
    psi = population_stability_index(base, shifted)
    assert psi > 0.25


def test_psi_handles_constant_reference():
    base = np.zeros(1000)
    cur = np.ones(1000)
    psi = population_stability_index(base, cur)
    assert psi >= 0.0
    assert np.isfinite(psi)


def test_ks_pvalue_high_for_identical_distribution():
    base = _normal(0, 1, n=2000, seed=0)
    same = _normal(0, 1, n=2000, seed=1)
    _, p = kolmogorov_smirnov(base, same)
    assert p > 0.05


def test_ks_pvalue_low_for_shifted_distribution():
    base = _normal(0, 1, n=2000, seed=0)
    shifted = _normal(1, 1, n=2000, seed=1)
    statistic, p = kolmogorov_smirnov(base, shifted)
    assert statistic > 0.1
    assert p < 0.01


def test_detect_drift_psi_returns_per_feature_results():
    ref = {
        "mean_confidence": _normal(0.5, 0.1, seed=0),
        "positive_fraction": _normal(0.2, 0.05, seed=1),
    }
    cur = {
        "mean_confidence": _normal(0.5, 0.1, seed=2),
        "positive_fraction": _normal(0.4, 0.05, seed=3),
    }
    report = detect_drift(ref, cur, method="psi")
    assert isinstance(report, DriftReport)
    assert len(report.results) == 2
    by_feature = {r.feature: r for r in report.results}
    assert by_feature["mean_confidence"].severity == "stable"
    assert by_feature["positive_fraction"].severity in {"moderate", "severe"}
    assert report.any_drifted


def test_detect_drift_ks_method():
    ref = {"x": _normal(0, 1, seed=0)}
    cur = {"x": _normal(2, 1, seed=1)}
    report = detect_drift(ref, cur, method="ks")
    assert report.method == "ks"
    assert report.results[0].drifted is True
    assert report.results[0].p_value is not None
    assert report.results[0].p_value < 0.05


def test_detect_drift_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        detect_drift({"x": [1.0]}, {"x": [1.0]}, method="bogus")


def test_detect_drift_rejects_feature_mismatch():
    with pytest.raises(ValueError, match="feature mismatch"):
        detect_drift({"a": [1.0, 2.0]}, {"b": [1.0, 2.0]})


def test_severe_features_isolated():
    ref = {
        "stable_feat": _normal(0, 1, seed=0),
        "drift_feat": _normal(0, 1, seed=1),
    }
    cur = {
        "stable_feat": _normal(0, 1, seed=2),
        "drift_feat": _normal(5, 1, seed=3),
    }
    report = detect_drift(ref, cur, method="psi")
    assert report.severe_features == ["drift_feat"]


def test_validation_rejects_non_finite():
    with pytest.raises(ValueError, match="non-finite"):
        population_stability_index([np.nan, 1.0], [1.0, 2.0])


def test_validation_rejects_empty_window():
    with pytest.raises(ValueError, match="empty"):
        population_stability_index([], [1.0, 2.0])


def test_write_drift_report_round_trips_json(tmp_path):
    ref = {"x": _normal(0, 1, seed=0)}
    cur = {"x": _normal(0, 1, seed=1)}
    report = detect_drift(ref, cur, method="psi")
    out = write_drift_report(report, tmp_path / "drift.json")
    loaded = json.loads(out.read_text())
    assert loaded["method"] == "psi"
    assert "any_drifted" in loaded
    assert len(loaded["results"]) == 1
