"""Tests for governance.calibration."""

from __future__ import annotations

import json

import numpy as np
import pytest

from climatevision.governance.calibration import (
    CalibrationReport,
    brier_score,
    evaluate_calibration,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
    write_calibration_report,
)


def _perfectly_calibrated(n: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.0, 1.0, size=n)
    targets = (rng.uniform(0.0, 1.0, size=n) < probs).astype(np.int32)
    return probs, targets


def _overconfident(n: int = 10_000, seed: int = 1):
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.8, 1.0, size=n)
    targets = (rng.uniform(0.0, 1.0, size=n) < 0.5).astype(np.int32)
    return probs, targets


def test_reliability_bins_partition_inputs():
    probs, targets = _perfectly_calibrated()
    bins = reliability_bins(probs, targets, n_bins=10)
    assert len(bins) == 10
    assert sum(b.count for b in bins) == probs.size
    for b in bins:
        assert 0.0 <= b.lower < b.upper <= 1.0


def test_perfectly_calibrated_has_low_ece():
    probs, targets = _perfectly_calibrated()
    bins = reliability_bins(probs, targets, n_bins=15)
    assert expected_calibration_error(bins) < 0.05


def test_overconfident_has_high_ece():
    probs, targets = _overconfident()
    bins = reliability_bins(probs, targets, n_bins=15)
    assert expected_calibration_error(bins) > 0.2


def test_mce_is_at_least_ece():
    probs, targets = _overconfident()
    bins = reliability_bins(probs, targets, n_bins=15)
    assert maximum_calibration_error(bins) >= expected_calibration_error(bins)


def test_brier_score_zero_for_certain_correct_predictions():
    probs = np.array([1.0, 0.0, 1.0, 0.0])
    targets = np.array([1, 0, 1, 0])
    assert brier_score(probs, targets) == pytest.approx(0.0)


def test_brier_score_one_for_certain_wrong_predictions():
    probs = np.array([1.0, 0.0, 1.0, 0.0])
    targets = np.array([0, 1, 0, 1])
    assert brier_score(probs, targets) == pytest.approx(1.0)


def test_evaluate_calibration_returns_report_with_bins():
    probs, targets = _perfectly_calibrated()
    report = evaluate_calibration(
        probs, targets, model_version="unet-test-1", n_bins=10
    )
    assert isinstance(report, CalibrationReport)
    assert report.model_version == "unet-test-1"
    assert report.n_samples == probs.size
    assert report.n_bins == 10
    assert len(report.bins) == 10
    assert 0.0 <= report.ece <= 1.0
    assert 0.0 <= report.brier_score <= 1.0


def test_well_calibrated_threshold():
    probs, targets = _perfectly_calibrated()
    report = evaluate_calibration(probs, targets, model_version="v")
    assert report.is_well_calibrated(ece_threshold=0.05)
    bad_probs, bad_targets = _overconfident()
    bad = evaluate_calibration(bad_probs, bad_targets, model_version="v")
    assert not bad.is_well_calibrated(ece_threshold=0.05)


def test_validates_probability_range():
    with pytest.raises(ValueError, match="probabilities must lie in"):
        evaluate_calibration(
            np.array([1.5, 0.5]), np.array([1, 0]), model_version="v"
        )


def test_validates_binary_targets():
    with pytest.raises(ValueError, match="targets must be binary"):
        evaluate_calibration(
            np.array([0.5, 0.5]), np.array([1, 2]), model_version="v"
        )


def test_validates_shape_match():
    with pytest.raises(ValueError, match="same shape"):
        evaluate_calibration(
            np.array([0.5, 0.5]), np.array([1, 0, 1]), model_version="v"
        )


def test_write_calibration_report_round_trips_json(tmp_path):
    probs, targets = _perfectly_calibrated(n=1000)
    report = evaluate_calibration(probs, targets, model_version="v0.1")
    out = write_calibration_report(report, tmp_path / "calib.json")
    loaded = json.loads(out.read_text())
    assert loaded["model_version"] == "v0.1"
    assert loaded["n_samples"] == 1000
    assert len(loaded["bins"]) == report.n_bins
