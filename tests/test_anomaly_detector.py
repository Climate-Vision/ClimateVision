"""Tests for governance.anomaly_detector."""

from __future__ import annotations

import numpy as np
import pytest

from climatevision.governance.anomaly_detector import (
    AnomalyDetector,
    extract_features,
    write_anomaly_report,
)


def _normal_confidence(rng: np.random.Generator) -> np.ndarray:
    return np.clip(rng.normal(0.7, 0.05, size=(64, 64)), 0.0, 1.0)


def _degenerate_confidence() -> np.ndarray:
    return np.ones((64, 64)) * 0.999


def test_extract_features_shapes_and_ranges():
    rng = np.random.default_rng(0)
    feats = extract_features(_normal_confidence(rng))
    assert 0.0 <= feats.mean_confidence <= 1.0
    assert feats.std_confidence >= 0
    assert 0.0 <= feats.positive_fraction <= 1.0
    assert feats.entropy >= 0


def test_extract_features_rejects_empty():
    with pytest.raises(ValueError):
        extract_features(np.array([]))


def test_statistical_detector_flags_outlier(tmp_path):
    rng = np.random.default_rng(42)
    detector = AnomalyDetector(history_path=tmp_path / "history.jsonl")

    for _ in range(30):
        detector.detect(_normal_confidence(rng))

    result = detector.detect(_degenerate_confidence())
    assert result.method == "statistical"
    assert result.is_anomaly
    assert result.reasons


def test_isolation_forest_kicks_in_after_threshold(tmp_path):
    rng = np.random.default_rng(7)
    detector = AnomalyDetector(
        history_path=tmp_path / "history.jsonl",
        min_history_for_iforest=20,
    )

    for _ in range(25):
        detector.detect(_normal_confidence(rng))

    assert detector._iforest is not None
    result = detector.detect(_normal_confidence(rng))
    assert result.method == "isolation_forest"


def test_history_persistence_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    history_path = tmp_path / "history.jsonl"

    d1 = AnomalyDetector(history_path=history_path)
    for _ in range(5):
        d1.detect(_normal_confidence(rng))

    d2 = AnomalyDetector(history_path=history_path)
    d2.load_history()
    assert len(d2._history) == 5


def test_write_anomaly_report(tmp_path):
    rng = np.random.default_rng(3)
    detector = AnomalyDetector(history_path=tmp_path / "history.jsonl")

    results = [detector.detect(_normal_confidence(rng)) for _ in range(3)]
    out = write_anomaly_report(results, output_path=tmp_path / "report.json")
    assert out.exists()
    assert out.read_text().count("mean_confidence") == 3
