"""Tests for models.regression.

Imports the module via importlib because ``climatevision.models.__init__``
pulls in torch-based U-Net code that is heavy and unrelated to the
regression module under test.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "climatevision"
    / "models"
    / "regression.py"
)
_spec = importlib.util.spec_from_file_location("cv_regression", _PATH)
reg = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["cv_regression"] = reg
_spec.loader.exec_module(reg)


def _synthetic_dataset(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(low=0.0, high=1.0, size=(n, 10))
    # biomass = 200 * NDVI + 50 * EVI + 20 * NIR + noise
    y = 200 * X[:, 0] + 50 * X[:, 1] + 20 * X[:, 8] + rng.normal(0, 5, size=n)
    return X, y


def test_biomass_to_carbon_uses_default_fraction():
    assert reg.biomass_to_carbon(100.0) == pytest.approx(47.0)
    arr = reg.biomass_to_carbon(np.array([0.0, 100.0, 200.0]))
    np.testing.assert_allclose(arr, [0.0, 47.0, 94.0])


def test_biomass_to_co2e_round_trip():
    co2e = reg.biomass_to_co2e(100.0)
    assert co2e == pytest.approx(100.0 * 0.47 * (44.0 / 12.0))


def test_evaluate_regression_perfect_fit():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = reg.evaluate_regression(y, y)
    assert metrics.rmse == 0.0
    assert metrics.mae == 0.0
    assert metrics.r2 == pytest.approx(1.0)
    assert metrics.mape == pytest.approx(0.0)


def test_evaluate_regression_shape_mismatch_raises():
    with pytest.raises(ValueError):
        reg.evaluate_regression(np.array([1.0, 2.0]), np.array([1.0]))


def test_random_forest_fit_predict_evaluate():
    X, y = _synthetic_dataset(n=300)
    r = reg.BiomassRegressor(model_type="random_forest", model_kwargs={"n_estimators": 50})
    r.fit(X, y)
    metrics = r.evaluate(X, y)
    assert metrics.rmse < 25.0  # very loose, just sanity
    assert metrics.r2 > 0.5

    importances = r.feature_importances()
    assert set(importances) == set(reg.DEFAULT_FEATURE_NAMES)
    assert pytest.approx(sum(importances.values()), rel=1e-6) == 1.0


def test_unsupported_model_type_raises():
    with pytest.raises(ValueError):
        reg.BiomassRegressor(model_type="lightgbm")


def test_predict_before_fit_raises():
    r = reg.BiomassRegressor()
    with pytest.raises(RuntimeError):
        r.predict(np.zeros((1, 10)))


def test_save_and_load_round_trip(tmp_path):
    X, y = _synthetic_dataset(n=200)
    r = reg.BiomassRegressor(model_type="random_forest", model_kwargs={"n_estimators": 30})
    r.fit(X, y)

    out = r.save(tmp_path / "rf.pkl")
    assert out.exists()

    loaded = reg.BiomassRegressor.load(out)
    np.testing.assert_allclose(loaded.predict(X), r.predict(X))


def test_estimate_biomass_from_indices(tmp_path):
    X, y = _synthetic_dataset(n=200)
    r = reg.BiomassRegressor(model_kwargs={"n_estimators": 30}).fit(X, y)

    indices = {name: X[:, i] for i, name in enumerate(reg.DEFAULT_FEATURE_NAMES)}
    pred = reg.estimate_biomass_from_indices(indices, r)
    assert pred.shape == (X.shape[0],)
    np.testing.assert_allclose(pred, r.predict(X))


def test_estimate_biomass_missing_index_raises():
    r = reg.BiomassRegressor(model_kwargs={"n_estimators": 5})
    r.fit(*_synthetic_dataset(n=80))

    incomplete = {name: np.zeros(10) for name in reg.DEFAULT_FEATURE_NAMES[:-1]}
    with pytest.raises(KeyError):
        reg.estimate_biomass_from_indices(incomplete, r)


def test_serialize_metrics_writes_json(tmp_path):
    metrics = reg.RegressionMetrics(rmse=1.0, mae=0.5, r2=0.9, mape=0.05)
    out = reg.serialize_metrics(metrics, tmp_path / "metrics.json")
    payload = json.loads(out.read_text())
    assert payload == {"rmse": 1.0, "mae": 0.5, "r2": 0.9, "mape": 0.05}
