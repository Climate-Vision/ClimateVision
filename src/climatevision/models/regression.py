"""
Biomass and carbon-stock regression models for ClimateVision.

Where the U-Net produces per-pixel deforestation masks, this module
turns those masks (plus the underlying spectral indices) into a scalar
estimate of above-ground biomass (Mg/ha) and the carbon equivalent
(tCO2e). It supports two regressors out of the box:

- ``"random_forest"`` — sklearn RandomForestRegressor (default).
- ``"xgboost"``       — XGBRegressor when xgboost is installed.

The conversion from biomass to CO2e uses the IPCC default carbon
fraction of 0.47 and the molecular-weight ratio 44/12. Both constants
are exposed so users can override them per ecosystem.

Typical usage::

    from climatevision.models.regression import (
        BiomassRegressor, biomass_to_carbon, biomass_to_co2e,
    )

    reg = BiomassRegressor(model_type="random_forest").fit(X_train, y_train)
    biomass = reg.predict(X_test)        # Mg/ha
    co2e = biomass_to_co2e(biomass)      # tCO2e/ha
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

CARBON_FRACTION = 0.47           # IPCC default for tropical forests
CO2_TO_C_RATIO = 44.0 / 12.0     # molecular weight ratio
DEFAULT_FEATURE_NAMES = (
    "ndvi",
    "evi",
    "savi",
    "ndmi",
    "nbr",
    "red",
    "green",
    "blue",
    "nir",
    "swir1",
)


def biomass_to_carbon(biomass: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert above-ground biomass (Mg/ha) to carbon (tC/ha)."""
    return np.asarray(biomass) * CARBON_FRACTION


def biomass_to_co2e(biomass: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert above-ground biomass (Mg/ha) to CO2 equivalent (tCO2e/ha)."""
    return biomass_to_carbon(biomass) * CO2_TO_C_RATIO


@dataclass
class RegressionMetrics:
    rmse: float
    mae: float
    r2: float
    mape: float

    def to_dict(self) -> dict[str, float]:
        return {"rmse": self.rmse, "mae": self.mae, "r2": self.r2, "mape": self.mape}


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Compute RMSE / MAE / R² / MAPE for a regression run."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: y_true={y_true.shape} y_pred={y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("cannot evaluate empty arrays")

    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot

    return RegressionMetrics(rmse=rmse, mae=mae, r2=r2, mape=_safe_mape(y_true, y_pred))


class BiomassRegressor:
    """
    Wrapper around sklearn / xgboost regressors with a stable API for
    ClimateVision pipelines.
    """

    SUPPORTED_MODELS = ("random_forest", "xgboost")

    def __init__(
        self,
        model_type: str = "random_forest",
        *,
        feature_names: Optional[Sequence[str]] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        random_state: int = 42,
    ) -> None:
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"model_type must be one of {self.SUPPORTED_MODELS}, got {model_type!r}"
            )
        self.model_type = model_type
        self.feature_names = tuple(feature_names) if feature_names else DEFAULT_FEATURE_NAMES
        self.model_kwargs = dict(model_kwargs or {})
        self.random_state = random_state
        self._model: Any = None
        self._fitted = False

    def _build(self) -> Any:
        if self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor

            kwargs = {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_leaf": 2,
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            kwargs.update(self.model_kwargs)
            return RandomForestRegressor(**kwargs)

        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "xgboost is required for model_type='xgboost'. "
                "Install with `pip install xgboost`."
            ) from exc

        kwargs = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "objective": "reg:squarederror",
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        kwargs.update(self.model_kwargs)
        return XGBRegressor(**kwargs)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "BiomassRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"row mismatch: X has {X.shape[0]} rows, y has {y.shape[0]}"
            )

        self._model = self._build()
        if sample_weight is not None:
            self._model.fit(X, y, sample_weight=sample_weight)
        else:
            self._model.fit(X, y)
        self._fitted = True
        logger.info(
            "Trained %s on %d samples with %d features",
            self.model_type,
            X.shape[0],
            X.shape[1],
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("regressor must be fit() before predict()")
        X = np.asarray(X, dtype=np.float64)
        return np.asarray(self._model.predict(X), dtype=np.float64)

    def feature_importances(self) -> dict[str, float]:
        if not self._fitted:
            raise RuntimeError("regressor must be fit() before feature_importances()")
        importances = getattr(self._model, "feature_importances_", None)
        if importances is None:
            raise AttributeError(
                f"underlying {self.model_type} has no feature_importances_"
            )
        names = self.feature_names
        if len(names) != len(importances):
            names = tuple(f"f{i}" for i in range(len(importances)))
        return {name: float(value) for name, value in zip(names, importances)}

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> RegressionMetrics:
        return evaluate_regression(y_true, self.predict(X))

    def save(self, path: Union[str, Path]) -> Path:
        if not self._fitted:
            raise RuntimeError("regressor must be fit() before save()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "model": self._model,
                    "model_type": self.model_type,
                    "feature_names": list(self.feature_names),
                    "random_state": self.random_state,
                },
                fh,
            )
        logger.info("Saved %s regressor to %s", self.model_type, path)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BiomassRegressor":
        path = Path(path)
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        instance = cls(
            model_type=payload["model_type"],
            feature_names=payload["feature_names"],
            random_state=payload["random_state"],
        )
        instance._model = payload["model"]
        instance._fitted = True
        return instance


def estimate_biomass_from_indices(
    indices: dict[str, np.ndarray],
    regressor: BiomassRegressor,
    feature_order: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Build a feature matrix from a dict of spectral-index arrays and run
    inference. The dict is expected to map index name -> 1-D array of
    pixel values (one row per pixel).
    """
    feature_order = tuple(feature_order or regressor.feature_names)
    missing = [k for k in feature_order if k not in indices]
    if missing:
        raise KeyError(f"missing spectral indices: {missing}")

    columns = [np.asarray(indices[k]).reshape(-1) for k in feature_order]
    if len({c.size for c in columns}) != 1:
        raise ValueError("all input indices must have the same length")
    X = np.stack(columns, axis=1)
    return regressor.predict(X)


def serialize_metrics(
    metrics: RegressionMetrics, output_path: Union[str, Path]
) -> Path:
    """Persist regression metrics as JSON for the eval / model-card pipeline."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics.to_dict(), indent=2))
    return output_path
