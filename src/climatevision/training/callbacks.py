"""
Training callbacks for extensible training loop hooks.

Callbacks are invoked by the Trainer at specific points:
  - on_train_begin / on_train_end
  - on_epoch_begin / on_epoch_end
  - on_batch_end
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Callback:
    """Base callback — all hooks are no-ops by default."""

    should_stop: bool = False

    def on_train_begin(self, state: dict[str, Any]) -> None: ...
    def on_train_end(self, state: dict[str, Any]) -> None: ...
    def on_epoch_begin(self, epoch: int, state: dict[str, Any]) -> None: ...
    def on_epoch_end(self, epoch: int, state: dict[str, Any]) -> None: ...
    def on_batch_end(self, batch: int, state: dict[str, Any]) -> None: ...


class CallbackRunner:
    """Dispatches hook calls to a list of callbacks."""

    def __init__(self, callbacks: list[Callback] | None = None):
        self.callbacks = callbacks or []

    def on_train_begin(self, state: dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(state)

    def on_train_end(self, state: dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_train_end(state)

    def on_epoch_begin(self, epoch: int, state: dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, state)

    def on_epoch_end(self, epoch: int, state: dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, state)

    def on_batch_end(self, batch: int, state: dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(batch, state)


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping(Callback):
    """
    Stop training when a monitored metric stops improving.

    Args:
        monitor: Metric key to watch (e.g. ``val_iou_forest``).
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: ``max`` (higher is better) or ``min`` (lower is better).
    """

    def __init__(
        self,
        monitor: str = "val_iou_forest",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "max":
            return current > self.best + self.min_delta
        return current < self.best - self.min_delta

    def on_epoch_end(self, epoch: int, state: dict[str, Any]) -> None:
        current = state.get(self.monitor)
        if current is None:
            return
        if self._is_improvement(current):
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "EarlyStopping: no improvement in %s for %d epochs (best=%.4f)",
                    self.monitor,
                    self.patience,
                    self.best,
                )


# ---------------------------------------------------------------------------
# MLflow Logger
# ---------------------------------------------------------------------------

class MLflowLogger(Callback):
    """
    Log metrics, parameters, and artefacts to MLflow.

    Requires ``mlflow`` to be installed. If unavailable, logging is silently
    skipped so training can proceed without the dependency.
    """

    def __init__(
        self,
        experiment_name: str = "climatevision",
        run_name: str | None = None,
        tracking_uri: str | None = None,
        log_model: bool = True,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.log_model = log_model
        self._mlflow: Optional[ModuleType] = None
        self._run: Any = None

    def _try_import(self) -> bool:
        if self._mlflow is not None:
            return True
        try:
            import mlflow
            self._mlflow = mlflow
            return True
        except ImportError:
            logger.warning("mlflow not installed — MLflowLogger disabled")
            return False

    def on_train_begin(self, state: dict[str, Any]) -> None:
        if not self._try_import() or self._mlflow is None:
            return
        mlflow = self._mlflow
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._run = mlflow.start_run(run_name=self.run_name)

        # Log hyperparameters
        params = state.get("cfg", {})
        flat = self._flatten_dict(params)
        mlflow.log_params(flat)

    def on_epoch_end(self, epoch: int, state: dict[str, Any]) -> None:
        if self._mlflow is None:
            return
        train = state.get("train_metrics", {})
        val = state.get("val_metrics", {})
        for k, v in train.items():
            self._mlflow.log_metric(f"train_{k}", v, step=epoch)
        for k, v in val.items():
            self._mlflow.log_metric(f"val_{k}", v, step=epoch)
        lr = state.get("lr")
        if lr is not None:
            self._mlflow.log_metric("lr", lr, step=epoch)

    def on_train_end(self, state: dict[str, Any]) -> None:
        if self._mlflow is None:
            return
        # Log best model artefact
        if self.log_model:
            save_dir = state.get("save_dir")
            if save_dir:
                best = Path(save_dir) / "best_model.pth"
                if best.exists():
                    self._mlflow.log_artifact(str(best))
                history = Path(save_dir) / "training_history.json"
                if history.exists():
                    self._mlflow.log_artifact(str(history))
        self._mlflow.end_run()

    @staticmethod
    def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowLogger._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class EpochTimer(Callback):
    """Logs wall-clock time per epoch."""

    def __init__(self):
        self._t0: float = 0.0

    def on_epoch_begin(self, epoch: int, state: dict[str, Any]) -> None:
        self._t0 = time.time()

    def on_epoch_end(self, epoch: int, state: dict[str, Any]) -> None:
        elapsed = time.time() - self._t0
        state["epoch_time"] = elapsed
        logger.info("Epoch %d completed in %.1f s", epoch, elapsed)
