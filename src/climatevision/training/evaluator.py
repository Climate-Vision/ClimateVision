"""
Model evaluation framework for forest segmentation and change detection.

Provides comprehensive metrics computation, confusion matrix analysis,
test-time augmentation (TTA), and per-region evaluation reporting.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

class ConfusionMatrix:
    """Accumulates a pixel-level confusion matrix across batches."""

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds: np.ndarray, targets: np.ndarray) -> None:
        p = preds.flatten()
        t = targets.flatten()
        valid = (t >= 0) & (t < self.num_classes)
        indices = t[valid] * self.num_classes + p[valid]
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def reset(self) -> None:
        self.matrix.fill(0)

    # --- per-class metrics ------------------------------------------------

    def iou(self) -> np.ndarray:
        tp = np.diag(self.matrix)
        fp = self.matrix.sum(axis=0) - tp
        fn = self.matrix.sum(axis=1) - tp
        denom = tp + fp + fn
        iou = np.where(denom > 0, tp / denom, 0.0)
        return iou

    def dice(self) -> np.ndarray:
        tp = np.diag(self.matrix)
        fp = self.matrix.sum(axis=0) - tp
        fn = self.matrix.sum(axis=1) - tp
        denom = 2 * tp + fp + fn
        return np.where(denom > 0, 2 * tp / denom, 0.0)

    def precision(self) -> np.ndarray:
        tp = np.diag(self.matrix)
        fp = self.matrix.sum(axis=0) - tp
        denom = tp + fp
        return np.where(denom > 0, tp / denom, 0.0)

    def recall(self) -> np.ndarray:
        tp = np.diag(self.matrix)
        fn = self.matrix.sum(axis=1) - tp
        denom = tp + fn
        return np.where(denom > 0, tp / denom, 0.0)

    def f1(self) -> np.ndarray:
        p = self.precision()
        r = self.recall()
        denom = p + r
        return np.where(denom > 0, 2 * p * r / denom, 0.0)

    def pixel_accuracy(self) -> float:
        total = self.matrix.sum()
        if total == 0:
            return 0.0
        return float(np.diag(self.matrix).sum() / total)

    def kappa(self) -> float:
        total = self.matrix.sum()
        if total == 0:
            return 0.0
        po = np.diag(self.matrix).sum() / total
        pe = (self.matrix.sum(axis=0) * self.matrix.sum(axis=1)).sum() / (total ** 2)
        if pe == 1.0:
            return 0.0
        return float((po - pe) / (1 - pe))

    def summary(self, class_names: list[str] | None = None) -> dict[str, Any]:
        if class_names is None:
            class_names = [f"class_{i}" for i in range(self.num_classes)]

        iou = self.iou()
        dice = self.dice()
        prec = self.precision()
        rec = self.recall()
        f1 = self.f1()

        per_class = {}
        for i, name in enumerate(class_names):
            per_class[name] = {
                "iou": float(iou[i]),
                "dice": float(dice[i]),
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
            }

        return {
            "pixel_accuracy": self.pixel_accuracy(),
            "mean_iou": float(np.mean(iou)),
            "mean_f1": float(np.mean(f1)),
            "kappa": self.kappa(),
            "per_class": per_class,
            "confusion_matrix": self.matrix.tolist(),
        }


# ---------------------------------------------------------------------------
# Test-Time Augmentation
# ---------------------------------------------------------------------------

_TTA_TRANSFORMS = [
    lambda x: x,
    lambda x: torch.flip(x, dims=[-1]),              # horizontal flip
    lambda x: torch.flip(x, dims=[-2]),              # vertical flip
    lambda x: torch.flip(x, dims=[-1, -2]),          # both flips
    lambda x: torch.rot90(x, k=1, dims=[-2, -1]),   # 90° rotation
    lambda x: torch.rot90(x, k=2, dims=[-2, -1]),   # 180° rotation
    lambda x: torch.rot90(x, k=3, dims=[-2, -1]),   # 270° rotation
    lambda x: torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-1]),  # 90° + flip
]

_TTA_INVERSES = [
    lambda x: x,
    lambda x: torch.flip(x, dims=[-1]),
    lambda x: torch.flip(x, dims=[-2]),
    lambda x: torch.flip(x, dims=[-1, -2]),
    lambda x: torch.rot90(x, k=-1, dims=[-2, -1]),
    lambda x: torch.rot90(x, k=-2, dims=[-2, -1]),
    lambda x: torch.rot90(x, k=-3, dims=[-2, -1]),
    lambda x: torch.rot90(torch.flip(x, dims=[-1]), k=-1, dims=[-2, -1]),
]


def tta_predict(
    model: nn.Module,
    images: torch.Tensor,
    num_augments: int = 8,
) -> torch.Tensor:
    """
    Test-time augmentation: average softmax predictions across augmented views.

    Args:
        model: Segmentation model returning logits (N, C, H, W).
        images: Input batch (N, C_in, H, W).
        num_augments: Number of augmented views (1–8).

    Returns:
        Averaged softmax probabilities (N, C, H, W).
    """
    num_augments = min(num_augments, len(_TTA_TRANSFORMS))
    accum = None

    for fwd, inv in zip(_TTA_TRANSFORMS[:num_augments], _TTA_INVERSES[:num_augments]):
        aug_input = fwd(images)
        logits = model(aug_input)
        probs = F.softmax(logits, dim=1)
        probs = inv(probs)
        accum = probs if accum is None else accum + probs

    return accum / num_augments


# ---------------------------------------------------------------------------
# Monte Carlo Dropout Uncertainty
# ---------------------------------------------------------------------------

def mc_dropout_predict(
    model: nn.Module,
    images: torch.Tensor,
    n_forward: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Monte Carlo Dropout for per-pixel uncertainty estimation.

    Enables dropout at inference time and collects multiple stochastic
    forward passes. Returns the mean prediction and pixel-wise entropy.

    Args:
        model: Model with dropout layers.
        images: Input batch (N, C, H, W).
        n_forward: Number of stochastic forward passes.

    Returns:
        mean_probs: Averaged softmax probabilities (N, C, H, W).
        entropy:    Per-pixel entropy (N, H, W) — higher = more uncertain.
    """
    # Enable dropout layers (keep batchnorm in eval mode)
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    stacked = []
    with torch.no_grad():
        for _ in range(n_forward):
            logits = model(images)
            stacked.append(F.softmax(logits, dim=1))

    model.eval()  # restore full eval mode

    probs = torch.stack(stacked, dim=0)           # (T, N, C, H, W)
    mean_probs = probs.mean(dim=0)                # (N, C, H, W)
    entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum(dim=1)  # (N, H, W)

    return mean_probs, entropy


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    metrics: dict[str, Any]
    confusion_matrix: list[list[int]]
    per_class: dict[str, dict[str, float]]
    num_samples: int
    model_name: str = ""
    dataset_name: str = ""


class Evaluator:
    """
    Comprehensive model evaluation on a test DataLoader.

    Usage::

        evaluator = Evaluator(model, test_loader, device="cuda", num_classes=2)
        result = evaluator.evaluate()
        evaluator.save_report(result, "models/eval_report.json")
    """

    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader,
        *,
        device: str | torch.device = "cpu",
        num_classes: int = 2,
        class_names: list[str] | None = None,
        use_tta: bool = False,
        tta_augments: int = 8,
        use_mc_dropout: bool = False,
        mc_forward_passes: int = 10,
    ):
        self.model = model
        self.loader = loader
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.class_names = class_names or ["non_forest", "forest"]
        self.use_tta = use_tta
        self.tta_augments = tta_augments
        self.use_mc_dropout = use_mc_dropout
        self.mc_forward_passes = mc_forward_passes

    @torch.no_grad()
    def evaluate(self) -> EvaluationResult:
        self.model.to(self.device)
        self.model.eval()

        cm = ConfusionMatrix(self.num_classes)
        total_samples = 0
        all_entropy: list[np.ndarray] = []

        for images, masks in self.loader:
            images = images.to(self.device, non_blocking=True)
            masks_np = masks.numpy()

            if self.use_mc_dropout:
                mean_probs, entropy = mc_dropout_predict(
                    self.model, images, self.mc_forward_passes
                )
                preds = mean_probs.argmax(dim=1).cpu().numpy()
                all_entropy.append(entropy.cpu().numpy())
            elif self.use_tta:
                probs = tta_predict(self.model, images, self.tta_augments)
                preds = probs.argmax(dim=1).cpu().numpy()
            else:
                logits = self.model(images)
                preds = logits.argmax(dim=1).cpu().numpy()

            cm.update(preds, masks_np)
            total_samples += images.size(0)

        summary = cm.summary(self.class_names)

        if all_entropy:
            ent = np.concatenate(all_entropy)
            summary["mean_entropy"] = float(ent.mean())
            summary["max_entropy"] = float(ent.max())

        return EvaluationResult(
            metrics=summary,
            confusion_matrix=summary["confusion_matrix"],
            per_class=summary["per_class"],
            num_samples=total_samples,
        )

    @staticmethod
    def save_report(result: EvaluationResult, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "model_name": result.model_name,
            "dataset_name": result.dataset_name,
            "num_samples": result.num_samples,
            "metrics": result.metrics,
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Evaluation report saved to %s", path)

    @staticmethod
    def compare_models(results: list[EvaluationResult]) -> dict[str, Any]:
        """Compare evaluation results across multiple models."""
        comparison: dict[str, Any] = {}
        for r in results:
            name = r.model_name or "unnamed"
            comparison[name] = {
                "mean_iou": r.metrics.get("mean_iou", 0),
                "mean_f1": r.metrics.get("mean_f1", 0),
                "pixel_accuracy": r.metrics.get("pixel_accuracy", 0),
                "kappa": r.metrics.get("kappa", 0),
            }
        return comparison
