"""
Production training loop for forest segmentation.

Features:
  - Mixed-precision training (torch.cuda.amp)
  - Linear LR warm-up → cosine annealing
  - Gradient clipping
  - Exponential Moving Average (EMA) of model weights
  - Early stopping on validation IoU
  - Full metric tracking (loss, IoU, F1, precision, recall, pixel-acc)
  - Checkpointing: best model + periodic snapshots
  - JSON training history log
"""
from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """
    Args:
        preds:   (N, H, W) int64 predicted class indices
        targets: (N, H, W) int64 ground truth
    Returns dict with iou_forest, f1, precision, recall, pixel_acc.
    Pure PyTorch — no numpy dependency.
    """
    p = preds.view(-1)
    t = targets.view(-1)

    tp = int(((p == 1) & (t == 1)).sum().item())
    fp = int(((p == 1) & (t == 0)).sum().item())
    fn = int(((p == 0) & (t == 1)).sum().item())
    tn = int(((p == 0) & (t == 0)).sum().item())

    eps = 1e-6
    iou       = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "iou_forest": iou,
        "f1":         f1,
        "precision":  precision,
        "recall":     recall,
        "pixel_acc":  pixel_acc,
    }


class EMA:
    """Exponential Moving Average of trainable model parameters only.

    Deliberately excludes BatchNorm running statistics so that
    applying EMA weights does not corrupt the model's learned BN stats.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay  = decay
        # Only track trainable parameters (not buffers like BN running stats)
        self.shadow: dict[str, torch.Tensor] = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad or name not in self.shadow:
                    continue
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module) -> None:
        """Copy EMA weights into model parameters, leaving buffers untouched."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module, original_state: dict) -> None:
        model.load_state_dict(original_state)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Self-contained training loop.

    Usage:
        trainer = Trainer(model, criterion, loaders, cfg, save_dir)
        history = trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        loaders: dict[str, DataLoader],
        cfg: dict[str, Any],
        save_dir: str | Path = "models",
    ):
        self.cfg      = cfg
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
        logger.info("Training device: %s", self.device)

        self.model     = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.loaders   = loaders

        # Optimiser
        lr = cfg.get("learning_rate", 1e-4)
        wd = cfg.get("weight_decay", 1e-4)
        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=wd,
        )

        # LR schedule: linear warm-up → cosine annealing
        n_epochs    = cfg.get("epochs", 50)
        warmup_eps  = cfg.get("warmup_epochs", 5)
        warmup_sched = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_eps,
        )
        cosine_sched = CosineAnnealingLR(
            self.optimizer,
            T_max=max(n_epochs - warmup_eps, 1),
            eta_min=cfg.get("min_lr", 1e-6),
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_eps],
        )

        # Mixed precision
        self.use_amp = self.device.type == "cuda" and cfg.get("mixed_precision", True)
        self.scaler  = GradScaler(enabled=self.use_amp)

        # EMA
        self.ema = EMA(self.model, decay=cfg.get("ema_decay", 0.9999)) if cfg.get("use_ema", True) else None

        # Training state
        self.n_epochs       = n_epochs
        self.grad_clip      = cfg.get("grad_clip", 1.0)
        self.patience       = cfg.get("early_stopping_patience", 10)
        self.best_iou       = -1.0
        self.epochs_no_imp  = 0
        self.history: dict[str, list] = {"train": [], "val": []}

    # ------------------------------------------------------------------
    def fit(self) -> dict[str, list]:
        logger.info("=" * 60)
        logger.info("Starting training for %d epochs", self.n_epochs)
        logger.info("=" * 60)

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()

            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch(epoch)

            self.scheduler.step()

            elapsed = time.time() - t0
            self._log_epoch(epoch, train_metrics, val_metrics, elapsed)

            self.history["train"].append(train_metrics)
            self.history["val"].append(val_metrics)

            improved = self._checkpoint(epoch, val_metrics)
            if not improved:
                self.epochs_no_imp += 1
                if self.epochs_no_imp >= self.patience:
                    logger.info(
                        "Early stopping: no improvement for %d epochs.", self.patience
                    )
                    break
            else:
                self.epochs_no_imp = 0

        self._save_history()
        logger.info("Training complete. Best val IoU: %.4f", self.best_iou)
        return self.history

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        loader = self.loaders["train"]

        total_loss = 0.0
        all_metrics: list[dict] = []

        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss   = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema is not None:
                self.ema.update(self.model)

            preds = logits.argmax(dim=1)
            m = _compute_metrics(preds.detach(), masks.detach())
            m["loss"] = loss.item()
            all_metrics.append(m)
            total_loss += loss.item()

        keys = list(all_metrics[0].keys())
        return {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in keys}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict[str, float]:
        # Use EMA weights for validation if available
        original_state = None
        if self.ema is not None:
            original_state = deepcopy(self.model.state_dict())
            self.ema.apply(self.model)

        self.model.eval()
        loader = self.loaders.get("val")
        if loader is None:
            return {}

        all_metrics: list[dict] = []

        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss   = self.criterion(logits, masks)

            preds = logits.argmax(dim=1)
            m = _compute_metrics(preds, masks)
            m["loss"] = loss.item()
            all_metrics.append(m)

        if original_state is not None:
            self.model.load_state_dict(original_state)

        keys = list(all_metrics[0].keys())
        return {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in keys}

    # ------------------------------------------------------------------
    def _checkpoint(self, epoch: int, val_metrics: dict) -> bool:
        """Save best model + periodic checkpoints. Returns True if improved."""
        iou = val_metrics.get("iou_forest", 0.0)
        improved = iou > self.best_iou

        if improved:
            self.best_iou = iou
            state = {
                "epoch":             epoch,
                "model_state_dict":  self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss":          val_metrics.get("loss", 0.0),
                "val_iou":           iou,
                "val_f1":            val_metrics.get("f1", 0.0),
                "cfg":               self.cfg,
            }
            if self.ema is not None:
                state["ema_state_dict"] = self.ema.shadow
            torch.save(state, self.save_dir / "best_model.pth")
            logger.info(
                "  ✓ New best model saved  (IoU %.4f  F1 %.4f)",
                iou,
                val_metrics.get("f1", 0.0),
            )

        # Periodic checkpoint every 10 epochs
        checkpoint_interval = self.cfg.get("checkpoint_interval", 10)
        if epoch % checkpoint_interval == 0:
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_iou":          iou,
                },
                self.save_dir / f"checkpoint_epoch_{epoch:04d}.pth",
            )

        return improved

    # ------------------------------------------------------------------
    def _log_epoch(
        self,
        epoch: int,
        train: dict,
        val: dict,
        elapsed: float,
    ) -> None:
        lr = self.optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %3d/%d | lr %.2e | "
            "train loss %.4f  iou %.4f  f1 %.4f | "
            "val   loss %.4f  iou %.4f  f1 %.4f | "
            "%.1f s",
            epoch, self.n_epochs, lr,
            train.get("loss", 0), train.get("iou_forest", 0), train.get("f1", 0),
            val.get("loss", 0),   val.get("iou_forest", 0),   val.get("f1", 0),
            elapsed,
        )

    # ------------------------------------------------------------------
    def _save_history(self) -> None:
        history_path = self.save_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info("Training history saved to %s", history_path)
