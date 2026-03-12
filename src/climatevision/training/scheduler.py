"""
Learning rate schedulers for training.

Provides composable scheduler factories and a warm-restart cosine schedule.
"""
from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    LinearLR,
    SequentialLR,
    _LRScheduler,
)


class WarmupCosineScheduler(_LRScheduler):
    """
    Linear warmup followed by cosine annealing to ``eta_min``.

    This is a single scheduler object (no ``SequentialLR`` needed) which
    makes checkpoint resume simpler.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total training epochs.
        eta_min: Minimum learning rate after decay.
        last_epoch: Index of the last epoch (for resume).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # Linear warmup from eta_min → base_lr
            alpha = epoch / max(self.warmup_epochs, 1)
            return [
                self.eta_min + alpha * (base_lr - self.eta_min)
                for base_lr in self.base_lrs
            ]
        # Cosine annealing
        progress = (epoch - self.warmup_epochs) / max(
            self.total_epochs - self.warmup_epochs, 1
        )
        return [
            self.eta_min
            + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


class WarmupCosineWarmRestarts(_LRScheduler):
    """
    Linear warmup followed by cosine annealing with warm restarts (SGDR).

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of warmup epochs.
        restart_period: Epochs per cosine cycle after warmup.
        restart_mult: Multiplier for cycle length after each restart.
        eta_min: Minimum learning rate.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 5,
        restart_period: int = 20,
        restart_mult: int = 2,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.restart_period = restart_period
        self.restart_mult = restart_mult
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            alpha = epoch / max(self.warmup_epochs, 1)
            return [
                self.eta_min + alpha * (base_lr - self.eta_min)
                for base_lr in self.base_lrs
            ]

        t = epoch - self.warmup_epochs
        period = self.restart_period
        # Find which cycle we're in
        while t >= period:
            t -= period
            period = int(period * self.restart_mult)
        progress = t / period
        return [
            self.eta_min
            + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "warmup_cosine",
    **kwargs,
) -> _LRScheduler:
    """
    Build a learning rate scheduler from config.

    Args:
        optimizer: Wrapped optimizer.
        scheduler_type: One of ``warmup_cosine``, ``warmup_cosine_restarts``,
            ``cosine``, ``linear``.
        **kwargs: Scheduler-specific keyword arguments.

    Returns:
        Learning rate scheduler instance.
    """
    if scheduler_type == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=kwargs.get("warmup_epochs", 5),
            total_epochs=kwargs.get("total_epochs", 100),
            eta_min=kwargs.get("eta_min", 1e-6),
        )
    if scheduler_type == "warmup_cosine_restarts":
        return WarmupCosineWarmRestarts(
            optimizer,
            warmup_epochs=kwargs.get("warmup_epochs", 5),
            restart_period=kwargs.get("restart_period", 20),
            restart_mult=kwargs.get("restart_mult", 2),
            eta_min=kwargs.get("eta_min", 1e-6),
        )
    if scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("total_epochs", 100),
            eta_min=kwargs.get("eta_min", 1e-6),
        )
    if scheduler_type == "linear":
        return LinearLR(
            optimizer,
            start_factor=kwargs.get("start_factor", 1.0),
            end_factor=kwargs.get("end_factor", 0.01),
            total_iters=kwargs.get("total_epochs", 100),
        )
    raise ValueError(f"Unknown scheduler type: {scheduler_type}")
