"""
Production loss functions for imbalanced binary segmentation.

CombinedLoss = α·Focal + (1-α)·Dice

Why both?
  - Focal loss:  penalises confidently wrong predictions; handles class imbalance
                 through the (1-p_t)^γ modulating factor.
  - Dice loss:   optimises region overlap directly; stable when positives are rare.
  - Together:    fast convergence (Focal) + good boundary precision (Dice).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Args:
        alpha:      Scalar weight for the positive class.
        gamma:      Focusing parameter. 0 → standard CE. 2 is a good default.
        class_weights: (n_classes,) tensor of per-class weights (optional).
        ignore_index: Pixel label to ignore (-1 = none).
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(2),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (N, C, H, W) — raw model output
            targets: (N, H, W)    — class indices
        """
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights.to(logits.device),
            ignore_index=self.ignore_index,
            reduction="none",
        )
        p_t = torch.exp(-ce)
        loss = self.alpha * (1.0 - p_t) ** self.gamma * ce
        return loss.mean()


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Differentiable even when predictions are poor.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        n_classes = probs.shape[1]

        # One-hot encode targets → (N, C, H, W)
        targets_oh = F.one_hot(targets.long(), num_classes=n_classes)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()

        # Per-class Dice, averaged
        inter = (probs * targets_oh).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3))
        dice  = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Focal + Dice combined loss.

    Args:
        focal_weight:   Weight of Focal loss (0–1). Dice weight = 1 - focal_weight.
        focal_alpha:    Class balance weight for Focal.
        focal_gamma:    Focusing parameter for Focal.
        class_weights:  Per-class weights for cross-entropy component.
    """

    def __init__(
        self,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.focal_w = focal_weight
        self.dice_w  = 1.0 - focal_weight
        self.focal = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            class_weights=class_weights,
        )
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_w * self.focal(logits, targets) + self.dice_w * self.dice(logits, targets)


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax loss — directly optimises the IoU metric.
    Use as an auxiliary loss in late training for IoU-focused tasks.

    Reference: Berman et al., 2018. https://arxiv.org/abs/1705.08790
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        loss = 0.0
        n_classes = probs.shape[1]
        for c in range(n_classes):
            fg = (targets == c).float()
            errors = (fg - probs[:, c]).abs()
            errors_sorted, perm = torch.sort(errors.view(-1), descending=True)
            fg_sorted = fg.view(-1)[perm]
            loss += torch.dot(errors_sorted, self._lovasz_grad(fg_sorted))
        return loss / n_classes

    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard
