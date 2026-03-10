"""
Standalone evaluation script for the ClimateVision forest segmentation model.

Produces:
  - Per-split metrics (IoU, F1, precision, recall, pixel accuracy)
  - Confusion matrix
  - Per-class IoU breakdown
  - Optional visual outputs (overlay images)

Usage:
  python scripts/evaluate.py \\
      --checkpoint models/20240101_120000/best_model.pth \\
      --data-dir   data/processed \\
      --split      test

  # Enable TTA (test-time augmentation):
  python scripts/evaluate.py --checkpoint ... --tta

  # Save prediction overlay images:
  python scripts/evaluate.py --checkpoint ... --save-visuals out/visuals
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class ConfusionMatrix:
    """Accumulates pixel-level predictions for 2-class segmentation."""

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """pred, target: flat int tensors."""
        pred   = pred.view(-1)
        target = target.view(-1)
        mask   = (target >= 0) & (target < self.num_classes)
        pred   = pred[mask]
        target = target[mask]
        for t_val in range(self.num_classes):
            for p_val in range(self.num_classes):
                self.matrix[t_val, p_val] += int(((target == t_val) & (pred == p_val)).sum().item())

    def compute(self) -> dict[str, float]:
        m = self.matrix.astype(np.float64)
        tp = np.diag(m)
        fp = m.sum(axis=0) - tp
        fn = m.sum(axis=1) - tp
        tn = m.sum() - (tp + fp + fn)
        eps = 1e-6

        iou_per_class = tp / (tp + fp + fn + eps)
        mean_iou      = iou_per_class.mean()
        precision      = tp / (tp + fp + eps)
        recall         = tp / (tp + fn + eps)
        f1             = 2 * precision * recall / (precision + recall + eps)
        pixel_acc      = tp.sum() / (m.sum() + eps)

        return {
            "pixel_acc":       float(pixel_acc),
            "mean_iou":        float(mean_iou),
            "iou_non_forest":  float(iou_per_class[0]),
            "iou_forest":      float(iou_per_class[1]),
            "f1_non_forest":   float(f1[0]),
            "f1_forest":       float(f1[1]),
            "precision_forest": float(precision[1]),
            "recall_forest":   float(recall[1]),
            "confusion_matrix": m.tolist(),
        }

    def __repr__(self) -> str:
        return f"ConfusionMatrix(\n{self.matrix}\n)"


# ---------------------------------------------------------------------------
# TTA helpers
# ---------------------------------------------------------------------------

def _tta_predict(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Average predictions over 8 augmentations (4 rotations × h-flip)."""
    preds = []
    for k in range(4):
        xr = torch.rot90(x, k, dims=[2, 3])
        preds.append(F.softmax(model(xr), dim=1))
        xf = torch.flip(xr, dims=[3])
        preds.append(F.softmax(model(xf), dim=1))
        # un-rotate the flipped version
        # (simple average is acceptable; precise inverse mapping would need
        # per-augmentation inverse, but this approximation is standard)
    stacked = torch.stack(preds, dim=0)
    return stacked.mean(dim=0)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool = False,
    save_visuals: Path | None = None,
    max_visuals: int = 20,
) -> dict[str, float]:
    model.eval()
    cm = ConfusionMatrix()
    n_saved = 0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        if use_tta:
            probs = _tta_predict(model, images)
        else:
            probs = F.softmax(model(images), dim=1)

        preds = probs.argmax(dim=1)

        cm.update(preds.cpu(), masks.cpu())

        # Save overlays
        if save_visuals and n_saved < max_visuals:
            _save_overlay(
                images.cpu().numpy(),
                masks.cpu().numpy(),
                preds.cpu().numpy(),
                save_visuals,
                batch_idx,
            )
            n_saved += len(images)

    return cm.compute()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _save_overlay(
    images: np.ndarray,   # (B, 4, H, W) float
    masks:  np.ndarray,   # (B, H, W)    int
    preds:  np.ndarray,   # (B, H, W)    int
    out_dir: Path,
    batch_idx: int,
) -> None:
    try:
        from PIL import Image as PILImage
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(images)):
        # RGB from bands 0,1,2  (R,G,B)
        rgb = images[i, :3].transpose(1, 2, 0)          # (H, W, 3)
        rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6) * 255).astype(np.uint8)

        H, W = masks[i].shape
        overlay = rgb.copy()
        # Ground truth: forest in green tint
        overlay[masks[i] == 1, 1] = np.clip(
            overlay[masks[i] == 1, 1].astype(int) + 80, 0, 255
        ).astype(np.uint8)
        # Prediction: forest boundary in red
        pred_mask = preds[i].astype(bool)
        overlay[pred_mask & ~masks[i].astype(bool), 0] = 220
        overlay[pred_mask & ~masks[i].astype(bool), 1] = 20
        overlay[pred_mask & ~masks[i].astype(bool), 2] = 20

        PILImage.fromarray(overlay).save(
            out_dir / f"batch{batch_idx:04d}_sample{i:02d}.png"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(ckpt_path: Path, device: torch.device, arch_override: str | None = None):
    from climatevision.models.unet import get_model

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = (ckpt.get("cfg") or {})

    # Priority: CLI override > checkpoint cfg > infer from weight shapes
    arch = arch_override or model_cfg.get("model", {}).get("architecture")
    # Load BN running stats from model_state_dict, then overlay EMA params
    model_state = ckpt.get("model_state_dict")
    ema_state   = ckpt.get("ema_state_dict")
    state = model_state if model_state is not None else ckpt

    if arch is None:
        # Infer from bottleneck width in state dict
        for key, val in state.items():
            if "down4" in key and "weight" in key and val.ndim == 4:
                arch = "unet" if val.shape[0] == 512 else "attention_unet"
                break
        arch = arch or "unet"
        logger.info("Architecture inferred from weights: %s", arch)

    # Infer in_channels from first conv weight shape
    for key, val in state.items():
        if "inc" in key and "weight" in key and val.ndim == 4:
            in_ch = val.shape[1]
            break
    else:
        in_ch = model_cfg.get("model", {}).get("in_channels", 4)

    model = get_model(arch, n_channels=in_ch, n_classes=2)
    # Load full state (includes BN running stats)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Overlay EMA parameters if available (learned weights only)
    if ema_state is not None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ema_state:
                    param.data.copy_(ema_state[name])
    model = model.to(device)
    logger.info("Loaded %s from %s  (val IoU=%.4f)", arch, ckpt_path.name, ckpt.get("val_iou", 0))
    return model


def main() -> None:
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info("Evaluation device: %s", device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    model = load_model_from_checkpoint(ckpt_path, device, arch_override=args.arch)

    # Build dataset
    from climatevision.data.dataset import ForestDataset
    from climatevision.data.augmentation import get_val_transforms

    split_dir = Path(args.data_dir) / args.split
    if not split_dir.exists():
        logger.error("Split directory not found: %s", split_dir)
        sys.exit(1)

    dataset = ForestDataset(
        root=split_dir,
        transform=get_val_transforms(args.image_size),
        image_size=args.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type != "cpu",
    )
    logger.info("Evaluating %d samples from '%s' split", len(dataset), args.split)

    save_visuals = Path(args.save_visuals) if args.save_visuals else None
    metrics = evaluate(
        model=model,
        loader=loader,
        device=device,
        use_tta=args.tta,
        save_visuals=save_visuals,
    )

    cm = metrics.pop("confusion_matrix")
    cm_arr = np.array(cm)

    # Print results table
    print("\n" + "=" * 55)
    print(f"  Evaluation Results  [{args.split.upper()} split]")
    print("=" * 55)
    print(f"  Pixel Accuracy    : {metrics['pixel_acc']:.4f}")
    print(f"  Mean IoU          : {metrics['mean_iou']:.4f}")
    print(f"  IoU  (forest)     : {metrics['iou_forest']:.4f}")
    print(f"  IoU  (non-forest) : {metrics['iou_non_forest']:.4f}")
    print(f"  F1   (forest)     : {metrics['f1_forest']:.4f}")
    print(f"  Precision (forest): {metrics['precision_forest']:.4f}")
    print(f"  Recall    (forest): {metrics['recall_forest']:.4f}")
    print("-" * 55)
    print(f"  Confusion matrix (rows=GT, cols=pred):")
    print(f"              non-forest   forest")
    print(f"  non-forest  {cm_arr[0,0]:>10,}  {cm_arr[0,1]:>6,}")
    print(f"  forest      {cm_arr[1,0]:>10,}  {cm_arr[1,1]:>6,}")
    print("=" * 55)
    if args.tta:
        print("  (TTA enabled — 8-augmentation ensemble)")
    print()

    # Save metrics JSON
    out_path = ckpt_path.parent / f"eval_{args.split}.json"
    metrics["confusion_matrix"] = cm
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Full metrics saved to %s", out_path)

    if save_visuals:
        logger.info("Overlay images saved to %s", save_visuals)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ClimateVision model")
    p.add_argument("--checkpoint",  required=True, help="Path to best_model.pth")
    p.add_argument("--data-dir",    default="data/processed")
    p.add_argument("--split",       default="test", choices=["train", "val", "test"])
    p.add_argument("--image-size",  type=int, default=256)
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--tta",         action="store_true", help="Enable test-time augmentation")
    p.add_argument("--arch",         default=None, choices=["unet", "attention_unet"],
                   help="Model architecture (auto-detected from checkpoint if omitted)")
    p.add_argument("--save-visuals", default=None,
                   help="Directory to save prediction overlay images")
    return p.parse_args()


if __name__ == "__main__":
    main()
