"""
Train a ClimateVision U-Net on a REAL labeled dataset (not synthetic).

This is the production training entrypoint that replaces the synthetic-data path
in run_training.py. It is config-driven: channels and classes are read from
config.yaml for the chosen analysis type, so the same script trains
deforestation (4 bands / 2 classes), flooding (3 bands / 3 classes), and
ice_melting (4 bands / 3 classes).

It reuses the existing infrastructure:
  - data.dataset.create_dataloaders  (expects <data-dir>/{train,val,test}/{images,masks}/*.tif)
  - models.unet.UNet
  - training.losses.CombinedLoss
  - training.trainer.Trainer        (saves best_model.pth with model_state_dict + val_iou)

Data layout required (stem-matched GeoTIFF pairs):
    <data-dir>/
      train/ images/ *.tif   masks/ *.tif
      val/   images/ *.tif   masks/ *.tif
      test/  images/ *.tif   masks/ *.tif

Usage:
    python scripts/train_real.py --analysis-type flooding \
        --data-dir data/datasets/flooding --epochs 50 --batch-size 8 \
        --out /content/drive/MyDrive/climatevision/models

After training, export to ONNX so the API serves it:
    python scripts/export_model.py --checkpoint <out>/<run>/best_model.pth
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from climatevision.data.band_mapping import get_model_config          # noqa: E402
from climatevision.data.dataset import create_dataloaders             # noqa: E402
from climatevision.models.unet import UNet                            # noqa: E402
from climatevision.training.losses import CombinedLoss                # noqa: E402
from climatevision.training.trainer import Trainer                    # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("train_real")


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a ClimateVision model on real data.")
    ap.add_argument("--analysis-type", required=True,
                    choices=["deforestation", "flooding", "ice_melting"])
    ap.add_argument("--data-dir", required=True,
                    help="Root with train/ val/ test/ subdirs of images/ + masks/")
    ap.add_argument("--out", default="models", help="Output dir for checkpoints")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    # Channels / classes come from config.yaml so we never hardcode per type.
    model_cfg = get_model_config(args.analysis_type)
    n_channels = int(model_cfg.get("in_channels", 4))
    n_classes = int(model_cfg.get("num_classes", 2))
    logger.info("Analysis=%s  channels=%d  classes=%d", args.analysis_type, n_channels, n_classes)

    data_dir = Path(args.data_dir)
    if not (data_dir / "train").exists():
        logger.error("No train/ split under %s. Convert the dataset to the "
                     "train|val|test / images|masks layout first (see TRAINING_HANDOFF.md).",
                     data_dir)
        return 2

    loaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        n_channels=n_channels,
        # The weighted sampler assumes binary forest masks; only use it for 2-class.
        use_weighted_sampler=(n_classes == 2),
    )
    if "train" not in loaders:
        logger.error("No training data loaded. Check %s/train/{images,masks}.", data_dir)
        return 2

    model = UNet(n_channels=n_channels, n_classes=n_classes)
    criterion = CombinedLoss(focal_weight=0.5)

    cfg = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "warmup_epochs": min(5, max(1, args.epochs // 10)),
        "use_ema": True,
        "mixed_precision": True,
        "early_stopping_patience": 10,
        "grad_clip": 1.0,
    }

    run_name = f"{args.analysis_type}_{datetime.now():%Y%m%d_%H%M%S}"
    save_dir = Path(args.out) / run_name
    logger.info("Saving checkpoints to %s", save_dir)

    trainer = Trainer(model, criterion, loaders, cfg, save_dir=save_dir)
    history = trainer.fit()

    best = save_dir / "best_model.pth"
    logger.info("Training complete. Best checkpoint: %s", best)
    logger.info("Next: python scripts/export_model.py --checkpoint %s", best)
    if history.get("val"):
        logger.info("Final val metrics: %s", history["val"][-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
