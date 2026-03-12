"""
Hyperparameter search using Optuna for ClimateVision models.

Usage:
  python scripts/hyperparameter_search.py --config config/train.yaml --n-trials 50
  python scripts/hyperparameter_search.py --data-dir data/processed --n-trials 20 --study-name my_study
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load_yaml(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML not installed. Run: pip install pyyaml")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f) or {}


def objective(trial, base_cfg: dict, data_dir: Path) -> float:
    """
    Single Optuna trial: sample hyperparameters, train, return val IoU.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from climatevision.models.unet import get_model
    from climatevision.training.losses import CombinedLoss, FocalLoss, DiceLoss
    from climatevision.training.trainer import Trainer

    # --- Sample hyperparameters -----------------------------------------
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    focal_gamma = trial.suggest_float("focal_gamma", 0.5, 5.0)
    focal_weight = trial.suggest_float("focal_weight", 0.2, 0.8)
    ema_decay = trial.suggest_float("ema_decay", 0.99, 0.9999, log=True)
    architecture = trial.suggest_categorical(
        "architecture", ["unet", "attention_unet", "resnet_unet"]
    )

    # Clone base config and apply sampled values
    cfg = {
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "epochs": base_cfg.get("schedule", {}).get("epochs", 30),
        "warmup_epochs": base_cfg.get("schedule", {}).get("warmup_epochs", 3),
        "mixed_precision": base_cfg.get("training", {}).get("mixed_precision", True),
        "grad_clip": base_cfg.get("training", {}).get("grad_clip", 1.0),
        "use_ema": True,
        "ema_decay": ema_decay,
        "early_stopping_patience": 8,
        "checkpoint_interval": 999,  # don't save periodic checkpoints during search
    }

    # Model
    mcfg = base_cfg.get("model", {})
    n_ch = mcfg.get("in_channels", 4)
    n_cls = mcfg.get("num_classes", 2)

    model_kwargs = {"n_channels": n_ch, "n_classes": n_cls}
    if architecture == "unet":
        model_kwargs["bilinear"] = True
    model = get_model(architecture, **model_kwargs)

    # Loss
    criterion = CombinedLoss(
        focal_weight=focal_weight,
        focal_gamma=focal_gamma,
    )

    # Data
    try:
        from climatevision.data.dataset import create_dataloaders
        loaders = create_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=min(2, base_cfg.get("data", {}).get("num_workers", 2)),
            image_size=mcfg.get("image_size", 256) if "image_size" in mcfg else 256,
            pin_memory=False,
            use_weighted_sampler=False,
        )
    except Exception:
        # Fallback: generate small synthetic dataset for search
        logger.info("Creating synthetic data for hyperparameter search")
        img_size = base_cfg.get("data", {}).get("image_size", 128)
        n_train, n_val = 200, 50
        train_x = torch.randn(n_train, n_ch, img_size, img_size)
        train_y = torch.randint(0, n_cls, (n_train, img_size, img_size))
        val_x = torch.randn(n_val, n_ch, img_size, img_size)
        val_y = torch.randint(0, n_cls, (n_val, img_size, img_size))
        loaders = {
            "train": DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True),
            "val": DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size),
        }

    # Train
    save_dir = PROJECT_ROOT / "models" / "optuna" / f"trial_{trial.number:04d}"
    trainer = Trainer(model=model, criterion=criterion, loaders=loaders, cfg=cfg, save_dir=save_dir)
    history = trainer.fit()

    # Return best validation IoU (Optuna maximises by default when direction="maximize")
    best_iou = max((e.get("iou_forest", 0) for e in history["val"]), default=0)
    return best_iou


def main() -> None:
    args = parse_args()

    try:
        import optuna
    except ImportError:
        logger.error("Optuna not installed. Run: pip install optuna")
        sys.exit(1)

    base_cfg: dict = {}
    if args.config and Path(args.config).exists():
        base_cfg = _load_yaml(args.config)

    # Override epochs for faster search
    if args.epochs:
        base_cfg.setdefault("schedule", {})["epochs"] = args.epochs

    data_dir = Path(args.data_dir) if args.data_dir else Path(base_cfg.get("data", {}).get("dir", "data/processed"))

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    study.optimize(
        lambda trial: objective(trial, base_cfg, data_dir),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    # Report
    logger.info("=" * 60)
    logger.info("Best trial:")
    best = study.best_trial
    logger.info("  Value (IoU): %.4f", best.value)
    logger.info("  Params:")
    for k, v in best.params.items():
        logger.info("    %s: %s", k, v)
    logger.info("=" * 60)

    # Save best params
    import json
    out_path = PROJECT_ROOT / "models" / "optuna" / "best_params.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"best_value": best.value, "best_params": best.params}, f, indent=2)
    logger.info("Best params saved to %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna hyperparameter search for ClimateVision")
    p.add_argument("--config", default=str(PROJECT_ROOT / "config" / "train.yaml"))
    p.add_argument("--data-dir", default=None, help="Override data directory")
    p.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    p.add_argument("--epochs", type=int, default=None, help="Epochs per trial (override config)")
    p.add_argument("--timeout", type=int, default=None, help="Timeout in seconds for entire study")
    p.add_argument("--study-name", default="climatevision_hpo")
    p.add_argument("--storage", default=None, help="Optuna storage URL (e.g. sqlite:///optuna.db)")
    return p.parse_args()


if __name__ == "__main__":
    main()
