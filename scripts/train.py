"""
Production training entry-point for ClimateVision forest segmentation.

Usage:
  # Train with defaults (generates synthetic data if none exists):
  python scripts/train.py

  # Custom config:
  python scripts/train.py --config config/train.yaml

  # Override specific keys:
  python scripts/train.py --config config/train.yaml \\
      --data-dir data/processed \\
      --epochs 50 \\
      --batch-size 8

  # Resume from a checkpoint:
  python scripts/train.py --resume models/my_run/checkpoint_epoch_0030.pth
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project root on the Python path so `climatevision` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict:
    try:
        import yaml  # PyYAML
    except ImportError:
        logger.error("PyYAML not installed. Run: pip install pyyaml")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def build_config(args: argparse.Namespace) -> dict:
    """Load YAML config and apply CLI overrides."""
    cfg: dict = {}

    if args.config and Path(args.config).exists():
        cfg = _load_yaml(args.config)
        logger.info("Config loaded from %s", args.config)
    else:
        logger.info("No config file — using defaults")

    # CLI overrides (only non-None values)
    overrides: dict = {}
    if args.data_dir:
        overrides.setdefault("data", {})["dir"] = args.data_dir
    if args.epochs:
        overrides.setdefault("schedule", {})["epochs"] = args.epochs
    if args.batch_size:
        overrides.setdefault("data", {})["batch_size"] = args.batch_size
    if args.lr:
        overrides.setdefault("optimizer", {})["learning_rate"] = args.lr
    if args.save_dir:
        overrides.setdefault("output", {})["save_dir"] = args.save_dir
    if args.run_name:
        overrides.setdefault("output", {})["run_name"] = args.run_name
    if args.no_amp:
        overrides.setdefault("training", {})["mixed_precision"] = False
    if args.num_workers is not None:
        overrides.setdefault("data", {})["num_workers"] = args.num_workers
    if args.image_size is not None:
        overrides.setdefault("data", {})["image_size"] = args.image_size
    if args.arch:
        overrides.setdefault("model", {})["architecture"] = args.arch

    cfg = _deep_merge(cfg, overrides)

    # Defaults for any missing keys
    cfg.setdefault("data", {})
    cfg["data"].setdefault("dir",                  "data/processed")
    cfg["data"].setdefault("image_size",            256)
    cfg["data"].setdefault("batch_size",             16)
    cfg["data"].setdefault("num_workers",             4)
    cfg["data"].setdefault("use_weighted_sampler", True)
    cfg["data"].setdefault("pin_memory",           True)

    cfg.setdefault("model", {})
    cfg["model"].setdefault("architecture", "attention_unet")
    cfg["model"].setdefault("in_channels",            4)
    cfg["model"].setdefault("num_classes",             2)
    cfg["model"].setdefault("bilinear",            True)

    cfg.setdefault("loss", {})
    cfg["loss"].setdefault("type",              "combined")
    cfg["loss"].setdefault("focal_weight",          0.5)
    cfg["loss"].setdefault("focal_alpha",           0.25)
    cfg["loss"].setdefault("focal_gamma",           2.0)
    cfg["loss"].setdefault("use_class_weights",    True)

    cfg.setdefault("optimizer", {})
    cfg["optimizer"].setdefault("learning_rate",  1e-4)
    cfg["optimizer"].setdefault("weight_decay",   1e-4)
    cfg["optimizer"].setdefault("min_lr",         1e-6)

    cfg.setdefault("schedule", {})
    cfg["schedule"].setdefault("epochs",                100)
    cfg["schedule"].setdefault("warmup_epochs",           5)
    cfg["schedule"].setdefault("checkpoint_interval",    10)

    cfg.setdefault("training", {})
    cfg["training"].setdefault("mixed_precision",      True)
    cfg["training"].setdefault("grad_clip",            1.0)
    cfg["training"].setdefault("use_ema",             True)
    cfg["training"].setdefault("ema_decay",          0.9999)
    cfg["training"].setdefault("early_stopping_patience", 15)

    cfg.setdefault("output", {})
    cfg["output"].setdefault("save_dir",  "models")
    cfg["output"].setdefault("run_name",  "")
    cfg.setdefault("normalizer_stats", "")

    return cfg


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict):
    """Instantiate the segmentation model from config."""
    from climatevision.models.unet import get_model
    mcfg = cfg["model"]
    arch = mcfg["architecture"]

    kwargs = {
        "n_channels": mcfg["in_channels"],
        "n_classes":  mcfg["num_classes"],
    }
    if arch == "unet":
        kwargs["bilinear"] = mcfg.get("bilinear", True)
    elif arch == "resnet_unet":
        kwargs["backbone"] = mcfg.get("backbone", "resnet34")
        kwargs["pretrained"] = mcfg.get("pretrained", True)

    model = get_model(arch, **kwargs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s  (%.0f trainable parameters)", arch, n_params)
    return model


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

def build_criterion(cfg: dict, class_weights=None):
    from climatevision.training.losses import (
        CombinedLoss, FocalLoss, DiceLoss, LovaszSoftmaxLoss,
    )
    import torch

    lcfg = cfg["loss"]
    loss_type = lcfg["type"]

    cw = class_weights if lcfg.get("use_class_weights") else None

    if loss_type == "combined":
        return CombinedLoss(
            focal_weight=lcfg["focal_weight"],
            focal_alpha=lcfg["focal_alpha"],
            focal_gamma=lcfg["focal_gamma"],
            class_weights=cw,
        )
    if loss_type == "focal":
        return FocalLoss(
            alpha=lcfg["focal_alpha"],
            gamma=lcfg["focal_gamma"],
            class_weights=cw,
        )
    if loss_type == "dice":
        return DiceLoss()
    if loss_type == "lovasz":
        return LovaszSoftmaxLoss()
    raise ValueError(f"Unknown loss type: {loss_type}")


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

def load_normalizer(cfg: dict):
    stats_path = cfg.get("normalizer_stats", "")
    if not stats_path:
        return None
    from climatevision.data.preprocessing import Sentinel2Normalizer  # type: ignore[import-not-found]
    norm = Sentinel2Normalizer()
    try:
        norm.load(stats_path)
        logger.info("Normalizer loaded from %s", stats_path)
    except Exception as exc:
        logger.warning("Could not load normalizer (%s) — using built-in defaults", exc)
    return norm


# ---------------------------------------------------------------------------
# Checkpoint resume
# ---------------------------------------------------------------------------

def maybe_resume(model, optimizer, resume_path: str | None) -> int:
    """Load weights from checkpoint. Returns start epoch (0 if no resume)."""
    if not resume_path:
        return 0
    import torch
    path = Path(resume_path)
    if not path.exists():
        logger.warning("Checkpoint %s not found — starting from scratch", path)
        return 0
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    if "optimizer_state_dict" in ckpt and optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt.get("epoch", 0)
    logger.info("Resumed from %s (epoch %d)", path.name, start_epoch)
    return start_epoch


# ---------------------------------------------------------------------------
# Auto-generate data if directory is empty
# ---------------------------------------------------------------------------

def maybe_generate_data(data_dir: Path, patch_size: int = 256, n_patches: int = 1000) -> None:
    train_img = data_dir / "train" / "images"
    if train_img.exists() and any(train_img.glob("*.tif")):
        return

    logger.warning("No training data found in %s", data_dir)
    logger.info("Auto-generating %d synthetic patches…", n_patches)

    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "prepare_data.py"),
        "--mode", "synthetic",
        "--n-patches", str(n_patches),
        "--patch-size", str(patch_size),
        "--out", str(data_dir),
        "--fit-normalizer",
    ]
    import subprocess
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("Data generation failed — check prepare_data.py output")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = build_config(args)

    # Run name / output directory
    run_name = cfg["output"]["run_name"] or datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(cfg["output"]["save_dir"]) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Persist effective config
    try:
        import yaml
        with open(save_dir / "config.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
    except ImportError:
        import json
        with open(save_dir / "config.json", "w") as f:
            import json
            json.dump(cfg, f, indent=2)

    logger.info("Run: %s  →  %s", run_name, save_dir)

    # Data
    data_dir   = Path(cfg["data"]["dir"])
    image_size = cfg["data"]["image_size"]
    maybe_generate_data(data_dir, patch_size=image_size)

    normalizer = load_normalizer(cfg)

    from climatevision.data.dataset import create_dataloaders  # type: ignore[import-not-found]
    loaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        image_size=image_size,
        normalizer=normalizer,
        pin_memory=cfg["data"]["pin_memory"],
        use_weighted_sampler=cfg["data"]["use_weighted_sampler"],
    )

    if "train" not in loaders:
        logger.error("No training split found in %s", data_dir)
        sys.exit(1)

    logger.info(
        "Splits — train: %d  val: %d  test: %d",
        len(loaders["train"].dataset),
        len(loaders.get("val", loaders["train"]).dataset),
        len(loaders["test"].dataset) if "test" in loaders else 0,
    )

    # Class weights
    class_weights = None
    if cfg["loss"]["use_class_weights"]:
        class_weights = loaders["train"].dataset.compute_class_weights()
        logger.info("Class weights: %s", class_weights.tolist())

    # Model + loss
    model     = build_model(cfg)
    criterion = build_criterion(cfg, class_weights=class_weights)

    # Trainer config dict (flat, as Trainer expects)
    trainer_cfg = {
        "learning_rate":            cfg["optimizer"]["learning_rate"],
        "weight_decay":             cfg["optimizer"]["weight_decay"],
        "min_lr":                   cfg["optimizer"]["min_lr"],
        "epochs":                   cfg["schedule"]["epochs"],
        "warmup_epochs":            cfg["schedule"]["warmup_epochs"],
        "checkpoint_interval":      cfg["schedule"]["checkpoint_interval"],
        "mixed_precision":          cfg["training"]["mixed_precision"],
        "grad_clip":                cfg["training"]["grad_clip"],
        "gradient_accumulation_steps": cfg["training"].get("gradient_accumulation_steps", 1),
        "use_ema":                  cfg["training"]["use_ema"],
        "ema_decay":                cfg["training"]["ema_decay"],
        "early_stopping_patience":  cfg["training"]["early_stopping_patience"],
    }

    # Build callbacks
    callbacks = []
    tracking_cfg = cfg.get("tracking", {})
    if tracking_cfg.get("use_mlflow", False):
        from climatevision.training.callbacks import MLflowLogger
        callbacks.append(MLflowLogger(
            experiment_name=tracking_cfg.get("mlflow_experiment", "climatevision"),
            run_name=run_name,
            tracking_uri=tracking_cfg.get("mlflow_tracking_uri") or None,
        ))

    from climatevision.training.callbacks import EpochTimer
    callbacks.append(EpochTimer())

    from climatevision.training.trainer import Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        loaders=loaders,
        cfg=trainer_cfg,
        save_dir=save_dir,
        callbacks=callbacks,
    )

    # Optional resume
    if args.resume:
        maybe_resume(model, trainer.optimizer, args.resume)

    t_start = time.time()
    history = trainer.fit()
    elapsed = time.time() - t_start

    best_iou = max((e.get("iou_forest", 0) for e in history["val"]), default=0)
    best_f1  = max((e.get("f1",         0) for e in history["val"]), default=0)

    logger.info("=" * 60)
    logger.info("Training complete in %.1f min", elapsed / 60)
    logger.info("Best val IoU: %.4f   F1: %.4f", best_iou, best_f1)
    logger.info("Weights saved to:  %s/best_model.pth", save_dir)
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  Evaluate:  python scripts/evaluate.py --checkpoint %s/best_model.pth --data-dir %s",
                save_dir, data_dir)
    logger.info("  Export:    python scripts/export_model.py --checkpoint %s/best_model.pth",
                save_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ClimateVision forest segmentation model")
    p.add_argument("--config",     default=str(PROJECT_ROOT / "config" / "train.yaml"),
                   help="Path to YAML config file")
    p.add_argument("--data-dir",   default=None, help="Override data.dir")
    p.add_argument("--epochs",     type=int, default=None, help="Override schedule.epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override data.batch_size")
    p.add_argument("--lr",         type=float, default=None, help="Override optimizer.learning_rate")
    p.add_argument("--save-dir",   default=None, help="Override output.save_dir")
    p.add_argument("--run-name",   default=None, help="Override output.run_name")
    p.add_argument("--resume",     default=None, help="Path to checkpoint to resume from")
    p.add_argument("--arch",       choices=["unet", "attention_unet", "resnet_unet"], default=None)
    p.add_argument("--no-amp",      action="store_true", help="Disable mixed-precision (AMP)")
    p.add_argument("--num-workers",  type=int, default=None, help="DataLoader worker count (0=main process)")
    p.add_argument("--image-size",   type=int, default=None, help="Spatial crop size in pixels")
    return p.parse_args()


if __name__ == "__main__":
    main()
