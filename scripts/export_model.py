"""
Export the trained ClimateVision model to ONNX for production serving.

Produces (per exported model ``<name>``):
  - <dir>/<name>.onnx              — standard ONNX graph
  - <dir>/<name>_quantized.onnx    — INT8 quantized (CPU-optimised)
  - <dir>/<name>_export_info.json  — metadata (opset, input shape, benchmark)

Usage:
  python scripts/export_model.py \\
      --checkpoint models/20240101_120000/best_model.pth

  # Export every enabled analysis type to models/unet_<type>.onnx,
  # reading each type's checkpoint and num_classes from config.yaml:
  python scripts/export_model.py --all

  # Override output path and input size:
  python scripts/export_model.py \\
      --checkpoint models/best_model.pth \\
      --out        models/production/model.onnx \\
      --image-size 512

  # Skip quantization (requires onnxruntime):
  python scripts/export_model.py --checkpoint ... --no-quantize
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, n_classes: int = 2) -> tuple[nn.Module, dict]:
    from climatevision.models.unet import get_model

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt.get("cfg", {})

    arch = cfg.get("model", {}).get("architecture", "attention_unet")
    state = ckpt.get("ema_state_dict") or ckpt.get("model_state_dict", ckpt)

    # Infer in_channels from weight shape
    in_ch = 4
    for key, val in state.items():
        if "inc" in key and "weight" in key and val.ndim == 4:
            in_ch = val.shape[1]
            break

    model = get_model(arch, n_channels=in_ch, n_classes=n_classes)
    model.load_state_dict(state, strict=False)
    model.eval()

    logger.info(
        "Loaded %s  (in_channels=%d, n_classes=%d)  from epoch %d  val_iou=%.4f",
        arch,
        in_ch,
        n_classes,
        ckpt.get("epoch", 0),
        ckpt.get("val_iou", 0.0),
    )
    return model, cfg


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(
    model: nn.Module,
    onnx_path: Path,
    image_size: int,
    in_channels: int,
    opset: int = 17,
) -> None:
    dummy = torch.zeros(1, in_channels, image_size, image_size)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image":  {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
    )
    size_mb = onnx_path.stat().st_size / 1e6
    logger.info("ONNX model saved: %s  (%.1f MB)", onnx_path, size_mb)


# ---------------------------------------------------------------------------
# ONNX validation
# ---------------------------------------------------------------------------

def validate_onnx(
    onnx_path: Path, in_channels: int, image_size: int, n_classes: int = 2
) -> float:
    """
    Run a forward pass with onnxruntime and return inference latency (ms).

    Also verifies the output tensor signature is ``(N, n_classes, H, W)`` so a
    shape regression (wrong channel count, missing dynamic axis) is caught at
    export time rather than at serve time.
    """
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        logger.warning("onnxruntime not installed — skipping validation. "
                       "Run: pip install onnxruntime")
        return -1.0

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    dummy = np.random.rand(1, in_channels, image_size, image_size).astype(np.float32)

    # Verify output signature matches (N, n_classes, H, W)
    out = sess.run(None, {"image": dummy})[0]
    expected = (1, n_classes, image_size, image_size)
    if tuple(out.shape) != expected:
        raise ValueError(
            f"ONNX output shape {tuple(out.shape)} != expected {expected} "
            f"for {onnx_path}"
        )

    # Warm-up
    for _ in range(3):
        sess.run(None, {"image": dummy})

    # Benchmark
    N = 20
    t0 = time.perf_counter()
    for _ in range(N):
        sess.run(None, {"image": dummy})
    latency_ms = (time.perf_counter() - t0) / N * 1000

    logger.info("ONNX validation OK  |  avg latency: %.1f ms  (batch=1, %dx%d)",
                latency_ms, image_size, image_size)
    return latency_ms


# ---------------------------------------------------------------------------
# INT8 quantization
# ---------------------------------------------------------------------------

def quantize_onnx(onnx_path: Path, out_path: Path) -> None:
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        logger.warning("onnxruntime quantization not available — skipping. "
                       "Run: pip install onnxruntime")
        return

    quantize_dynamic(
        str(onnx_path),
        str(out_path),
        weight_type=QuantType.QInt8,
    )
    size_mb = out_path.stat().st_size / 1e6
    logger.info("INT8 quantized model: %s  (%.1f MB)", out_path, size_mb)


# ---------------------------------------------------------------------------
# PyTorch benchmark helper
# ---------------------------------------------------------------------------

def benchmark_pytorch(model: nn.Module, in_channels: int, image_size: int) -> float:
    device = torch.device("cpu")
    dummy = torch.zeros(1, in_channels, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(dummy)
        N = 20
        t0 = time.perf_counter()
        for _ in range(N):
            model(dummy)
    return (time.perf_counter() - t0) / N * 1000


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def export_checkpoint(
    ckpt_path: Path,
    onnx_path: Path,
    image_size: int,
    opset: int,
    no_quantize: bool,
    n_classes: int = 2,
) -> dict:
    """Export a single ``.pth`` checkpoint to ONNX and return export metadata."""
    model, cfg = load_model(ckpt_path, n_classes=n_classes)

    in_channels = getattr(model, "n_channels", cfg.get("model", {}).get("in_channels", 4))
    quantized_path = onnx_path.parent / f"{onnx_path.stem}_quantized.onnx"

    # PyTorch baseline latency
    pt_ms = benchmark_pytorch(model, in_channels, image_size)
    logger.info("PyTorch (CPU) baseline: %.1f ms", pt_ms)

    export_onnx(
        model=model,
        onnx_path=onnx_path,
        image_size=image_size,
        in_channels=in_channels,
        opset=opset,
    )

    onnx_ms = validate_onnx(onnx_path, in_channels, image_size, n_classes)

    q_ms = -1.0
    if not no_quantize:
        quantize_onnx(onnx_path, quantized_path)
        if quantized_path.exists():
            q_ms = validate_onnx(quantized_path, in_channels, image_size, n_classes)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    info = {
        "checkpoint":        str(ckpt_path),
        "architecture":      cfg.get("model", {}).get("architecture", "unknown"),
        "in_channels":       in_channels,
        "num_classes":       n_classes,
        "image_size":        image_size,
        "onnx_opset":        opset,
        "onnx_path":         str(onnx_path),
        "quantized_path":    str(quantized_path) if not no_quantize else None,
        "val_iou":           ckpt.get("val_iou", None),
        "val_f1":            ckpt.get("val_f1", None),
        "epoch":             ckpt.get("epoch", None),
        "benchmark_ms": {
            "pytorch_cpu":   round(pt_ms, 2),
            "onnx_cpu":      round(onnx_ms, 2) if onnx_ms > 0 else None,
            "onnx_int8_cpu": round(q_ms, 2) if q_ms > 0 else None,
        },
    }
    info_path = onnx_path.parent / f"{onnx_path.stem}_export_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info("Export metadata saved to %s", info_path)

    # Summary
    print("\n" + "=" * 55)
    print(f"  Export Summary — {onnx_path.name}")
    print("=" * 55)
    print(f"  ONNX model   : {onnx_path}")
    if not no_quantize and quantized_path.exists():
        print(f"  INT8 model   : {quantized_path}")
    print(f"  Val IoU      : {info['val_iou']:.4f}" if info["val_iou"] else "  Val IoU      : N/A")
    print(f"  PyTorch (CPU): {pt_ms:.1f} ms")
    if onnx_ms > 0:
        print(f"  ONNX (CPU)   : {onnx_ms:.1f} ms   ({pt_ms / onnx_ms:.1f}× speedup)")
    if q_ms > 0:
        print(f"  INT8 (CPU)   : {q_ms:.1f} ms   ({pt_ms / q_ms:.1f}× speedup)")
    print("=" * 55)
    return info


def _discover_analysis_checkpoints() -> list[tuple[str, Path, int]]:
    """
    Resolve (analysis_type, checkpoint_path, num_classes) for every enabled
    analysis type that declares neural ``weights`` in config.yaml.
    """
    from climatevision.data.band_mapping import (
        get_model_config,
        list_enabled_analysis_types,
    )

    found: list[tuple[str, Path, int]] = []
    for analysis_type in list_enabled_analysis_types():
        model_cfg = get_model_config(analysis_type)
        weights = model_cfg.get("weights")
        if not weights:  # e.g. SAR ensemble — no neural checkpoint to export
            continue
        ckpt = PROJECT_ROOT / weights
        n_classes = model_cfg.get("num_classes", 2)
        found.append((analysis_type, ckpt, n_classes))
    return found


def _export_all(args: argparse.Namespace) -> None:
    """Export every enabled analysis type's checkpoint to ``models/unet_<type>.onnx``."""
    targets = _discover_analysis_checkpoints()
    if not targets:
        logger.error("No analysis types with neural weights found in config.yaml")
        sys.exit(1)

    exported, skipped = 0, 0
    for analysis_type, ckpt_path, n_classes in targets:
        if not ckpt_path.exists():
            logger.warning("Skipping %s — checkpoint not found: %s", analysis_type, ckpt_path)
            skipped += 1
            continue
        onnx_path = PROJECT_ROOT / "models" / f"unet_{analysis_type}.onnx"
        logger.info("Exporting %s → %s", analysis_type, onnx_path)
        export_checkpoint(
            ckpt_path=ckpt_path,
            onnx_path=onnx_path,
            image_size=args.image_size,
            opset=args.opset,
            no_quantize=args.no_quantize,
            n_classes=n_classes,
        )
        exported += 1

    logger.info("Batch export complete — %d exported, %d skipped", exported, skipped)
    if exported == 0:
        sys.exit(1)


def main() -> None:
    args = parse_args()

    if args.all:
        _export_all(args)
        return

    if not args.checkpoint:
        logger.error("Provide --checkpoint <path> or --all")
        sys.exit(1)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    run_dir = ckpt_path.parent
    onnx_path = Path(args.out) if args.out else run_dir / "model.onnx"

    export_checkpoint(
        ckpt_path=ckpt_path,
        onnx_path=onnx_path,
        image_size=args.image_size,
        opset=args.opset,
        no_quantize=args.no_quantize,
    )
    print()
    print("Serve with:")
    print(f"  onnxruntime  →  sess = ort.InferenceSession('{onnx_path}')")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export ClimateVision model to ONNX")
    p.add_argument("--checkpoint",   default=None, help="Path to best_model.pth")
    p.add_argument("--all",          action="store_true",
                   help="Export every enabled analysis type to models/unet_<type>.onnx")
    p.add_argument("--out",          default=None,
                   help="ONNX output path (default: <checkpoint_dir>/model.onnx)")
    p.add_argument("--image-size",   type=int, default=256,
                   help="Spatial size used for export (any size works at inference via dynamic axes)")
    p.add_argument("--opset",        type=int, default=17, help="ONNX opset version")
    p.add_argument("--no-quantize",  action="store_true", help="Skip INT8 quantization")
    return p.parse_args()


if __name__ == "__main__":
    main()
