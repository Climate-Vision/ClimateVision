"""
Export the trained ClimateVision model to ONNX for production serving.

Produces:
  - <run_dir>/model.onnx           — standard ONNX graph
  - <run_dir>/model_quantized.onnx — INT8 quantized (CPU-optimised)
  - <run_dir>/export_info.json     — metadata (opset, input shape, benchmark)

Usage:
  python scripts/export_model.py \\
      --checkpoint models/20240101_120000/best_model.pth

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

def _load_run_config(ckpt_path: Path) -> dict:
    """Load the full training config from the run directory."""
    run_dir = ckpt_path.parent
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


def load_model(ckpt_path: Path) -> tuple[nn.Module, dict]:
    from climatevision.models.unet import get_model
    from climatevision.models.flood_unet import build_flood_model

    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Trainer cfg is in ckpt['cfg']; full model cfg is in config.yaml next to checkpoint
    cfg = _load_run_config(ckpt_path)
    if not cfg:
        cfg = ckpt.get("cfg", {})

    arch = cfg.get("model", {}).get("architecture", "attention_unet")
    state = ckpt.get("ema_state_dict") or ckpt.get("model_state_dict", ckpt)

    # Infer in_channels and n_classes from weight shape
    in_ch = cfg.get("model", {}).get("in_channels", 4)
    n_classes = cfg.get("model", {}).get("num_classes", 2)
    for key, val in state.items():
        if val.ndim == 4:
            if val.shape[1] in (3, 4, 5) and "encoder" not in key and "down" not in key:
                # first conv layer — input channels
                in_ch = val.shape[1]
            if val.shape[0] in (2, 3) and ("outc" in key or "segmentation_head" in key or "classifier" in key):
                # final conv layer — output classes
                n_classes = val.shape[0]

    # Flood models use smp-based architectures
    if arch in ("flood_unet", "flood_unet_s2only"):
        use_sar = in_ch == 5
        model = build_flood_model(
            use_sar=use_sar,
            encoder_name=cfg.get("model", {}).get("encoder", "efficientnet-b7"),
        )
    else:
        model = get_model(arch, n_channels=in_ch, n_classes=n_classes)

    model.load_state_dict(state, strict=False)
    model.eval()

    logger.info(
        "Loaded %s  (in_channels=%d, classes=%d)  from epoch %d  val_iou=%.4f",
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

def validate_onnx(onnx_path: Path, in_channels: int, image_size: int) -> float:
    """Run a forward pass with onnxruntime and return inference latency (ms)."""
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

def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    model, cfg = load_model(ckpt_path)

    in_channels = cfg.get("model", {}).get("in_channels", 4)
    image_size  = args.image_size

    # Determine output paths
    run_dir = ckpt_path.parent
    onnx_path = Path(args.out) if args.out else run_dir / "model.onnx"
    quantized_path = onnx_path.parent / "model_quantized.onnx"

    # PyTorch baseline latency
    pt_ms = benchmark_pytorch(model, in_channels, image_size)
    logger.info("PyTorch (CPU) baseline: %.1f ms", pt_ms)

    # Export
    export_onnx(
        model=model,
        onnx_path=onnx_path,
        image_size=image_size,
        in_channels=in_channels,
        opset=args.opset,
    )

    # Validate
    onnx_ms = validate_onnx(onnx_path, in_channels, image_size)

    # Quantize
    q_ms = -1.0
    if not args.no_quantize:
        quantize_onnx(onnx_path, quantized_path)
        if quantized_path.exists():
            q_ms = validate_onnx(quantized_path, in_channels, image_size)

    # Export metadata
    ckpt = torch.load(ckpt_path, map_location="cpu")
    info = {
        "checkpoint":        str(ckpt_path),
        "architecture":      cfg.get("model", {}).get("architecture", "unknown"),
        "in_channels":       in_channels,
        "num_classes":       cfg.get("model", {}).get("num_classes", 2),
        "image_size":        image_size,
        "onnx_opset":        args.opset,
        "onnx_path":         str(onnx_path),
        "quantized_path":    str(quantized_path) if not args.no_quantize else None,
        "val_iou":           ckpt.get("val_iou", None),
        "val_f1":            ckpt.get("val_f1", None),
        "epoch":             ckpt.get("epoch", None),
        "benchmark_ms": {
            "pytorch_cpu":   round(pt_ms, 2),
            "onnx_cpu":      round(onnx_ms, 2) if onnx_ms > 0 else None,
            "onnx_int8_cpu": round(q_ms, 2) if q_ms > 0 else None,
        },
    }
    info_path = onnx_path.parent / "export_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info("Export metadata saved to %s", info_path)

    # Summary
    print("\n" + "=" * 55)
    print("  Export Summary")
    print("=" * 55)
    print(f"  ONNX model   : {onnx_path}")
    if not args.no_quantize and quantized_path.exists():
        print(f"  INT8 model   : {quantized_path}")
    print(f"  Val IoU      : {info['val_iou']:.4f}" if info["val_iou"] else "  Val IoU      : N/A")
    print(f"  PyTorch (CPU): {pt_ms:.1f} ms")
    if onnx_ms > 0:
        print(f"  ONNX (CPU)   : {onnx_ms:.1f} ms   ({pt_ms / onnx_ms:.1f}× speedup)")
    if q_ms > 0:
        print(f"  INT8 (CPU)   : {q_ms:.1f} ms   ({pt_ms / q_ms:.1f}× speedup)")
    print("=" * 55)
    print()
    print("Serve with:")
    print(f"  onnxruntime  →  sess = ort.InferenceSession('{onnx_path}')")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export ClimateVision model to ONNX")
    p.add_argument("--checkpoint",   required=True, help="Path to best_model.pth")
    p.add_argument("--out",          default=None,
                   help="ONNX output path (default: <checkpoint_dir>/model.onnx)")
    p.add_argument("--image-size",   type=int, default=256,
                   help="Spatial size used for export (any size works at inference via dynamic axes)")
    p.add_argument("--opset",        type=int, default=17, help="ONNX opset version")
    p.add_argument("--no-quantize",  action="store_true", help="Skip INT8 quantization")
    return p.parse_args()


if __name__ == "__main__":
    main()
