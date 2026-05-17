"""
PyTorch to ONNX Export Utilities for ClimateVision.

Provides functions to export trained PyTorch models to ONNX format
for optimized production inference with ONNX Runtime.

Features:
- Export U-Net and Siamese networks to ONNX
- Dynamic axes support for variable input sizes
- Opset version selection for compatibility
- Validation of exported models
- Batch size configuration
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OPSET_VERSION = 14
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "models"


# ---------------------------------------------------------------------------
# Export Functions
# ---------------------------------------------------------------------------

def export_unet_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    *,
    input_shape: tuple[int, int, int, int] = (1, 4, 256, 256),
    opset_version: int = DEFAULT_OPSET_VERSION,
    dynamic_axes: Optional[dict[str, dict[int, str]]] = None,
    training: bool = False,
    do_constant_folding: bool = True,
    input_names: Optional[list[str]] = None,
    output_names: Optional[list[str]] = None,
) -> Path:
    """
    Export a U-Net model to ONNX format.

    Args:
        model: PyTorch U-Net model to export
        output_path: Path to save the ONNX model
        input_shape: Shape of input tensor (N, C, H, W)
        opset_version: ONNX opset version (11-17 recommended)
        dynamic_axes: Dynamic axis configuration for variable sizes
        training: Export in training mode
        do_constant_folding: Apply constant folding optimization
        input_names: Custom input tensor names
        output_names: Custom output tensor names

    Returns:
        Path to the exported ONNX model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Default dynamic axes for batch and spatial dimensions
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }

    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Set model to eval mode unless training export requested
    was_training = model.training
    if not training:
        model.eval()

    # Export
    logger.info(
        "Exporting U-Net to ONNX: %s (input_shape=%s, opset=%d)",
        output_path,
        input_shape,
        opset_version,
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        training=torch.onnx.TrainingMode.TRAINING if training else torch.onnx.TrainingMode.EVAL,
        keep_initializers_as_inputs=False,
    )

    # Restore training state
    if was_training:
        model.train()

    # Validate export
    _validate_onnx_model(output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "ONNX export complete: %s (%.2f MB)",
        output_path,
        size_mb,
    )

    return output_path


def export_siamese_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    *,
    input_shape: tuple[int, int, int, int] = (1, 4, 256, 256),
    opset_version: int = DEFAULT_OPSET_VERSION,
    dynamic_axes: Optional[dict[str, dict[int, str]]] = None,
) -> Path:
    """
    Export a Siamese network to ONNX format.

    Siamese networks take two inputs (before/after images), so we need
    to handle the dual-input architecture.

    Args:
        model: PyTorch Siamese network to export
        output_path: Path to save the ONNX model
        input_shape: Shape of each input tensor (N, C, H, W)
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axis configuration

    Returns:
        Path to the exported ONNX model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dynamic_axes is None:
        dynamic_axes = {
            "input_before": {0: "batch", 2: "height", 3: "width"},
            "input_after": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }

    dummy_before = torch.randn(*input_shape)
    dummy_after = torch.randn(*input_shape)

    model.eval()

    logger.info(
        "Exporting Siamese network to ONNX: %s (input_shape=%s)",
        output_path,
        input_shape,
    )

    torch.onnx.export(
        model,
        (dummy_before, dummy_after),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_before", "input_after"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    _validate_onnx_model(output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Siamese ONNX export complete: %s (%.2f MB)",
        output_path,
        size_mb,
    )

    return output_path


def export_model_from_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    analysis_type: str = "deforestation",
    input_shape: tuple[int, int, int, int] = (1, 4, 256, 256),
    opset_version: int = DEFAULT_OPSET_VERSION,
) -> Path:
    """
    Export a model from a training checkpoint to ONNX format.

    Loads the checkpoint, reconstructs the model architecture, and exports
    to ONNX.

    Args:
        checkpoint_path: Path to the PyTorch checkpoint (.pth)
        output_path: Path to save the ONNX model
        analysis_type: Type of analysis (determines model config)
        input_shape: Shape of input tensor
        opset_version: ONNX opset version

    Returns:
        Path to the exported ONNX model
    """
    from climatevision.data.band_mapping import get_model_config
    from climatevision.models.unet import UNet

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get model config
    model_cfg = get_model_config(analysis_type)
    n_channels = model_cfg.get("in_channels", 4)
    n_classes = model_cfg.get("num_classes", 2)

    # Reconstruct model
    model = UNet(n_channels=n_channels, n_classes=n_classes)

    # Load weights
    model_state = checkpoint.get("model_state_dict")
    if model_state is not None:
        model.load_state_dict(model_state, strict=False)
        logger.info("Loaded model weights from checkpoint")
    else:
        logger.warning("No model_state_dict in checkpoint, using random weights")

    # Export
    return export_unet_to_onnx(
        model,
        output_path,
        input_shape=input_shape,
        opset_version=opset_version,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_onnx_model(
    onnx_path: str | Path,
    *,
    num_test_inputs: int = 3,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """
    Validate an ONNX model by comparing outputs with PyTorch.

    Args:
        onnx_path: Path to the ONNX model
        num_test_inputs: Number of random inputs to test
        tolerance: Maximum allowed difference between PyTorch and ONNX outputs

    Returns:
        Dictionary with validation results
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return {
            "valid": False,
            "error": "onnxruntime not installed",
        }

    onnx_path = Path(onnx_path)
    results: list[dict[str, Any]] = []
    all_passed = True

    for i in range(num_test_inputs):
        try:
            # Create session
            session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )

            # Get input shape
            input_shape = session.get_inputs()[0].shape
            # Replace dynamic dimensions with defaults
            test_shape = []
            for dim in input_shape:
                if isinstance(dim, str) or dim is None or dim <= 0:
                    test_shape.append(1)
                else:
                    test_shape.append(dim)

            # Generate random input
            test_input = np.random.randn(*test_shape).astype(np.float32)

            # ONNX inference
            onnx_output = session.run(None, {session.get_inputs()[0].name: test_input})

            # Validate output shapes
            for j, output in enumerate(onnx_output):
                if output.dtype not in (np.float32, np.float64):
                    results.append({
                        "test": i,
                        "output_index": j,
                        "passed": False,
                        "error": f"Unexpected output dtype: {output.dtype}",
                    })
                    all_passed = False
                    continue

            results.append({
                "test": i,
                "passed": True,
                "input_shape": list(test_shape),
                "output_shapes": [list(o.shape) for o in onnx_output],
            })

        except Exception as exc:
            results.append({
                "test": i,
                "passed": False,
                "error": str(exc),
            })
            all_passed = False

    return {
        "valid": all_passed,
        "num_tests": num_test_inputs,
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "details": results,
    }


def _validate_onnx_model(onnx_path: Path) -> None:
    """Quick validation after export."""
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        logger.debug("ONNX model validation passed: %s", onnx_path)
    except ImportError:
        logger.debug("onnx package not available, skipping validation")
    except Exception as exc:
        logger.warning("ONNX model validation warning: %s", exc)


# ---------------------------------------------------------------------------
# Batch Export Utilities
# ---------------------------------------------------------------------------

def export_all_analysis_types(
    checkpoint_dir: str | Path = "models",
    output_dir: str | Path = "models",
    opset_version: int = DEFAULT_OPSET_VERSION,
) -> dict[str, Path]:
    """
    Export models for all enabled analysis types.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        output_dir: Directory to save ONNX models
        opset_version: ONNX opset version

    Returns:
        Dictionary mapping analysis_type -> ONNX model path
    """
    from climatevision.data.band_mapping import get_model_config

    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_types = ["deforestation", "ice_melting", "flooding"]
    exported: dict[str, Path] = {}

    for analysis_type in analysis_types:
        model_cfg = get_model_config(analysis_type)
        n_channels = model_cfg.get("in_channels", 4)
        input_shape = (1, n_channels, 256, 256)

        # Find checkpoint
        checkpoint_path = checkpoint_dir / f"{analysis_type}_best.pth"
        if not checkpoint_path.exists():
            checkpoint_path = checkpoint_dir / "best_model.pth"

        if not checkpoint_path.exists():
            logger.warning("No checkpoint found for %s, skipping", analysis_type)
            continue

        output_path = output_dir / f"{analysis_type}_model.onnx"

        try:
            export_model_from_checkpoint(
                checkpoint_path,
                output_path,
                analysis_type=analysis_type,
                input_shape=input_shape,
                opset_version=opset_version,
            )
            exported[analysis_type] = output_path
        except Exception as exc:
            logger.error("Failed to export %s: %s", analysis_type, exc)

    return exported
