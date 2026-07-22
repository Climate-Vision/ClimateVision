"""Tests for the ONNX-preferred inference path and PyTorch fallback (issue #12)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import climatevision.inference.pipeline as pipeline
from climatevision.inference.pipeline import (
    UNet,
    _ONNXSegmenter,
    _load_model,
    benchmark_inference,
    run_inference,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Each test starts with an empty model cache."""
    pipeline._model_cache.clear()
    yield
    pipeline._model_cache.clear()


def _fake_onnx_model(n_channels: int = 4, n_classes: int = 2):
    """A stand-in that quacks like an ONNX-backed segmenter without onnxruntime."""
    model = UNet(n_channels=n_channels, n_classes=n_classes)
    model.backend = "onnx"  # type: ignore[attr-defined]
    return model


# ---------------------------------------------------------------------------
# Backend selection priority
# ---------------------------------------------------------------------------

def test_onnx_is_preferred_when_present(monkeypatch):
    """When a servable ONNX model loads, _load_model must use it over the .pth."""
    monkeypatch.setattr(pipeline, "_find_onnx", lambda a: Path("models/unet_deforestation.onnx"))
    monkeypatch.setattr(pipeline, "_try_load_onnx", lambda p, c, k: _fake_onnx_model(c, k))
    # A .pth also exists — ONNX must still win.
    monkeypatch.setattr(pipeline, "_find_best_checkpoint", lambda a: Path("models/best_model.pth"))

    model, _ = _load_model("deforestation")
    assert getattr(model, "backend", "pytorch") == "onnx"


def test_falls_back_to_pytorch_when_no_onnx(monkeypatch):
    """No ONNX file present → PyTorch UNet backend."""
    monkeypatch.setattr(pipeline, "_find_onnx", lambda a: None)
    monkeypatch.setattr(pipeline, "_find_best_checkpoint", lambda a: None)

    model, _ = _load_model("deforestation")
    assert isinstance(model, UNet)
    assert getattr(model, "backend", "pytorch") == "pytorch"


def test_falls_back_when_onnx_unloadable(monkeypatch):
    """ONNX path present but unloadable (e.g. LFS pointer) → PyTorch fallback."""
    monkeypatch.setattr(pipeline, "_find_onnx", lambda a: Path("models/unet_deforestation.onnx"))
    monkeypatch.setattr(pipeline, "_try_load_onnx", lambda p, c, k: None)
    monkeypatch.setattr(pipeline, "_find_best_checkpoint", lambda a: None)

    model, _ = _load_model("deforestation")
    assert isinstance(model, UNet)
    assert getattr(model, "backend", "pytorch") == "pytorch"


# ---------------------------------------------------------------------------
# Latency / backend surfaced in the response
# ---------------------------------------------------------------------------

def test_run_inference_reports_backend_and_latency(monkeypatch):
    monkeypatch.setattr(pipeline, "_find_onnx", lambda a: None)
    monkeypatch.setattr(pipeline, "_find_best_checkpoint", lambda a: None)

    image = np.zeros((4, 64, 64), dtype=np.float32)
    result = run_inference(image, analysis_type="deforestation")
    inf = result["inference"]

    assert inf["backend"] == "pytorch"
    assert "inference_ms" in inf
    assert isinstance(inf["inference_ms"], float)
    assert inf["inference_ms"] >= 0.0


def test_onnx_segmenter_labels_backend():
    assert _ONNXSegmenter.backend == "onnx"
    assert getattr(UNet(n_channels=4, n_classes=2), "backend", "pytorch") == "pytorch"


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def test_benchmark_inference_without_onnx(monkeypatch):
    monkeypatch.setattr(pipeline, "_find_onnx", lambda a: None)

    result = benchmark_inference("deforestation", image_size=64, runs=2)
    assert result["backend_active"] == "pytorch"
    assert result["pytorch_ms"] > 0
    assert result["onnx_ms"] is None
    assert result["speedup"] is None


# ---------------------------------------------------------------------------
# End-to-end with a real ONNX export
# ---------------------------------------------------------------------------

def test_end_to_end_onnx_roundtrip(monkeypatch, tmp_path):
    """Export a small UNet to ONNX, then confirm the pipeline loads and serves it."""
    ort = pytest.importorskip("onnxruntime")
    import torch

    onnx_path = tmp_path / "unet_deforestation.onnx"
    model = UNet(n_channels=4, n_classes=2).eval()
    dummy = torch.zeros(1, 4, 64, 64)
    export_kwargs = dict(
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"image": {0: "batch", 2: "h", 3: "w"},
                      "logits": {0: "batch", 2: "h", 3: "w"}},
    )
    try:
        # dynamo=False forces the legacy TorchScript exporter, which does not
        # require the optional onnxscript dependency.
        try:
            torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **export_kwargs)
        except TypeError:  # older torch without the dynamo kwarg
            torch.onnx.export(model, dummy, str(onnx_path), **export_kwargs)
    except Exception as exc:  # optional export tooling (onnx/onnxscript) may be absent
        if "onnx" in str(exc).lower():
            pytest.skip(f"ONNX export tooling unavailable: {exc}")
        raise
    assert onnx_path.exists()

    monkeypatch.setattr(pipeline, "_find_onnx", lambda a: onnx_path)
    monkeypatch.setattr(pipeline, "_find_best_checkpoint", lambda a: None)

    loaded, _ = _load_model("deforestation")
    assert isinstance(loaded, _ONNXSegmenter)

    image = np.zeros((4, 64, 64), dtype=np.float32)
    result = run_inference(image, analysis_type="deforestation")
    assert result["inference"]["backend"] == "onnx"
    assert result["inference"]["image_size"] == [64, 64]


# ---------------------------------------------------------------------------
# Export discovery
# ---------------------------------------------------------------------------

def test_discover_analysis_checkpoints_covers_enabled_types():
    import importlib.util

    root = Path(pipeline.__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "export_model", root / "scripts" / "export_model.py"
    )
    export_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_model)

    targets = {t[0]: (t[1], t[2]) for t in export_model._discover_analysis_checkpoints()}
    assert "deforestation" in targets
    ckpt, n_classes = targets["deforestation"]
    assert n_classes == 2
    assert str(ckpt).endswith("unet_deforestation.pth")
