# ClimateVision ONNX Runtime Inference Implementation

## Issue #12: ONNX Runtime Inference Path

This implementation adds ONNX Runtime as a high-performance inference backend for ClimateVision,
with automatic fallback to PyTorch when ONNX models are unavailable.

### Files

| File | Description |
|------|-------------|
| `inference/onnx_runtime.py` | ONNX Runtime inference engine with session caching, benchmarking, and fallback |
| `inference/onnx_export.py` | PyTorch-to-ONNX export utilities for U-Net and Siamese networks |
| `inference/__init__.py` | Updated module exports (replaces existing `__init__.py`) |
| `tests/test_onnx_runtime.py` | Comprehensive unit tests (40+ test cases) |

### Features

1. **ONNX Runtime Session Management**
   - Automatic device selection (CPU/CUDA/MPS)
   - Per-model session caching for repeated inference
   - Configurable execution providers and session options

2. **High-Performance Inference**
   - Batch inference support
   - Latency benchmarking with percentile statistics (p50/p95/p99)
   - Throughput measurement (frames per second)

3. **PyTorch-to-ONNX Export**
   - Export U-Net and Siamese networks to ONNX format
   - Dynamic axes support for variable input sizes
   - Configurable opset versions (11-17)
   - Automatic model validation after export

4. **Graceful Fallback**
   - Automatic ONNX → PyTorch fallback
   - Clear logging of which engine is used
   - No code changes needed for existing PyTorch inference

### Usage

```python
# Export a trained model to ONNX
from climatevision.inference import export_unet_to_onnx
from climatevision.models.unet import UNet

model = UNet(n_channels=4, n_classes=2)
export_unet_to_onnx(model, "models/deforestation_model.onnx")

# Run inference with ONNX Runtime
from climatevision.inference import run_onnx_inference
import numpy as np

image = np.random.randn(4, 256, 256).astype(np.float32)
result = run_onnx_inference(image, "models/deforestation_model.onnx")
print(f"Mean confidence: {result.mean_confidence:.4f}")
print(f"Latency: {result.latency_ms:.2f}ms")

# Automatic fallback to PyTorch if ONNX unavailable
from climatevision.inference import run_inference_with_fallback
result = run_inference_with_fallback(
    image,
    onnx_model_path="models/deforestation_model.onnx",
)
print(f"Engine used: {result['engine']}")

# Benchmark ONNX performance
from climatevision.inference import benchmark_onnx_model
bench = benchmark_onnx_model(
    "models/deforestation_model.onnx",
    input_shape=(1, 4, 256, 256),
)
print(f"Mean latency: {bench.mean_latency_ms:.2f}ms")
print(f"Throughput: {bench.throughput_fps:.1f} FPS")
```

### Dependencies

```
onnx>=1.14.0       # Model export and validation
onnxruntime>=1.15.0 # ONNX Runtime inference
```

### Testing

```bash
cd /home/fa/projects/deliverables/climatevision
pip install -e /home/fa/projects/climatevision-work  # Install base package
pip install onnx onnxruntime pytest
pytest tests/test_onnx_runtime.py -v
```

### Architecture

```
PyTorch Model
      │
      ▼ (export_unet_to_onnx)
   ONNX Model
      │
      ▼ (ONNXSession)
ONNX Runtime
      │
      ├── CPUExecutionProvider
      ├── CUDAExecutionProvider
      └── CoreMLExecutionProvider (macOS)
      │
      ▼ (run_onnx_inference)
   Predictions
```

### Performance

Typical speedup over PyTorch CPU inference:
- **CPU**: 2-5x faster (ONNX Runtime graph optimization)
- **CUDA**: 1.5-3x faster (optimized CUDA kernels)
- **Batch inference**: Additional 2-4x throughput improvement

### Integration with Existing Code

To integrate with the existing ClimateVision API (`api/main.py`):

```python
# In api/main.py, modify the predict endpoint:
from climatevision.inference.onnx_runtime import run_inference_with_fallback

# Replace:
#   result_payload = run_inference_from_gee(...)
# With:
result_payload = run_inference_with_fallback(
    image_array,
    onnx_model_path=f"models/{body.analysis_type}_model.onnx",
    analysis_type=body.analysis_type,
)
```
