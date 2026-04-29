"""
Inference utilities for ClimateVision model predictions.

Provides multiple inference backends:
- PyTorch inference (default, from pipeline.py)
- ONNX Runtime inference (optimized, from onnx_runtime.py)
- PyTorch -> ONNX export utilities (from onnx_export.py)

Usage:
    # PyTorch inference (existing)
    from climatevision.inference import run_inference
    
    # ONNX Runtime inference
    from climatevision.inference import run_onnx_inference, get_onnx_session
    
    # Export to ONNX
    from climatevision.inference import export_unet_to_onnx
"""

from .pipeline import (
    run_inference,
    run_inference_from_file,
    run_inference_from_gee,
)

from .onnx_runtime import (
    ONNXSession,
    ONNXInferenceResult,
    ONNXBenchmarkResult,
    run_onnx_inference,
    get_onnx_session,
    clear_session_cache,
    benchmark_onnx_model,
    get_onnx_model_info,
    run_inference_with_fallback,
)

from .onnx_export import (
    export_unet_to_onnx,
    export_siamese_to_onnx,
    export_model_from_checkpoint,
    validate_onnx_model,
    export_all_analysis_types,
)

__all__ = [
    # PyTorch inference (existing)
    "run_inference",
    "run_inference_from_file",
    "run_inference_from_gee",
    # ONNX Runtime inference
    "ONNXSession",
    "ONNXInferenceResult",
    "ONNXBenchmarkResult",
    "run_onnx_inference",
    "get_onnx_session",
    "clear_session_cache",
    "benchmark_onnx_model",
    "get_onnx_model_info",
    "run_inference_with_fallback",
    # ONNX export
    "export_unet_to_onnx",
    "export_siamese_to_onnx",
    "export_model_from_checkpoint",
    "validate_onnx_model",
    "export_all_analysis_types",
]
