"""
ONNX Runtime Inference Engine for ClimateVision.

Provides a high-performance inference backend using ONNX Runtime,
with automatic fallback to PyTorch when ONNX models are unavailable.

Features:
- ONNX Runtime session management with per-model caching
- Automatic device selection (CPU/CUDA/MPS)
- Latency benchmarking and performance metrics
- Graceful fallback to PyTorch inference
- Batch inference support for throughput optimization
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_ONNX_DIR = Path(__file__).resolve().parents[3] / "models"
_ONNX_SESSION_CACHE: dict[str, "ONNXSession"] = {}

# ONNX Runtime execution providers (ordered by preference)
_EXECUTION_PROVIDERS = [
    ("CUDAExecutionProvider", {"device_id": "0"}),
    "CPUExecutionProvider",
]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ONNXInferenceResult:
    """Result from ONNX Runtime inference."""
    predictions: np.ndarray
    """Class predictions (H, W) or (N, H, W)"""
    probabilities: np.ndarray
    """Class probabilities (N, n_classes, H, W)"""
    mean_confidence: float
    """Mean confidence across all pixels"""
    latency_ms: float
    """Inference latency in milliseconds"""
    model_path: str
    """Path to the ONNX model used"""
    execution_provider: str
    """Execution provider used (CPU/CUDA)"""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""


@dataclass
class ONNXBenchmarkResult:
    """Benchmark results for ONNX Runtime inference."""
    model_path: str
    input_shape: tuple[int, ...]
    num_warmup_runs: int
    num_benchmark_runs: int
    mean_latency_ms: float
    std_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_fps: float
    execution_provider: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ONNX Session Manager
# ---------------------------------------------------------------------------

class ONNXSession:
    """
    Manages an ONNX Runtime inference session with caching and device selection.

    Automatically selects the best available execution provider and caches
    sessions for repeated inference calls.

    Args:
        model_path: Path to the ONNX model file
        execution_providers: List of execution providers (auto-detected if None)
        session_options: Optional ONNX Runtime session options dict
    """

    def __init__(
        self,
        model_path: str | Path,
        execution_providers: Optional[list] = None,
        session_options: Optional[dict] = None,
    ):
        self.model_path = str(model_path)
        self._session = self._create_session(
            execution_providers or _EXECUTION_PROVIDERS,
            session_options or {},
        )
        self._input_name: Optional[str] = None
        self._output_names: Optional[list[str]] = None

    @property
    def input_name(self) -> str:
        """Get the name of the model's input tensor."""
        if self._input_name is None:
            self._input_name = self._session.get_inputs()[0].name
        return self._input_name

    @property
    def output_names(self) -> list[str]:
        """Get the names of the model's output tensors."""
        if self._output_names is None:
            self._output_names = [o.name for o in self._session.get_outputs()]
        return self._output_names

    @property
    def input_shape(self) -> list[int]:
        """Get the expected input shape of the model."""
        return self._session.get_inputs()[0].shape

    @property
    def execution_provider(self) -> str:
        """Get the active execution provider."""
        providers = self._session.get_providers()
        return providers[0] if providers else "Unknown"

    def _create_session(
        self,
        execution_providers: list,
        session_options: dict,
    ) -> Any:
        """Create an ONNX Runtime inference session."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install with: pip install onnxruntime"
            )

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        # Build session options
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # Suppress info/warning logs

        # Thread configuration
        if "intra_op_num_threads" in session_options:
            opts.intra_op_num_threads = session_options["intra_op_num_threads"]
        if "inter_op_num_threads" in session_options:
            opts.inter_op_num_threads = session_options["inter_op_num_threads"]

        # Enable graph optimization
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Determine available providers
        available_providers = set(ort.get_available_providers())
        selected_providers = []
        for ep in execution_providers:
            ep_name = ep if isinstance(ep, str) else ep[0]
            if ep_name in available_providers:
                selected_providers.append(ep)

        if not selected_providers:
            selected_providers = ["CPUExecutionProvider"]

        logger.info(
            "Creating ONNX session: %s (providers: %s)",
            self.model_path,
            selected_providers,
        )

        return ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=selected_providers,
        )

    def run(self, input_data: np.ndarray) -> list[np.ndarray]:
        """
        Run inference on input data.

        Args:
            input_data: Input tensor as numpy array

        Returns:
            List of output tensors
        """
        return self._session.run(None, {self.input_name: input_data})

    def get_input_info(self) -> dict[str, Any]:
        """Get information about the model's input tensor."""
        inp = self._session.get_inputs()[0]
        return {
            "name": inp.name,
            "shape": inp.shape,
            "type": inp.type,
        }

    def get_output_info(self) -> list[dict[str, Any]]:
        """Get information about the model's output tensors."""
        return [
            {"name": o.name, "shape": o.shape, "type": o.type}
            for o in self._session.get_outputs()
        ]


# ---------------------------------------------------------------------------
# Session Cache
# ---------------------------------------------------------------------------

def get_onnx_session(
    model_path: str | Path,
    *,
    force_reload: bool = False,
    **kwargs,
) -> ONNXSession:
    """
    Get a cached ONNX session, creating one if necessary.

    Args:
        model_path: Path to the ONNX model
        force_reload: Force reload even if cached
        **kwargs: Additional arguments for ONNXSession

    Returns:
        ONNXSession instance
    """
    path_str = str(model_path)
    if path_str in _ONNX_SESSION_CACHE and not force_reload:
        return _ONNX_SESSION_CACHE[path_str]

    session = ONNXSession(model_path, **kwargs)
    _ONNX_SESSION_CACHE[path_str] = session
    return session


def clear_session_cache() -> None:
    """Clear all cached ONNX sessions."""
    _ONNX_SESSION_CACHE.clear()


# ---------------------------------------------------------------------------
# Core Inference Functions
# ---------------------------------------------------------------------------

def run_onnx_inference(
    image: np.ndarray,
    model_path: str | Path,
    *,
    analysis_type: str = "deforestation",
    n_classes: int = 2,
    batch_size: int = 1,
    return_latencies: bool = False,
) -> ONNXInferenceResult | dict[str, Any]:
    """
    Run inference using an ONNX model.

    Args:
        image: Input image array of shape (C, H, W) or (N, C, H, W)
        model_path: Path to the ONNX model file
        analysis_type: Type of analysis (for metadata)
        n_classes: Number of output classes
        batch_size: Batch size for inference
        return_latencies: If True, return detailed timing info

    Returns:
        ONNXInferenceResult with predictions and metadata
    """
    session = get_onnx_session(model_path)

    # Ensure input is (N, C, H, W)
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)
    elif image.ndim == 2:
        image = image.reshape(1, 1, image.shape[0], image.shape[1])

    # Ensure float32
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    # Warm-up run (skip timing)
    session.run(session.input_name, {session.input_name: image[:1]})

    # Benchmark run
    latencies: list[float] = []
    all_outputs: list[np.ndarray] = []

    # Process in batches
    n_samples = image.shape[0]
    for i in range(0, n_samples, batch_size):
        batch = image[i:i + batch_size]
        start = time.perf_counter()
        outputs = session.run(batch)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)
        all_outputs.extend(outputs)

    total_latency = sum(latencies)

    # Extract predictions
    # ONNX output is typically logits: (N, n_classes, H, W)
    logits = all_outputs[0] if isinstance(all_outputs, list) else all_outputs

    # Apply softmax to get probabilities
    probabilities = _softmax(logits, axis=1)
    predictions = np.argmax(probabilities, axis=1)  # (N, H, W)

    # Compute confidence
    max_probs = probabilities.max(axis=1)  # (N, H, W)
    mean_confidence = float(max_probs.mean())

    result = ONNXInferenceResult(
        predictions=predictions,
        probabilities=probabilities,
        mean_confidence=round(mean_confidence, 4),
        latency_ms=round(total_latency, 2),
        model_path=str(model_path),
        execution_provider=session.execution_provider,
        metadata={
            "analysis_type": analysis_type,
            "n_classes": n_classes,
            "input_shape": list(image.shape),
            "output_shape": list(logits.shape),
            "batch_size": batch_size,
            "num_batches": len(latencies),
        },
    )

    if return_latencies:
        result.metadata["batch_latencies_ms"] = [round(l, 2) for l in latencies]

    return result


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark_onnx_model(
    model_path: str | Path,
    input_shape: tuple[int, ...] = (1, 4, 256, 256),
    *,
    num_warmup_runs: int = 5,
    num_benchmark_runs: int = 50,
    batch_size: int = 1,
) -> ONNXBenchmarkResult:
    """
    Benchmark ONNX Runtime inference performance.

    Args:
        model_path: Path to the ONNX model
        input_shape: Shape of input tensor (N, C, H, W)
        num_warmup_runs: Number of warmup iterations
        num_benchmark_runs: Number of benchmark iterations
        batch_size: Batch size for throughput calculation

    Returns:
        ONNXBenchmarkResult with detailed timing statistics
    """
    session = get_onnx_session(model_path)

    # Generate random input data
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup runs
    for _ in range(num_warmup_runs):
        session.run(dummy_input)

    # Benchmark runs
    latencies: list[float] = []
    for _ in range(num_benchmark_runs):
        start = time.perf_counter()
        session.run(dummy_input)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    latencies_arr = np.array(latencies)

    return ONNXBenchmarkResult(
        model_path=str(model_path),
        input_shape=input_shape,
        num_warmup_runs=num_warmup_runs,
        num_benchmark_runs=num_benchmark_runs,
        mean_latency_ms=round(float(latencies_arr.mean()), 2),
        std_latency_ms=round(float(latencies_arr.std()), 2),
        p50_latency_ms=round(float(np.percentile(latencies_arr, 50)), 2),
        p95_latency_ms=round(float(np.percentile(latencies_arr, 95)), 2),
        p99_latency_ms=round(float(np.percentile(latencies_arr, 99)), 2),
        throughput_fps=round(batch_size / (latencies_arr.mean() / 1000), 1),
        execution_provider=session.execution_provider,
        metadata={
            "all_latencies_ms": [round(l, 2) for l in latencies],
        },
    )


# ---------------------------------------------------------------------------
# Model Info
# ---------------------------------------------------------------------------

def get_onnx_model_info(model_path: str | Path) -> dict[str, Any]:
    """
    Get detailed information about an ONNX model.

    Args:
        model_path: Path to the ONNX model

    Returns:
        Dictionary with model metadata
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return {"error": "onnxruntime not installed"}

    path = Path(model_path)
    if not path.exists():
        return {"error": f"Model not found: {model_path}"}

    # Load model proto for additional info
    try:
        import onnx
        onnx_model = onnx.load(str(path))
        doc_string = onnx_model.doc_string
        producer_name = onnx_model.producer_name
        producer_version = onnx_model.producer_version
        onnx_version = onnx_model.ir_version
    except ImportError:
        doc_string = None
        producer_name = None
        producer_version = None
        onnx_version = None
    except Exception:
        doc_string = None
        producer_name = None
        producer_version = None
        onnx_version = None

    # Session info
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "doc_string": doc_string,
        "producer": producer_name,
        "producer_version": producer_version,
        "onnx_ir_version": onnx_version,
        "inputs": session.get_inputs()[0].__dict__ if session.get_inputs() else {},
        "outputs": [o.__dict__ for o in session.get_outputs()],
        "providers": session.get_providers(),
    }


# ---------------------------------------------------------------------------
# Fallback Inference
# ---------------------------------------------------------------------------

def run_inference_with_fallback(
    image: np.ndarray,
    onnx_model_path: Optional[str | Path] = None,
    *,
    analysis_type: str = "deforestation",
    n_classes: int = 2,
) -> dict[str, Any]:
    """
    Run inference with automatic ONNX -> PyTorch fallback.

    Tries ONNX Runtime first, falls back to PyTorch if ONNX model
    is unavailable or inference fails.

    Args:
        image: Input image (C, H, W) or (N, C, H, W)
        onnx_model_path: Optional path to ONNX model
        analysis_type: Type of analysis
        n_classes: Number of output classes

    Returns:
        Dictionary with inference results and engine used
    """
    # Try ONNX first
    if onnx_model_path and Path(onnx_model_path).exists():
        try:
            result = run_onnx_inference(
                image,
                onnx_model_path,
                analysis_type=analysis_type,
                n_classes=n_classes,
            )
            return {
                "engine": "onnx_runtime",
                "predictions": result.predictions,
                "probabilities": result.probabilities,
                "mean_confidence": result.mean_confidence,
                "latency_ms": result.latency_ms,
                "model_path": result.model_path,
                "execution_provider": result.execution_provider,
            }
        except Exception as exc:
            logger.warning(
                "ONNX inference failed (%s), falling back to PyTorch", exc
            )

    # Fallback to PyTorch
    logger.info("Using PyTorch inference (ONNX unavailable)")
    try:
        from climatevision.inference.pipeline import run_inference as pytorch_inference
        result = pytorch_inference(
            image,
            analysis_type=analysis_type,
        )
        result["engine"] = "pytorch"
        return result
    except Exception as exc:
        logger.error("PyTorch fallback also failed: %s", exc)
        raise RuntimeError(
            f"All inference engines failed. ONNX error logged. PyTorch error: {exc}"
        )
