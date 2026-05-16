"""
ClimateVision Security Module

Provides API security and inference pipeline protection:
- Input validation and sanitization
- Rate limiting per API key
- File upload validation
- Adversarial input detection
- Security scanning and reporting
"""

from .api_security import (
    SecurityConfig,
    validate_payload_size,
    validate_bbox,
    validate_file_upload,
    sanitize_string_input,
    RateLimiter,
    SecurityMiddleware,
)
from .pipeline_guard import (
    PipelineGuard,
    detect_adversarial_input,
    validate_model_output,
    InputAnomalyDetector,
)

__all__ = [
    # API Security
    "SecurityConfig",
    "validate_payload_size",
    "validate_bbox",
    "validate_file_upload",
    "sanitize_string_input",
    "RateLimiter",
    "SecurityMiddleware",
    # Pipeline Guard
    "PipelineGuard",
    "detect_adversarial_input",
    "validate_model_output",
    "InputAnomalyDetector",
]
