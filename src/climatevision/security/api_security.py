"""
API Security Module for ClimateVision.

Implements OWASP-aligned security controls:
- Input validation and sanitization
- Rate limiting per API key
- File upload validation (magic bytes, extensions)
- Payload size limits
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Callable

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration settings."""

    max_payload_size_bytes: int = 50 * 1024 * 1024  # 50 MB
    max_bbox_area_degrees: float = 100.0  # Max area in square degrees
    max_date_range_days: int = 365  # Max date range
    rate_limit_requests: int = 100  # Requests per window
    rate_limit_window_seconds: int = 60  # Window size
    allowed_file_extensions: set[str] = field(
        default_factory=lambda: {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".geotiff"}
    )
    max_filename_length: int = 255
    blocked_patterns: list[str] = field(
        default_factory=lambda: [
            r"\.\.\/",  # Path traversal
            r"<script",  # XSS
            r"javascript:",  # XSS
            r"on\w+=",  # Event handlers
            r";\s*drop\s+table",  # SQL injection
            r";\s*delete\s+from",  # SQL injection
            r"\$\{",  # Template injection
            r"\{\{",  # Template injection
        ]
    )


# Magic bytes for file type validation
FILE_SIGNATURES = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"II*\x00": "image/tiff",  # Little-endian TIFF
    b"MM\x00*": "image/tiff",  # Big-endian TIFF
    b"II\x2b\x00": "image/tiff",  # BigTIFF little-endian
    b"MM\x00\x2b": "image/tiff",  # BigTIFF big-endian
}


class RateLimiter:
    """
    Token bucket rate limiter with per-key tracking.

    Thread-safe for concurrent access in async contexts.
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed for the given key."""
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        self._buckets[key] = [
            ts for ts in self._buckets[key] if ts > window_start
        ]

        # Check limit
        if len(self._buckets[key]) >= self.max_requests:
            return False

        # Record request
        self._buckets[key].append(now)
        return True

    def get_remaining(self, key: str) -> int:
        """Get remaining requests for the key."""
        now = time.time()
        window_start = now - self.window_seconds
        recent = [ts for ts in self._buckets[key] if ts > window_start]
        return max(0, self.max_requests - len(recent))

    def get_reset_time(self, key: str) -> float:
        """Get seconds until rate limit resets."""
        if not self._buckets[key]:
            return 0.0
        oldest = min(self._buckets[key])
        reset = oldest + self.window_seconds - time.time()
        return max(0.0, reset)


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for security checks.

    Applies rate limiting and basic request validation.
    """

    def __init__(
        self,
        app: Any,
        config: Optional[SecurityConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        super().__init__(app)
        self.config = config or SecurityConfig()
        self.rate_limiter = rate_limiter or RateLimiter(
            max_requests=self.config.rate_limit_requests,
            window_seconds=self.config.rate_limit_window_seconds,
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier (API key or IP)
        api_key = request.headers.get("X-API-Key")
        client_id = api_key or (request.client.host if request.client else "unknown")

        # Rate limit check
        if not self.rate_limiter.is_allowed(client_id):
            remaining = self.rate_limiter.get_remaining(client_id)
            reset_time = self.rate_limiter.get_reset_time(client_id)

            logger.warning(
                "Rate limit exceeded for client %s",
                client_id[:16] + "..." if len(client_id) > 16 else client_id,
            )

            return Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=429,
                headers={
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Reset": str(int(reset_time)),
                    "Retry-After": str(int(reset_time) + 1),
                    "Content-Type": "application/json",
                },
            )

        # Content-Length check
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.config.max_payload_size_bytes:
                    return Response(
                        content='{"detail": "Payload too large"}',
                        status_code=413,
                        headers={"Content-Type": "application/json"},
                    )
            except ValueError:
                pass

        # Add security headers to response
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Add rate limit headers
        remaining = self.rate_limiter.get_remaining(client_id)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Limit"] = str(self.config.rate_limit_requests)

        return response


def validate_payload_size(
    data: bytes,
    max_size: int = 50 * 1024 * 1024,
) -> tuple[bool, str]:
    """
    Validate payload size.

    Args:
        data: Raw payload bytes
        max_size: Maximum allowed size in bytes

    Returns:
        (is_valid, error_message)
    """
    if len(data) > max_size:
        return False, f"Payload size ({len(data)} bytes) exceeds maximum ({max_size} bytes)"
    return True, ""


def validate_bbox(
    bbox: list[float],
    max_area: float = 100.0,
) -> tuple[bool, str]:
    """
    Validate bounding box coordinates.

    Checks:
    - Exactly 4 values
    - Valid longitude/latitude ranges
    - West < East, South < North
    - Area within limits (prevent DoS via huge queries)

    Args:
        bbox: [west, south, east, north]
        max_area: Maximum area in square degrees

    Returns:
        (is_valid, error_message)
    """
    if len(bbox) != 4:
        return False, "bbox must have exactly 4 values: [west, south, east, north]"

    west, south, east, north = bbox

    # Type check
    for i, v in enumerate(bbox):
        if not isinstance(v, (int, float)):
            return False, f"bbox[{i}] must be a number, got {type(v).__name__}"

    # Longitude range
    if not (-180 <= west <= 180):
        return False, f"Invalid west longitude: {west}. Must be between -180 and 180"
    if not (-180 <= east <= 180):
        return False, f"Invalid east longitude: {east}. Must be between -180 and 180"

    # Latitude range
    if not (-90 <= south <= 90):
        return False, f"Invalid south latitude: {south}. Must be between -90 and 90"
    if not (-90 <= north <= 90):
        return False, f"Invalid north latitude: {north}. Must be between -90 and 90"

    # Order check
    if west >= east:
        return False, f"West ({west}) must be less than east ({east})"
    if south >= north:
        return False, f"South ({south}) must be less than north ({north})"

    # Area check (prevent huge GEE queries)
    area = (east - west) * (north - south)
    if area > max_area:
        return False, f"Bounding box area ({area:.2f} sq degrees) exceeds maximum ({max_area} sq degrees)"

    return True, ""


def validate_file_upload(
    content: bytes,
    filename: str,
    config: Optional[SecurityConfig] = None,
) -> tuple[bool, str]:
    """
    Validate uploaded file.

    Checks:
    - Filename length and characters
    - File extension whitelist
    - Magic bytes match expected type
    - No path traversal in filename

    Args:
        content: File content bytes
        filename: Original filename
        config: Security configuration

    Returns:
        (is_valid, error_message)
    """
    config = config or SecurityConfig()

    # Filename length
    if len(filename) > config.max_filename_length:
        return False, f"Filename too long ({len(filename)} > {config.max_filename_length})"

    # Path traversal check
    if ".." in filename or "/" in filename or "\\" in filename:
        return False, "Invalid filename: path traversal detected"

    # Extension check
    ext = Path(filename).suffix.lower()
    if ext not in config.allowed_file_extensions:
        return False, f"File extension '{ext}' not allowed. Allowed: {config.allowed_file_extensions}"

    # Magic bytes validation
    detected_type = None
    for signature, mime_type in FILE_SIGNATURES.items():
        if content.startswith(signature):
            detected_type = mime_type
            break

    if detected_type is None:
        return False, "Unknown file type. Could not verify magic bytes."

    # Check extension matches detected type
    expected_extensions = {
        "image/png": {".png"},
        "image/jpeg": {".jpg", ".jpeg"},
        "image/tiff": {".tif", ".tiff", ".geotiff"},
    }

    if ext not in expected_extensions.get(detected_type, set()):
        return False, f"File extension '{ext}' does not match detected type '{detected_type}'"

    return True, ""


def sanitize_string_input(
    value: str,
    max_length: int = 1000,
    config: Optional[SecurityConfig] = None,
) -> tuple[str, list[str]]:
    """
    Sanitize string input by removing potentially dangerous patterns.

    Args:
        value: Input string
        max_length: Maximum allowed length
        config: Security configuration

    Returns:
        (sanitized_value, list of warnings)
    """
    config = config or SecurityConfig()
    warnings = []

    # Length limit
    if len(value) > max_length:
        value = value[:max_length]
        warnings.append(f"Input truncated to {max_length} characters")

    # Check for blocked patterns
    original = value
    for pattern in config.blocked_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            value = re.sub(pattern, "", value, flags=re.IGNORECASE)
            warnings.append(f"Removed blocked pattern: {pattern}")

    # HTML entity encoding for special characters
    value = (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )

    if value != original and not warnings:
        warnings.append("Input was sanitized")

    return value, warnings


def generate_request_hash(request: Request) -> str:
    """Generate a unique hash for a request for logging/tracking."""
    components = [
        request.method,
        str(request.url),
        request.headers.get("User-Agent", ""),
        request.headers.get("X-API-Key", "")[:8] if request.headers.get("X-API-Key") else "",
        str(time.time()),
    ]
    return hashlib.sha256("|".join(components).encode()).hexdigest()[:16]


def check_api_key_format(api_key: str) -> tuple[bool, str]:
    """
    Validate API key format.

    Args:
        api_key: The API key to validate

    Returns:
        (is_valid, error_message)
    """
    if not api_key:
        return False, "API key is required"

    if len(api_key) < 16:
        return False, "API key too short"

    if len(api_key) > 128:
        return False, "API key too long"

    # Only alphanumeric and limited special chars
    if not re.match(r"^[a-zA-Z0-9_\-]+$", api_key):
        return False, "API key contains invalid characters"

    return True, ""
