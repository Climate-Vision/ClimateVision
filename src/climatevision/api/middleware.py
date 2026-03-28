"""
Request logging and audit middleware for ClimateVision API.

Provides structured logging, request tracing, and audit trails
for all API requests to ensure observability and compliance.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured request logging and audit trails.

    Logs all requests with timing, status codes, and request IDs
    for traceability and debugging.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.perf_counter()

        # Log incoming request
        logger.info(
            "request_started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
            }
        )

        try:
            response = await call_next(request)

            # Calculate processing time
            process_time_ms = (time.perf_counter() - start_time) * 1000

            # Add headers for tracing
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"

            # Log completed request
            logger.info(
                "request_completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time_ms": round(process_time_ms, 2),
                }
            )

            return response

        except Exception as e:
            process_time_ms = (time.perf_counter() - start_time) * 1000

            logger.error(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "process_time_ms": round(process_time_ms, 2),
                },
                exc_info=True
            )
            raise


class AuditLogMiddleware(BaseHTTPMiddleware):
    """
    Middleware for audit logging of sensitive operations.

    Creates audit trail entries for data-modifying operations
    that may need to be reviewed for compliance.
    """

    AUDITED_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
    AUDITED_PATHS = {"/predict", "/organizations", "/subscriptions", "/alerts"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        should_audit = (
            request.method in self.AUDITED_METHODS and
            any(request.url.path.startswith(p) for p in self.AUDITED_PATHS)
        )

        if should_audit:
            request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

            # Log audit event before processing
            logger.info(
                "audit_event",
                extra={
                    "audit_type": "api_request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "client_ip": request.client.host if request.client else "unknown",
                    "timestamp": time.time(),
                }
            )

        response = await call_next(request)

        if should_audit:
            logger.info(
                "audit_event_completed",
                extra={
                    "audit_type": "api_response",
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "success": response.status_code < 400,
                }
            )

        return response


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured JSON logging for the API."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='{"timestamp":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S"
    )
