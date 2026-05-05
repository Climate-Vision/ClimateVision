"""Tests for the X-Request-ID middleware.

Covers issue #44: request log lines from the inference pipeline and helper
modules need to carry the same identifier as the inbound HTTP request.
"""

from __future__ import annotations

import logging
import re
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

from climatevision.api.middleware import (
    RequestIDLogFilter,
    RequestIDMiddleware,
    request_id_var,
)


UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    logger = logging.getLogger("climatevision.tests.request_id")

    @app.get("/echo")
    def echo() -> dict[str, str]:
        # Read the contextvar the way inference/pipeline.py would
        return {"request_id": request_id_var.get() or ""}

    @app.get("/log")
    def log_route() -> dict[str, str]:
        logger.info("inside-handler", extra={"sample": True})
        return {"ok": "1"}

    return app


def test_response_has_uuid_request_id_when_header_absent() -> None:
    """Without an X-Request-ID header the middleware mints a fresh UUID4."""
    client = TestClient(_build_app())
    response = client.get("/echo")
    assert response.status_code == 200

    request_id = response.headers["X-Request-ID"]
    assert UUID_RE.match(request_id), f"not a UUID4 shape: {request_id!r}"
    # And the same value flowed into the contextvar that the route saw.
    assert response.json()["request_id"] == request_id


def test_response_echoes_explicit_request_id() -> None:
    """When the client sends X-Request-ID the middleware must echo it back."""
    client = TestClient(_build_app())
    sent = "test-id-deadbeef"
    response = client.get("/echo", headers={"X-Request-ID": sent})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == sent
    assert response.json()["request_id"] == sent


def test_log_records_include_request_id_via_filter(
    caplog: object,
) -> None:
    """Logs emitted inside the request show request_id once the filter is on."""
    handler = logging.StreamHandler()
    handler.addFilter(RequestIDLogFilter())
    handler.setFormatter(
        logging.Formatter("%(request_id)s | %(message)s")
    )

    target_logger = logging.getLogger("climatevision.tests.request_id")
    target_logger.setLevel(logging.INFO)
    target_logger.addHandler(handler)
    try:
        # Use pytest's caplog with the same filter so we can introspect records.
        with caplog.at_level(logging.INFO, logger="climatevision.tests.request_id"):  # type: ignore[attr-defined]
            for record_filter in [RequestIDLogFilter()]:
                caplog.handler.addFilter(record_filter)  # type: ignore[attr-defined]

            sent = str(uuid.uuid4())
            client = TestClient(_build_app())
            response = client.get("/log", headers={"X-Request-ID": sent})
            assert response.status_code == 200

        records = [
            r for r in caplog.records  # type: ignore[attr-defined]
            if r.name == "climatevision.tests.request_id"
        ]
        assert records, "expected the route handler to emit a log record"
        record = records[0]
        assert getattr(record, "request_id", None) == sent
    finally:
        target_logger.removeHandler(handler)


def test_request_id_var_resets_after_response() -> None:
    """The ContextVar must not leak across requests."""
    client = TestClient(_build_app())
    client.get("/echo", headers={"X-Request-ID": "first"})
    # Outside any request the var returns its default ("" via ``or ''``).
    assert request_id_var.get() is None
