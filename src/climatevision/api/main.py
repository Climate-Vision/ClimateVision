"""
ClimateVision API

FastAPI-based REST API for climate monitoring including:
- Deforestation detection
- Arctic ice melting analysis
- Flood detection
- Organization (NGO) management
- Alert and subscription systems

The endpoints live in ``climatevision.api.routes``, one module per resource;
this module builds the application and wires in middleware and routers.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from climatevision.db import init_db
from climatevision.api.routes import include_routers
from climatevision.api.shared import PROJECT_ROOT
from climatevision.security.api_security import SecurityMiddleware
from climatevision.workers.alert_delivery import AlertDeliveryWorker

logger = logging.getLogger(__name__)


# ===== Audit Logging Middleware =====


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Log every API request with method, path, status code, and duration."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "API request | method=%s path=%s status=%s duration_ms=%s ip=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request.client.host if request.client else "unknown",
        )
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        return response


# ===== Application Factory =====


def create_app() -> FastAPI:
    init_db()

    # Set up alert delivery worker
    alert_worker = AlertDeliveryWorker()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Start background workers on app startup, stop on shutdown."""
        await alert_worker.start()
        logger.info("Alert delivery worker started")
        yield
        await alert_worker.stop()
        logger.info("Alert delivery worker stopped")

    app = FastAPI(
        title="ClimateVision API",
        version="0.2.0",
        description="""
        Climate monitoring API for detecting deforestation, ice melting, flooding, and more.
        
        ## Features
        - Multi-type climate analysis (deforestation, ice melting, flooding)
        - Organization (NGO) management
        - Region subscriptions and alerts
        - Satellite imagery processing
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(AuditLogMiddleware)

    # CORS: allow local dev origins plus any origins from CLIMATEVISION_CORS_ORIGINS
    default_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    extra_origins = [
        origin.strip()
        for origin in os.environ.get("CLIMATEVISION_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(dict.fromkeys(default_origins + extra_origins)),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Register the RequestIDMiddleware LAST so it sits OUTERMOST in the
    # middleware stack: it must run before every other middleware so the
    # request_id ContextVar is set in time for AuditLogMiddleware's logger
    # call (and for any inference-pipeline code further down). Starlette
    # wraps middleware in reverse add_middleware order, so the last call
    # is the outermost wrapper.
    from climatevision.api.middleware import RequestIDMiddleware

    app.add_middleware(RequestIDMiddleware)

    # Wire OWASP-aligned security controls (rate limiting, payload limits, etc.)
    app.add_middleware(SecurityMiddleware)

    from climatevision.api import admin as _admin

    app.include_router(_admin.router)

    # ===== Core Endpoints =====

    _frontend_dist = PROJECT_ROOT / "frontend" / "dist"

    if not (_frontend_dist / "index.html").exists():

        @app.get("/")
        def root() -> RedirectResponse:
            """Redirect to API docs when no frontend is built."""
            return RedirectResponse(url="/docs", status_code=302)

    include_routers(app)

    # ===== Static Files =====

    # Serve built frontend when dist exists (production mode).
    # Registered last so all API routes above take precedence. Unknown paths
    # fall back to index.html so React Router deep links (/runs, /upload, ...)
    # work on direct navigation and refresh.
    dist_dir = _frontend_dist
    if (dist_dir / "index.html").exists():
        app.mount(
            "/assets", StaticFiles(directory=str(dist_dir / "assets")), name="assets"
        )

        @app.get("/{full_path:path}", include_in_schema=False)
        def frontend(full_path: str) -> FileResponse:
            candidate = dist_dir / full_path
            if full_path and ".." not in full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist_dir / "index.html")

    return app


app = create_app()
