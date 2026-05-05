"""
ClimateVision API

FastAPI-based REST API for climate monitoring including:
- Deforestation detection
- Arctic ice melting analysis
- Flood detection
- Organization (NGO) management
- Alert and subscription systems
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Literal

from pydantic import field_validator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Header, Query, Depends, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from pydantic import BaseModel, Field, EmailStr, model_validator

from climatevision.db import (
    get_connection,
    init_db,
    create_organization,
    get_organization,
    get_organization_by_api_key,
    list_organizations,
    create_subscription,
    get_subscriptions_for_organization,
    create_organization_alert,
    get_alerts_for_organization,
    acknowledge_alert,
    mark_alert_delivered,
)
from climatevision.inference import run_inference_from_file, run_inference_from_gee
from climatevision.api.auth import require_api_key

logger = logging.getLogger(__name__)


# ===== Type Definitions =====

AnalysisType = Literal["deforestation", "ice_melting", "flooding", "drought", "wildfire"]
OrganizationType = Literal["ngo", "government", "research", "corporate"]
NotificationChannel = Literal["email", "webhook", "api", "sms"]
AlertSeverity = Literal["low", "medium", "high", "critical"]

SUPPORTED_ANALYSIS_TYPES: list[dict[str, Any]] = [
    {
        "name": "deforestation",
        "display_name": "Deforestation Detection",
        "description": "Monitor forest coverage and detect deforestation events",
        "enabled": True,
        "bands": ["B04", "B03", "B02", "B08"],
        "classes": ["non_forest", "forest"],
    },
    {
        "name": "ice_melting",
        "display_name": "Arctic Ice Melting",
        "description": "Monitor sea ice extent and melting patterns in polar regions",
        "enabled": True,
        "bands": ["B02", "B03", "B04", "B11"],
        "classes": ["sea_ice", "open_water", "land"],
    },
    {
        "name": "flooding",
        "display_name": "Flood Detection",
        "description": "Detect and monitor flooding events and affected areas",
        "enabled": True,
        "bands": ["B03", "B08", "B11"],
        "classes": ["water", "flooded", "dry_land"],
    },
    {
        "name": "drought",
        "display_name": "Drought Monitoring",
        "description": "Monitor vegetation stress and drought conditions",
        "enabled": False,
        "bands": ["B04", "B08", "B11", "B12"],
        "classes": ["normal", "stressed", "severe_drought"],
    },
    {
        "name": "wildfire",
        "display_name": "Wildfire Detection",
        "description": "Detect active fires and burned areas",
        "enabled": False,
        "bands": ["B04", "B08", "B11", "B12"],
        "classes": ["unburned", "burned", "active_fire"],
    },
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===== Request/Response Models =====

class PredictRequest(BaseModel):
    kind: str = Field(default="demo")
    analysis_type: AnalysisType = Field(default="deforestation")
    bbox: Optional[list[float]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is None:
            return v
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 values: [west, south, east, north]")
        west, south, east, north = v
        if not (-180 <= west <= 180 and -180 <= east <= 180):
            raise ValueError("bbox longitude values must be between -180 and 180")
        if not (-90 <= south <= 90 and -90 <= north <= 90):
            raise ValueError("bbox latitude values must be between -90 and 90")
        if west >= east:
            raise ValueError("bbox west longitude must be less than east longitude")
        if south >= north:
            raise ValueError("bbox south latitude must be less than north latitude")
        return v

    @model_validator(mode="after")
    def validate_date_range(self) -> "PredictRequest":
        if self.start_date and self.end_date:
            try:
                start = datetime.strptime(self.start_date, "%Y-%m-%d")
                end = datetime.strptime(self.end_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("start_date and end_date must be in YYYY-MM-DD format")
            if start >= end:
                raise ValueError("start_date must be earlier than end_date")
        return self


class RunRow(BaseModel):
    id: int
    kind: str
    status: str
    analysis_type: str = "deforestation"
    bbox: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    created_at: str
    updated_at: str


class ResultRow(BaseModel):
    id: int
    run_id: int
    payload: dict[str, Any]
    mask_path: Optional[str] = None
    created_at: str


# Organization models
class CreateOrganizationRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)
    type: OrganizationType = Field(default="ngo")
    description: Optional[str] = None
    contact_email: Optional[EmailStr] = None
    website_url: Optional[str] = None
    regions_of_interest: Optional[list[str]] = None


class OrganizationResponse(BaseModel):
    id: int
    name: str
    type: str
    description: Optional[str] = None
    logo_url: Optional[str] = None
    website_url: Optional[str] = None
    contact_email: Optional[str] = None
    active: bool
    created_at: str


class OrganizationWithKeyResponse(OrganizationResponse):
    api_key: str


class CreateSubscriptionRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    bbox: list[float] = Field(..., min_length=4, max_length=4)
    analysis_types: list[AnalysisType] = Field(default=["deforestation"])
    alert_threshold: float = Field(default=5.0, ge=0, le=100)
    notification_channel: NotificationChannel = Field(default="email")
    webhook_url: Optional[str] = None


class SubscriptionResponse(BaseModel):
    id: int
    organization_id: int
    name: Optional[str] = None
    bbox: list[float]
    analysis_types: list[str]
    alert_threshold: float
    notification_channel: str
    active: bool
    created_at: str


class AlertResponse(BaseModel):
    id: int
    organization_id: int
    alert_type: str
    severity: str
    title: str
    message: str
    delivered: bool
    acknowledged: bool
    created_at: str


class CreateAlertRequest(BaseModel):
    alert_type: str
    severity: AlertSeverity = Field(default="medium")
    title: str
    message: str
    subscription_id: Optional[int] = None
    run_id: Optional[int] = None
    details: Optional[str] = None


# ===== Helper Functions =====

def _load_template_result(
    *,
    bbox: Optional[list[float]],
    start_date: Optional[str],
    end_date: Optional[str],
    analysis_type: str = "deforestation",
) -> dict[str, Any]:
    """Load or create a template result for failed inference."""
    outputs_dir = Path(__file__).resolve().parents[3] / "outputs"
    template_path = outputs_dir / "inference_results.json"
    
    if template_path.exists():
        template: dict[str, Any] = json.loads(template_path.read_text(encoding="utf-8"))
    else:
        # Create analysis-specific template
        if analysis_type == "ice_melting":
            template = {
                "region": {"bbox": bbox or None},
                "inference": {
                    "image_size": [256, 256],
                    "ice_pixels": 0,
                    "water_pixels": 0,
                    "land_pixels": 0,
                    "ice_percentage": 0.0,
                    "mean_confidence": 0.0,
                },
            }
        elif analysis_type == "flooding":
            template = {
                "region": {"bbox": bbox or None},
                "inference": {
                    "image_size": [256, 256],
                    "flooded_pixels": 0,
                    "dry_pixels": 0,
                    "water_pixels": 0,
                    "flooded_percentage": 0.0,
                    "mean_confidence": 0.0,
                },
            }
        else:  # deforestation (default)
            template = {
                "region": {"bbox": bbox or None},
                "ndvi_stats": {"NDVI_min": 0.0, "NDVI_mean": 0.0, "NDVI_max": 0.0},
                "inference": {
                    "image_size": [256, 256],
                    "forest_pixels": 0,
                    "non_forest_pixels": 0,
                    "forest_percentage": 0.0,
                    "mean_confidence": 0.0,
                },
            }

    if bbox is not None:
        template.setdefault("region", {})["bbox"] = bbox
    if start_date and end_date:
        template.setdefault("region", {})["date_range"] = f"{start_date} to {end_date}"
    
    template["analysis_type"] = analysis_type

    return template


async def _persist_upload(*, run_id: int, file: UploadFile) -> str:
    """Save uploaded file to disk."""
    outputs_dir = Path(__file__).resolve().parents[3] / "outputs"
    uploads_dir = outputs_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / f"run_{run_id}_{file.filename}"
    dest.write_bytes(await file.read())
    return str(dest)


async def get_current_organization(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Optional[dict]:
    """Dependency to get current organization from API key."""
    if not x_api_key:
        return None
    org = get_organization_by_api_key(x_api_key)
    if org:
        return dict(org)
    return None


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
    )

    app.add_middleware(AuditLogMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
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

    # ===== Core Endpoints =====

    @app.get("/")
    def root() -> RedirectResponse:
        """Redirect to API docs when no frontend is built."""
        return RedirectResponse(url="/docs", status_code=302)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Health check endpoint with API information and config validation."""
        from climatevision.data.band_mapping import get_model_config

        enabled_types = [t for t in SUPPORTED_ANALYSIS_TYPES if t["enabled"]]
        config_issues: list[dict[str, Any]] = []

        for atype in enabled_types:
            name = atype["name"]
            try:
                cfg = get_model_config(name)
                expected_channels = len(atype["bands"])
                expected_classes = len(atype["classes"])
                if cfg.get("in_channels") != expected_channels:
                    config_issues.append(
                        {
                            "analysis_type": name,
                            "issue": "in_channels mismatch",
                            "expected": expected_channels,
                            "got": cfg.get("in_channels"),
                        }
                    )
                if cfg.get("num_classes") != expected_classes:
                    config_issues.append(
                        {
                            "analysis_type": name,
                            "issue": "num_classes mismatch",
                            "expected": expected_classes,
                            "got": cfg.get("num_classes"),
                        }
                    )
            except Exception as exc:
                config_issues.append(
                    {"analysis_type": name, "issue": "config missing", "error": str(exc)}
                )

        health_status = "ok" if not config_issues else "degraded"

        return {
            "status": health_status,
            "version": "0.2.0",
            "analysis_types": [t["name"] for t in enabled_types],
            "config_valid": len(config_issues) == 0,
            "config_issues": config_issues,
        }

    @app.get("/api/analysis-types")
    def list_analysis_types(enabled_only: bool = True) -> list[dict[str, Any]]:
        """List available analysis types."""
        if enabled_only:
            return [t for t in SUPPORTED_ANALYSIS_TYPES if t["enabled"]]
        return SUPPORTED_ANALYSIS_TYPES

    @app.get("/api/analysis-types/{analysis_type}")
    def get_analysis_type(analysis_type: str) -> dict[str, Any]:
        """Get details for a specific analysis type."""
        for t in SUPPORTED_ANALYSIS_TYPES:
            if t["name"] == analysis_type:
                return t
        raise HTTPException(status_code=404, detail=f"Analysis type '{analysis_type}' not found")

    # ===== Run Endpoints =====

    @app.get("/api/runs")
    def list_runs(
        limit: int = Query(default=50, le=200),
        offset: int = Query(default=0, ge=0),
        status: Optional[str] = None,
        analysis_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """List analysis runs with optional filtering and pagination metadata."""
        where_clauses = ["1=1"]
        params: list = []

        if status:
            where_clauses.append("status = ?")
            params.append(status)
        if analysis_type:
            where_clauses.append("analysis_type = ?")
            params.append(analysis_type)

        where = " AND ".join(where_clauses)

        with get_connection() as conn:
            total: int = conn.execute(
                f"SELECT COUNT(*) FROM runs WHERE {where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM runs WHERE {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                params + [int(limit), int(offset)],
            ).fetchall()

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "runs": [RunRow(**dict(r)) for r in rows],
        }

    @app.get("/api/runs/stats")
    def get_run_stats() -> dict[str, Any]:
        """Return aggregated run statistics for dashboard KPI cards."""
        with get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

            by_status = {
                row["status"]: row["count"]
                for row in conn.execute(
                    "SELECT status, COUNT(*) as count FROM runs GROUP BY status"
                ).fetchall()
            }

            by_analysis_type = {
                row["analysis_type"]: row["count"]
                for row in conn.execute(
                    "SELECT analysis_type, COUNT(*) as count FROM runs GROUP BY analysis_type"
                ).fetchall()
            }

            recent_completed = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE status = 'completed' "
                "AND created_at >= datetime('now', '-7 days')"
            ).fetchone()[0]

            alerts_total = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            alerts_unacknowledged = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE acknowledged = 0"
            ).fetchone()[0]

        return {
            "total_runs": total,
            "completed_last_7_days": recent_completed,
            "by_status": by_status,
            "by_analysis_type": by_analysis_type,
            "alerts": {
                "total": alerts_total,
                "unacknowledged": alerts_unacknowledged,
            },
        }

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: int) -> dict[str, Any]:
        """Get details for a specific run including results."""
        with get_connection() as conn:
            run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run is None:
                raise HTTPException(status_code=404, detail="Run not found")
            result = conn.execute(
                "SELECT * FROM results WHERE run_id = ? ORDER BY id DESC LIMIT 1", (run_id,)
            ).fetchone()

        payload: Optional[dict[str, Any]] = None
        mask_path: Optional[str] = None
        if result is not None:
            payload = json.loads(result["payload_json"])
            mask_path = result["mask_path"]

        return {
            "run": dict(run),
            "result": None
            if result is None
            else {
                "id": result["id"],
                "run_id": result["run_id"],
                "payload": payload,
                "mask_path": mask_path,
                "created_at": result["created_at"],
            },
        }

    # ===== Prediction Endpoints =====

    @app.post("/api/predict")
    async def predict_json(
        body: PredictRequest,
        org: dict[str, Any] = Depends(require_api_key),
    ) -> dict[str, Any]:
        """Run prediction using bounding box and date range."""
        if body.start_date and body.end_date and body.start_date > body.end_date:
            raise HTTPException(status_code=400, detail="start_date must be before end_date")

        created_at = _utc_now_iso()
        bbox_json = json.dumps(body.bbox) if body.bbox else None

        with get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs (kind, status, analysis_type, bbox, start_date, end_date, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    body.kind,
                    "running",
                    body.analysis_type,
                    bbox_json,
                    body.start_date,
                    body.end_date,
                    created_at,
                    created_at,
                ),
            )
            run_id = int(cur.lastrowid)

        # Run inference
        try:
            result_payload = run_inference_from_gee(
                bbox=body.bbox,
                start_date=body.start_date,
                end_date=body.end_date,
                analysis_type=body.analysis_type,
            )
            result_payload["analysis_type"] = body.analysis_type
            status = "completed"
        except Exception as exc:
            logger.exception("Inference failed for run %s", run_id)
            result_payload = _load_template_result(
                bbox=body.bbox,
                start_date=body.start_date,
                end_date=body.end_date,
                analysis_type=body.analysis_type,
            )
            result_payload["error"] = str(exc)
            status = "failed"

        # Persist result
        result_created_at = _utc_now_iso()
        with get_connection() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE id = ?",
                (status, result_created_at, run_id),
            )
            conn.execute(
                """
                INSERT INTO results (run_id, payload_json, mask_path, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, json.dumps(result_payload), None, result_created_at),
            )

        return {"run_id": run_id, "result": result_payload}

    @app.post("/api/predict/upload")
    async def predict_upload(
        kind: str = Form(default="upload"),
        org: dict[str, Any] = Depends(require_api_key),
        analysis_type: str = Form(default="deforestation"),
        bbox: str | None = Form(default=None),
        start_date: str | None = Form(default=None),
        end_date: str | None = Form(default=None),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        """Run prediction on uploaded satellite imagery file."""
        if start_date and end_date and start_date > end_date:
            raise HTTPException(status_code=400, detail="start_date must be before end_date")

        created_at = _utc_now_iso()

        parsed_bbox: Optional[list[float]] = None
        if bbox:
            try:
                parsed_bbox = json.loads(bbox)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail="Invalid bbox JSON") from e

        with get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs (kind, status, analysis_type, bbox, start_date, end_date, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    "running",
                    analysis_type,
                    json.dumps(parsed_bbox) if parsed_bbox else None,
                    start_date,
                    end_date,
                    created_at,
                    created_at,
                ),
            )
            run_id = int(cur.lastrowid)

        dest = await _persist_upload(run_id=run_id, file=file)

        # Run inference
        try:
            result_payload = run_inference_from_file(
                dest,
                bbox=parsed_bbox,
                start_date=start_date,
                end_date=end_date,
                analysis_type=analysis_type,
            )
            result_payload["analysis_type"] = analysis_type
            status = "completed"
        except Exception as exc:
            logger.exception("Inference failed for upload run %s", run_id)
            result_payload = _load_template_result(
                bbox=parsed_bbox,
                start_date=start_date,
                end_date=end_date,
                analysis_type=analysis_type,
            )
            result_payload.setdefault("input", {})["file"] = dest
            result_payload["error"] = str(exc)
            status = "failed"

        # Persist result
        result_created_at = _utc_now_iso()
        with get_connection() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE id = ?",
                (status, result_created_at, run_id),
            )
            conn.execute(
                """
                INSERT INTO results (run_id, payload_json, mask_path, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, json.dumps(result_payload), None, result_created_at),
            )

        return {"run_id": run_id, "result": result_payload}

    # ===== Organization (NGO) Endpoints =====

    @app.post("/api/organizations", response_model=OrganizationWithKeyResponse)
    def create_org(
        body: CreateOrganizationRequest,
        org: dict[str, Any] = Depends(require_api_key),
    ) -> dict[str, Any]:
        """Register a new organization. Returns API key (save it securely)."""
        result = create_organization(
            name=body.name,
            org_type=body.type,
            description=body.description,
            contact_email=body.contact_email,
            website_url=body.website_url,
            regions_of_interest=body.regions_of_interest,
        )
        
        # Fetch full organization data
        org = get_organization(result["id"])
        if not org:
            raise HTTPException(status_code=500, detail="Failed to create organization")
        
        return {
            **dict(org),
            "active": bool(org["active"]),
            "api_key": result["api_key"],
        }

    @app.get("/api/organizations")
    def list_orgs(
        type: Optional[str] = None,
        limit: int = Query(default=50, le=200),
    ) -> list[OrganizationResponse]:
        """List all registered organizations."""
        orgs = list_organizations(org_type=type, limit=limit)
        return [
            OrganizationResponse(
                id=org["id"],
                name=org["name"],
                type=org["type"],
                description=org["description"],
                logo_url=org["logo_url"],
                website_url=org["website_url"],
                contact_email=org["contact_email"],
                active=bool(org["active"]),
                created_at=org["created_at"],
            )
            for org in orgs
        ]

    @app.get("/api/organizations/{org_id}")
    def get_org(org_id: int) -> OrganizationResponse:
        """Get organization details by ID."""
        org = get_organization(org_id)
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        return OrganizationResponse(
            id=org["id"],
            name=org["name"],
            type=org["type"],
            description=org["description"],
            logo_url=org["logo_url"],
            website_url=org["website_url"],
            contact_email=org["contact_email"],
            active=bool(org["active"]),
            created_at=org["created_at"],
        )

    # ===== Subscription Endpoints =====

    @app.post("/api/organizations/{org_id}/subscriptions")
    def create_org_subscription(
        org_id: int,
        body: CreateSubscriptionRequest,
        org: dict[str, Any] = Depends(require_api_key),
    ) -> SubscriptionResponse:
        """Create a new region subscription for an organization."""
        org = get_organization(org_id)
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        result = create_subscription(
            organization_id=org_id,
            bbox=body.bbox,
            name=body.name,
            analysis_types=body.analysis_types,
            alert_threshold=body.alert_threshold,
            notification_channel=body.notification_channel,
            webhook_url=body.webhook_url,
        )
        
        return SubscriptionResponse(
            id=result["id"],
            organization_id=org_id,
            name=body.name,
            bbox=body.bbox,
            analysis_types=body.analysis_types,
            alert_threshold=body.alert_threshold,
            notification_channel=body.notification_channel,
            active=True,
            created_at=result["created_at"],
        )

    @app.get("/api/organizations/{org_id}/subscriptions")
    def list_org_subscriptions(org_id: int) -> list[SubscriptionResponse]:
        """List all subscriptions for an organization."""
        org = get_organization(org_id)
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        subs = get_subscriptions_for_organization(org_id)
        results = []
        for sub in subs:
            bbox = json.loads(sub["bbox"]) if sub["bbox"] else []
            analysis_types = json.loads(sub["analysis_types"]) if sub["analysis_types"] else []
            results.append(
                SubscriptionResponse(
                    id=sub["id"],
                    organization_id=sub["organization_id"],
                    name=sub["name"],
                    bbox=bbox,
                    analysis_types=analysis_types,
                    alert_threshold=sub["alert_threshold"],
                    notification_channel=sub["notification_channel"],
                    active=bool(sub["active"]),
                    created_at=sub["created_at"],
                )
            )
        return results

    # ===== Alert Endpoints =====

    @app.get("/api/organizations/{org_id}/alerts")
    def list_org_alerts(
        org_id: int,
        undelivered_only: bool = False,
        unacknowledged_only: bool = False,
        limit: int = Query(default=50, le=200),
    ) -> list[AlertResponse]:
        """List alerts for an organization."""
        org = get_organization(org_id)
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        alerts = get_alerts_for_organization(
            org_id,
            undelivered_only=undelivered_only,
            unacknowledged_only=unacknowledged_only,
            limit=limit,
        )
        
        return [
            AlertResponse(
                id=alert["id"],
                organization_id=alert["organization_id"],
                alert_type=alert["alert_type"],
                severity=alert["severity"],
                title=alert["title"],
                message=alert["message"],
                delivered=bool(alert["delivered"]),
                acknowledged=bool(alert["acknowledged"]),
                created_at=alert["created_at"],
            )
            for alert in alerts
        ]

    @app.post("/api/organizations/{org_id}/alerts")
    def create_org_alert(
        org_id: int,
        body: CreateAlertRequest,
        org: dict[str, Any] = Depends(require_api_key),
    ) -> AlertResponse:
        """Create a new alert for an organization."""
        org = get_organization(org_id)
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        alert_id = create_organization_alert(
            organization_id=org_id,
            alert_type=body.alert_type,
            title=body.title,
            message=body.message,
            severity=body.severity,
            subscription_id=body.subscription_id,
            run_id=body.run_id,
            details=body.details,
        )
        
        return AlertResponse(
            id=alert_id,
            organization_id=org_id,
            alert_type=body.alert_type,
            severity=body.severity,
            title=body.title,
            message=body.message,
            delivered=False,
            acknowledged=False,
            created_at=_utc_now_iso(),
        )

    @app.post("/api/alerts/{alert_id}/acknowledge")
    def acknowledge_org_alert(
        alert_id: int,
        acknowledged_by: Optional[str] = None,
        org: dict[str, Any] = Depends(require_api_key),
    ) -> dict[str, Any]:
        """Acknowledge an alert."""
        success = acknowledge_alert(alert_id, acknowledged_by)
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        return {"success": True, "alert_id": alert_id}

    @app.post("/api/alerts/{alert_id}/deliver")
    def mark_alert_as_delivered(
        alert_id: int,
        org: dict[str, Any] = Depends(require_api_key),
    ) -> dict[str, Any]:
        """Mark an alert as delivered."""
        success = mark_alert_delivered(alert_id)
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        return {"success": True, "alert_id": alert_id}

    # ===== Static Files =====

    # Serve built frontend when dist exists (production mode)
    frontend_dir = Path(__file__).resolve().parents[3] / "frontend"
    dist_dir = frontend_dir / "dist"
    if dist_dir.exists():
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="frontend")

    return app


app = create_app()
