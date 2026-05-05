"""
Admin endpoints for ClimateVision operational reporting.

Exposes two read-only endpoints intended for the operational dashboard
and on-call tooling:

- ``GET /api/reports`` — data-quality KPIs for a configurable time window
  (run count, error rate, mean confidence, alert count).
- ``GET /api/anomalies`` — list of flagged anomaly predictions, optionally
  filtered by severity and time window.

Both endpoints read from JSONL files written by the audit logger and the
anomaly detector. They never mutate state and never expose raw input
payloads — only summary fields safe for an operations dashboard.

The router is wired into the FastAPI app via ``include_router(admin.router)``
in ``api/main.py``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUDIT_LOG = _PROJECT_ROOT / "outputs" / "audit" / "predictions.jsonl"
DEFAULT_ANOMALY_LOG = _PROJECT_ROOT / "outputs" / "anomalies" / "history.jsonl"
DEFAULT_ALERT_LOG = _PROJECT_ROOT / "outputs" / "alerts" / "alerts.jsonl"


router = APIRouter(prefix="/api", tags=["admin"])


class ReportSummary(BaseModel):
    window_hours: int = Field(..., description="Time window in hours")
    run_count: int = Field(..., description="Predictions logged in window")
    error_rate: float = Field(..., description="Fraction of runs with non-OK status")
    mean_confidence: Optional[float] = Field(None, description="Mean confidence over window")
    positive_fraction_mean: Optional[float] = Field(None)
    alert_count: int = Field(0, description="Alerts fired in window")
    generated_at: str


class AnomalyRecord(BaseModel):
    triggered_at: Optional[str] = None
    severity: Optional[str] = None
    method: Optional[str] = None
    score: Optional[float] = None
    reasons: list[str] = Field(default_factory=list)
    summary: Optional[str] = None


class AnomalyList(BaseModel):
    count: int
    anomalies: list[AnomalyRecord]


def _read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return iter(())
    def _it() -> Iterator[dict]:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("skipping malformed line in %s", path)
    return _it()


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _within_window(ts: Optional[datetime], cutoff: datetime) -> bool:
    return ts is not None and ts >= cutoff


def build_report_summary(
    window_hours: int,
    audit_log: Optional[Path] = None,
    alert_log: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> ReportSummary:
    if window_hours <= 0:
        raise ValueError("window_hours must be positive")
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=window_hours)
    audit_log = audit_log or DEFAULT_AUDIT_LOG
    alert_log = alert_log or DEFAULT_ALERT_LOG

    runs = []
    for row in _read_jsonl(audit_log):
        ts = _parse_timestamp(row.get("timestamp"))
        if _within_window(ts, cutoff):
            runs.append(row)

    confidence_values = [
        r["output_summary"]["mean_confidence"]
        for r in runs
        if isinstance(r.get("output_summary"), dict)
        and r["output_summary"].get("mean_confidence") is not None
    ]
    positive_values = [
        r["output_summary"]["positive_fraction"]
        for r in runs
        if isinstance(r.get("output_summary"), dict)
        and r["output_summary"].get("positive_fraction") is not None
    ]
    error_count = sum(1 for r in runs if r.get("error"))

    alerts = [
        row
        for row in _read_jsonl(alert_log)
        if _within_window(_parse_timestamp(row.get("triggered_at")), cutoff)
    ]

    return ReportSummary(
        window_hours=window_hours,
        run_count=len(runs),
        error_rate=(error_count / len(runs)) if runs else 0.0,
        mean_confidence=(
            sum(confidence_values) / len(confidence_values) if confidence_values else None
        ),
        positive_fraction_mean=(
            sum(positive_values) / len(positive_values) if positive_values else None
        ),
        alert_count=len(alerts),
        generated_at=now.isoformat(),
    )


def list_anomalies(
    severity: Optional[str] = None,
    window_hours: Optional[int] = None,
    alert_log: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> AnomalyList:
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=window_hours) if window_hours else None
    alert_log = alert_log or DEFAULT_ALERT_LOG

    out: list[AnomalyRecord] = []
    for row in _read_jsonl(alert_log):
        if severity and row.get("severity") != severity:
            continue
        ts = _parse_timestamp(row.get("triggered_at"))
        if cutoff is not None and not _within_window(ts, cutoff):
            continue
        out.append(
            AnomalyRecord(
                triggered_at=row.get("triggered_at"),
                severity=row.get("severity"),
                method=row.get("method"),
                score=row.get("score"),
                reasons=row.get("reasons") or [],
                summary=row.get("summary"),
            )
        )
    return AnomalyList(count=len(out), anomalies=out)


@router.get("/reports", response_model=ReportSummary)
def get_reports(
    window_hours: int = Query(24, gt=0, le=24 * 30 * 6),
) -> ReportSummary:
    """Data-quality KPIs over a configurable time window."""
    try:
        return build_report_summary(window_hours=window_hours)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/anomalies", response_model=AnomalyList)
def get_anomalies(
    severity: Optional[str] = Query(None, pattern="^(low|medium|high|critical)$"),
    window_hours: Optional[int] = Query(None, gt=0, le=24 * 30 * 6),
) -> AnomalyList:
    """List flagged anomaly/alert records, optionally filtered."""
    return list_anomalies(severity=severity, window_hours=window_hours)
