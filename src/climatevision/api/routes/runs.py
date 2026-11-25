"""
Run endpoints: listing, statistics, details, live status and impact reports.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from climatevision.api.run_events import run_event_hub
from climatevision.api.shared import (
    _extract_deforested_pixels,
    _status_event_for,
    _validate_forest_type,
    _validate_region,
)
from climatevision.db import get_connection

logger = logging.getLogger(__name__)

router = APIRouter()


# ===== Run Status Streaming =====

# Statuses after which no further updates are expected, so the socket closes.
TERMINAL_RUN_STATUSES = frozenset({"completed", "failed"})

# Application-defined close code for "the requested run does not exist".
# The 4000-4999 range is reserved for application use by RFC 6455.
WS_RUN_NOT_FOUND = 4404


def _read_run_status(
    run_id: int,
) -> Optional[tuple[str, Optional[dict[str, Any]]]]:
    """Read a run's current status and latest result payload.

    Args:
        run_id: The run to read.

    Returns:
        A ``(status, result_payload)`` pair, or ``None`` if no such run exists.
        The payload is ``None`` when the run has not produced a result yet.
    """
    with get_connection() as conn:
        run = conn.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            return None
        result = conn.execute(
            "SELECT payload_json FROM results WHERE run_id = ? ORDER BY id DESC LIMIT 1",
            (run_id,),
        ).fetchone()

    payload: Optional[dict[str, Any]] = None
    if result is not None:
        payload = json.loads(result["payload_json"])
    return run["status"], payload


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


# ===== Run Endpoints =====


@router.get("/api/runs")
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


@router.get("/api/runs/stats")
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

        # Alerts are created and acknowledged in organization_alerts; the
        # legacy alerts table has no acknowledged column.
        alerts_total = conn.execute(
            "SELECT COUNT(*) FROM organization_alerts"
        ).fetchone()[0]
        alerts_unacknowledged = conn.execute(
            "SELECT COUNT(*) FROM organization_alerts WHERE acknowledged = 0"
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


@router.get("/api/runs/{run_id}")
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
        "result": (
            None
            if result is None
            else {
                "id": result["id"],
                "run_id": result["run_id"],
                "payload": payload,
                "mask_path": mask_path,
                "created_at": result["created_at"],
            }
        ),
    }


@router.websocket("/ws/runs/{run_id}")
async def watch_run(websocket: WebSocket, run_id: int) -> None:
    """Stream status updates for a single run.

    On connect the current status is read from the database and sent
    immediately, so a client attaching to an already-finished run still
    receives a terminal event. If the run is still in progress the socket
    then streams transitions until it reaches ``completed``/``failed``,
    and closes once a terminal status has been delivered.
    """
    await websocket.accept()

    # Subscribe before the initial read, otherwise a transition happening
    # between the read and the subscribe would be missed entirely.
    async with run_event_hub.subscribe(run_id) as queue:
        snapshot = _read_run_status(run_id)
        if snapshot is None:
            await websocket.send_json(
                {"type": "error", "run_id": run_id, "error": "Run not found"}
            )
            await websocket.close(code=WS_RUN_NOT_FOUND)
            return

        status, result_payload = snapshot
        await websocket.send_json(_status_event_for(run_id, status, result_payload))
        if status in TERMINAL_RUN_STATUSES:
            await websocket.close()
            return

        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event)
                if event.get("status") in TERMINAL_RUN_STATUSES:
                    break
        except WebSocketDisconnect:
            # Client went away mid-run; the subscription is released by
            # the context manager, nothing else to clean up.
            logger.debug("WebSocket client disconnected from run %s", run_id)
            return

        await websocket.close()


@router.get("/api/reports/{run_id}")
def get_report(run_id: int) -> dict[str, Any]:
    """Structured carbon impact report for a completed deforestation run.

    Recomputes the estimate (with uncertainty bounds) from the stored
    pixel count, so it works whether or not carbon was requested at
    prediction time.
    """
    with get_connection() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        result = conn.execute(
            "SELECT * FROM results WHERE run_id = ? ORDER BY id DESC LIMIT 1", (run_id,)
        ).fetchone()

    if result is None:
        raise HTTPException(status_code=404, detail="No result available for run")

    payload = json.loads(result["payload_json"])
    if payload.get("analysis_type") != "deforestation":
        raise HTTPException(
            status_code=400,
            detail="Impact reports are only available for deforestation runs",
        )

    pixels = _extract_deforested_pixels(payload)
    if pixels is None:
        raise HTTPException(
            status_code=422, detail="Result payload has no pixel count to report on"
        )

    # Reuse the forest_type/region chosen at prediction time so the report
    # and the /api/predict response describe the same run. Falling back to
    # the estimator defaults keeps reports working for runs saved before
    # carbon was requested.
    carbon_meta = payload.get("carbon_estimation") or {}
    forest_type = carbon_meta.get("forest_type", "tropical_moist")
    region = carbon_meta.get("region", "default")
    _validate_forest_type(forest_type)
    _validate_region(region)

    try:
        from climatevision.analytics.carbon import CarbonEstimator

        estimator = CarbonEstimator(forest_type=forest_type, region=region)
        estimate = estimator.estimate_from_pixel_count(pixels)
    except Exception as exc:
        logger.exception("Impact report generation failed for run %s", run_id)
        raise HTTPException(
            status_code=503, detail="Carbon analytics unavailable"
        ) from exc

    region_bbox = json.loads(run["bbox"]) if run["bbox"] else None

    return {
        "run_id": run_id,
        "hectares_lost": estimate.hectares,
        "biomass_tonnes": estimate.biomass_tonnes,
        "carbon_tonnes": estimate.carbon_tonnes,
        "co2_equivalent": estimate.co2_equivalent,
        "confidence_interval": {
            "lower": estimate.ci_lower,
            "upper": estimate.ci_upper,
            "uncertainty_pct": estimate.uncertainty_pct,
            "unit": "tCO2e",
        },
        "region_bbox": region_bbox,
        "forest_type": estimate.forest_type,
        "region": estimate.region,
    }
