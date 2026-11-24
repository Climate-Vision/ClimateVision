"""
Alert endpoints: listing, creation, acknowledgement and delivery of alerts.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from climatevision.api.auth import require_api_key
from climatevision.api.shared import AlertSeverity, _utc_now_iso
from climatevision.db import (
    acknowledge_alert,
    create_organization_alert,
    get_alerts_for_organization,
    get_organization,
    get_pending_alerts,
    mark_alert_delivered,
)

router = APIRouter()


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


# ===== Alert Endpoints =====


@router.get("/api/organizations/{org_id}/alerts")
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


@router.post("/api/organizations/{org_id}/alerts")
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


@router.post("/api/alerts/{alert_id}/acknowledge")
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


@router.post("/api/alerts/{alert_id}/deliver")
def mark_alert_as_delivered(
    alert_id: int,
    org: dict[str, Any] = Depends(require_api_key),
) -> dict[str, Any]:
    """Mark an alert as delivered."""
    success = mark_alert_delivered(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"success": True, "alert_id": alert_id}


@router.get("/api/organizations/{org_id}/alerts/pending")
def list_pending_alerts(
    org_id: int,
    limit: int = Query(default=50, le=200),
) -> list[AlertResponse]:
    """List pending (undelivered) alerts for an organization.

    This endpoint is designed for monitoring and dashboard use,
    showing alerts that the delivery worker has not yet processed.
    """
    org = get_organization(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    alerts = get_pending_alerts(max_attempts=3, limit=limit)
    # Filter to this specific organization's alerts
    org_alerts = [a for a in alerts if a["organization_id"] == org_id]

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
        for alert in org_alerts[:limit]
    ]
