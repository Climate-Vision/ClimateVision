"""
Subscription endpoints: regions an organization monitors.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from climatevision.api.auth import require_api_key
from climatevision.api.shared import AnalysisType, NotificationChannel
from climatevision.db import (
    create_subscription,
    get_organization,
    get_subscriptions_for_organization,
)

router = APIRouter()


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


# ===== Subscription Endpoints =====


@router.post("/api/organizations/{org_id}/subscriptions")
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


@router.get("/api/organizations/{org_id}/subscriptions")
def list_org_subscriptions(org_id: int) -> list[SubscriptionResponse]:
    """List all subscriptions for an organization."""
    org = get_organization(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    subs = get_subscriptions_for_organization(org_id)
    results = []
    for sub in subs:
        bbox = json.loads(sub["bbox"]) if sub["bbox"] else []
        analysis_types = (
            json.loads(sub["analysis_types"]) if sub["analysis_types"] else []
        )
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
