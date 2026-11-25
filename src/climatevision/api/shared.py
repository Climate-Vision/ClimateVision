"""
Types and helpers shared by the ClimateVision API route modules.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import Header, HTTPException

from climatevision.api.run_events import build_status_event
from climatevision.db import get_organization_by_api_key

logger = logging.getLogger(__name__)

# Repository root, which holds outputs/, models/ and frontend/.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ===== Type Definitions =====

AnalysisType = Literal[
    "deforestation", "ice_melting", "flooding", "flooding_sar", "drought", "wildfire"
]
OrganizationType = Literal["ngo", "government", "research", "corporate"]
NotificationChannel = Literal["email", "webhook", "api", "sms"]
AlertSeverity = Literal["low", "medium", "high", "critical"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_event_for(
    run_id: int, status: str, result_payload: Optional[dict[str, Any]]
) -> dict[str, Any]:
    """Build a status event, routing the payload to ``result`` or ``error``.

    A failed run stores its message under ``error`` inside the result payload,
    so surface that as the event's ``error`` field rather than as a result.

    Args:
        run_id: The run the event describes.
        status: The run's current status.
        result_payload: The stored result payload, if any.

    Returns:
        The event dictionary to send over the WebSocket.
    """
    if status == "failed":
        error = None
        if result_payload is not None:
            error = result_payload.get("error")
        return build_status_event(run_id, status, error=error or "Inference failed")
    if status == "completed":
        return build_status_event(run_id, status, result=result_payload)
    return build_status_event(run_id, status)


# ===== Carbon analytics helpers =====


def _extract_deforested_pixels(payload: dict[str, Any]) -> Optional[int]:
    """Pull the deforested-pixel count from a deforestation result payload.

    Returns None when the payload does not carry an inference pixel count.
    """
    inference = payload.get("inference")
    if not isinstance(inference, dict):
        return None
    pixels = inference.get("non_forest_pixels")
    if pixels is None:
        return None
    try:
        return int(pixels)
    except (TypeError, ValueError):
        return None


def _compute_carbon_estimation(
    payload: dict[str, Any],
    forest_type: Optional[str] = None,
    region: Optional[str] = None,
) -> Optional[dict[str, float]]:
    """Deterministic carbon loss estimate for embedding in a predict response.

    Returns None if the carbon module is unavailable or the payload lacks a
    pixel count, so the caller can degrade gracefully.
    """
    pixels = _extract_deforested_pixels(payload)
    if pixels is None:
        return None
    try:
        from climatevision.analytics.carbon import estimate_carbon_loss

        estimate = estimate_carbon_loss(
            deforested_pixels=pixels,
            forest_type=forest_type or "tropical_moist",
            region=region or "default",
        )
    except Exception:
        logger.exception("Carbon estimation failed")
        return None

    return {
        "carbon_tonnes": estimate["carbon_tonnes"],
        "hectares_lost": estimate["hectares"],
        "co2_equivalent": estimate["co2_equivalent"],
        "forest_type": forest_type or "tropical_moist",
        "region": region or "default",
    }


def _validate_forest_type(forest_type: Optional[str]) -> None:
    """Reject an unknown forest_type instead of silently defaulting.

    A typo such as ``tropical-moist`` (hyphen) would otherwise fall through
    to a generic biomass density and return plausible-looking but wrong
    numbers with a 200 OK.
    """
    if forest_type is None:
        return
    from climatevision.analytics.carbon import AGB_DENSITY

    if forest_type not in AGB_DENSITY:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown forest_type '{forest_type}'. "
                f"Valid values: {sorted(AGB_DENSITY)}"
            ),
        )


def _validate_region(region: Optional[str]) -> None:
    """Reject an unknown region instead of silently applying no adjustment.

    An unmatched region (e.g. ``Amazon`` capitalised, or ``amazon `` with a
    trailing space) would otherwise fall back to a 1.0 factor and return a
    confident-looking number that is wrong by the real regional adjustment.
    Since these figures are meant to be cited, fail loudly instead.
    """
    if region is None:
        return
    from climatevision.analytics.carbon import REGIONAL_FACTORS

    if region not in REGIONAL_FACTORS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown region '{region}'. "
                f"Valid values: {sorted(REGIONAL_FACTORS)}"
            ),
        )


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
