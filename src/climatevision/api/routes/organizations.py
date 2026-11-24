"""
Organization (NGO) endpoints: registration and lookup.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr, Field

from climatevision.api.auth import require_api_key
from climatevision.api.shared import OrganizationType
from climatevision.db import create_organization, get_organization, list_organizations

router = APIRouter()


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


# ===== Organization (NGO) Endpoints =====


@router.post("/api/organizations", response_model=OrganizationWithKeyResponse)
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


@router.get("/api/organizations")
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


@router.get("/api/organizations/{org_id}")
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
