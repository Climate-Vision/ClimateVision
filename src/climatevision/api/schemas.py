"""
Pydantic schemas for ClimateVision API request/response validation.

Provides strict type validation and serialization for all API endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, EmailStr, field_validator


# ===== Type Definitions =====

AnalysisType = Literal["deforestation", "ice_melting", "flooding", "drought", "wildfire"]
OrganizationType = Literal["ngo", "government", "research", "corporate"]
NotificationChannel = Literal["email", "webhook", "api", "sms"]
AlertSeverity = Literal["low", "medium", "high", "critical"]


# ===== Request Schemas =====

class BoundingBox(BaseModel):
    """Geographic bounding box for analysis region."""
    min_lon: float = Field(..., ge=-180, le=180, description="Minimum longitude")
    min_lat: float = Field(..., ge=-90, le=90, description="Minimum latitude")
    max_lon: float = Field(..., ge=-180, le=180, description="Maximum longitude")
    max_lat: float = Field(..., ge=-90, le=90, description="Maximum latitude")

    @field_validator("max_lon")
    @classmethod
    def validate_lon_range(cls, v: float, info) -> float:
        if "min_lon" in info.data and v <= info.data["min_lon"]:
            raise ValueError("max_lon must be greater than min_lon")
        return v

    @field_validator("max_lat")
    @classmethod
    def validate_lat_range(cls, v: float, info) -> float:
        if "min_lat" in info.data and v <= info.data["min_lat"]:
            raise ValueError("max_lat must be greater than min_lat")
        return v


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoints."""
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box [min_lon, min_lat, max_lon, max_lat]"
    )
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    analysis_type: AnalysisType = Field(..., description="Type of analysis to perform")

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 values")
        min_lon, min_lat, max_lon, max_lat = v
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("Invalid bounding box: min values must be less than max values")
        return v


class OrganizationCreate(BaseModel):
    """Request schema for creating an organization."""
    name: str = Field(..., min_length=2, max_length=255, description="Organization name")
    email: EmailStr = Field(..., description="Contact email")
    org_type: OrganizationType = Field(default="ngo", description="Organization type")
    region: Optional[str] = Field(None, max_length=100, description="Primary region of interest")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for notifications")


class SubscriptionCreate(BaseModel):
    """Request schema for creating an alert subscription."""
    bbox: list[float] = Field(..., min_length=4, max_length=4)
    analysis_type: AnalysisType
    alert_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    notification_channels: list[NotificationChannel] = Field(default=["email"])


# ===== Response Schemas =====

class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str
    version: str
    timestamp: datetime
    model_loaded: bool


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoints."""
    success: bool
    analysis_type: str
    region: dict[str, Any]
    inference: dict[str, Any]
    alerts: list[dict[str, Any]] = Field(default_factory=list)
    request_id: Optional[str] = None
    processing_time_ms: Optional[float] = None


class OrganizationResponse(BaseModel):
    """Response schema for organization endpoints."""
    id: int
    name: str
    email: str
    org_type: str
    api_key: str
    region: Optional[str]
    created_at: datetime


class AlertResponse(BaseModel):
    """Response schema for alert endpoints."""
    id: int
    organization_id: int
    alert_type: str
    severity: AlertSeverity
    title: str
    message: str
    acknowledged: bool
    delivered: bool
    created_at: datetime


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
