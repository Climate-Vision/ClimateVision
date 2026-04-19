"""Tests for ClimateVision API endpoints."""

import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient) -> None:
    """Health check should return 200 without auth."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")


def test_predict_json_rejects_missing_auth(client: TestClient) -> None:
    """POST /api/predict should reject requests without API key."""
    payload = {
        "bbox": [-60.0, -15.0, -45.0, -5.0],
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "analysis_type": "deforestation",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 401
    assert "API key required" in response.json()["detail"]


def test_predict_json_accepts_dev_key(client: TestClient) -> None:
    """POST /api/predict should accept the cv_dev development key."""
    payload = {
        "bbox": [-60.0, -15.0, -45.0, -5.0],
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "analysis_type": "deforestation",
    }
    response = client.post(
        "/api/predict",
        json=payload,
        headers={"X-API-Key": "cv_dev"},
    )
    # Should pass auth; inference may fail due to missing models/GEE
    assert response.status_code in (200, 500)
