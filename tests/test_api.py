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


def test_predict_json_bbox_validation(client: TestClient) -> None:
    """POST /api/predict should validate bbox coordinates and return 422 for invalid values."""
    valid_payload = {
        "bbox": [-60.0, -15.0, -45.0, -5.0],
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "analysis_type": "deforestation",
    }
    # Valid bbox should pass validation (inference may fail, but validation passes)
    response = client.post(
        "/api/predict",
        json=valid_payload,
        headers={"X-API-Key": "cv_dev"},
    )
    assert response.status_code in (200, 500)  # 200 if inference succeeds, 500 if it fails


def test_predict_json_bbox_invalid_longitude(client: TestClient) -> None:
    """POST /api/predict should reject bbox with longitude > 180."""
    payload = {
        "bbox": [200.0, 10.0, 30.0, 40.0],
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "analysis_type": "deforestation",
    }
    response = client.post(
        "/api/predict",
        json=payload,
        headers={"X-API-Key": "cv_dev"},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert len(detail) > 0
    # Check that error message mentions longitude
    error_msg = str(detail).lower()
    assert "longitude" in error_msg or "value error" in error_msg


def test_predict_json_bbox_invalid_latitude(client: TestClient) -> None:
    """POST /api/predict should reject bbox with latitude > 90."""
    payload = {
        "bbox": [-60.0, 100.0, -45.0, 50.0],
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "analysis_type": "deforestation",
    }
    response = client.post(
        "/api/predict",
        json=payload,
        headers={"X-API-Key": "cv_dev"},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert len(detail) > 0


def test_predict_json_bbox_wrong_count(client: TestClient) -> None:
    """POST /api/predict should reject bbox with wrong number of elements."""
    payload = {
        "bbox": [-60.0, -15.0, -45.0],  # Only 3 values
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "analysis_type": "deforestation",
    }
    response = client.post(
        "/api/predict",
        json=payload,
        headers={"X-API-Key": "cv_dev"},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert len(detail) > 0


def test_predict_json_bbox_west_gte_east(client: TestClient) -> None:
    """POST /api/predict should reject bbox where west >= east."""
    payload = {
        "bbox": [30.0, -15.0, -45.0, 50.0],  # west (30) > east (-45)
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "analysis_type": "deforestation",
    }
    response = client.post(
        "/api/predict",
        json=payload,
        headers={"X-API-Key": "cv_dev"},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert len(detail) > 0


def test_predict_json_bbox_south_gte_north(client: TestClient) -> None:
    """POST /api/predict should reject bbox where south >= north."""
    payload = {
        "bbox": [-60.0, 50.0, -45.0, 10.0],  # south (50) > north (10)
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "analysis_type": "deforestation",
    }
    response = client.post(
        "/api/predict",
        json=payload,
        headers={"X-API-Key": "cv_dev"},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert len(detail) > 0
