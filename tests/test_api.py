"""Tests for ClimateVision API endpoints."""

from unittest.mock import patch

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


def test_predict_json_accepts_dev_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/predict should accept the cv_dev development key.

    The dev bypass is gated behind CLIMATEVISION_ALLOW_DEV_KEY, so enable it
    explicitly for this test (production leaves it unset/disabled).
    """
    monkeypatch.setenv("CLIMATEVISION_ALLOW_DEV_KEY", "1")
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


def test_predict_valid_date_range_reaches_inference(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/predict with valid date range should reach the inference layer."""
    monkeypatch.setenv("CLIMATEVISION_ALLOW_DEV_KEY", "1")
    payload = {
        "bbox": [-60.0, -15.0, -45.0, -5.0],
        "start_date": "2023-01-01",
        "end_date": "2023-06-30",
        "analysis_type": "deforestation",
    }
    fake_result = {
        "region": {"bbox": payload["bbox"]},
        "inference": {"forest_percentage": 72.3},
        "analysis_type": "deforestation",
    }
    with patch(
        "climatevision.api.main.run_inference_from_gee", return_value=fake_result
    ) as mock_infer:
        response = client.post(
            "/api/predict",
            json=payload,
            headers={"X-API-Key": "cv_dev"},
        )
    assert response.status_code == 200
    mock_infer.assert_called_once()


def test_predict_reversed_date_range_returns_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/predict with start_date > end_date should return 422."""
    monkeypatch.setenv("CLIMATEVISION_ALLOW_DEV_KEY", "1")
    payload = {
        "bbox": [-60.0, -15.0, -45.0, -5.0],
        "start_date": "2026-06-01",
        "end_date": "2026-01-01",
        "analysis_type": "deforestation",
    }
    response = client.post(
        "/api/predict",
        json=payload,
        headers={"X-API-Key": "cv_dev"},
    )
    assert response.status_code == 422
    body = response.json()
    error_messages = [e["msg"] for e in body["detail"]]
    assert any("start_date" in msg or "end_date" in msg for msg in error_messages)


def test_predict_equal_dates_returns_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/predict with start_date == end_date should return 422."""
    monkeypatch.setenv("CLIMATEVISION_ALLOW_DEV_KEY", "1")
    payload = {
        "bbox": [-60.0, -15.0, -45.0, -5.0],
        "start_date": "2023-06-01",
        "end_date": "2023-06-01",
        "analysis_type": "deforestation",
    }
    response = client.post(
        "/api/predict",
        json=payload,
        headers={"X-API-Key": "cv_dev"},
    )
    assert response.status_code == 422
