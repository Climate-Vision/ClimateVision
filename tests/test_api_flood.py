"""
End-to-end API tests for flood detection endpoints.
"""
from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient

from climatevision.api.main import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestFloodPrediction:
    def test_health_endpoint(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "flooding" in data["analysis_types"]

    def test_predict_without_api_key(self, client):
        """Should return 401 without API key."""
        response = client.post(
            "/api/predict",
            json={
                "kind": "gee",
                "analysis_type": "flooding",
                "bbox": [36.7, -1.4, 37.0, -1.1],
                "start_date": "2024-04-01",
                "end_date": "2024-04-10",
            },
        )
        assert response.status_code == 401

    def test_predict_flooding_analysis_type_exists(self, client):
        """Flooding should be listed as an enabled analysis type."""
        response = client.get("/api/analysis-types")
        assert response.status_code == 200
        types = response.json()
        flooding = next((t for t in types if t["name"] == "flooding"), None)
        assert flooding is not None
        assert flooding["enabled"] is True
        assert flooding["bands"] == ["B03", "B08", "B11"]
