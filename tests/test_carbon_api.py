"""Tests for carbon analytics wiring and the impact report endpoint."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from climatevision.analytics.carbon import estimate_carbon_loss

# A 100x100 patch of 10 m pixels = 10,000 px = 100 ha. With tropical_moist
# defaults these values are fully determined:
#   hectares        = 10000 * (10^2 / 10000)        = 100.0
#   agb             = 100 * 300.0 * 1.0             = 30000.0
#   bgb             = 30000.0 * 0.24                = 7200.0
#   carbon_tonnes   = (30000 + 7200) * 0.47         = 17484.0
#   co2_equivalent  = 17484.0 * 44 / 12             = 64108.0
KNOWN_PIXELS = 10_000
EXPECTED_HECTARES = 100.0
EXPECTED_CARBON = 17_484.0
EXPECTED_CO2 = 64_108.0


def _deforestation_payload(pixels: int = KNOWN_PIXELS) -> dict:
    return {
        "region": {"bbox": [-60.0, -15.0, -45.0, -5.0]},
        "inference": {
            "image_size": [256, 256],
            "forest_pixels": 55_536,
            "non_forest_pixels": pixels,
            "forest_percentage": 84.7,
            "mean_confidence": 0.91,
        },
        "analysis_type": "deforestation",
    }


def test_carbon_math_matches_known_values() -> None:
    """estimate_carbon_loss must reproduce hand-computed IPCC figures."""
    result = estimate_carbon_loss(deforested_pixels=KNOWN_PIXELS)
    assert result["hectares"] == EXPECTED_HECTARES
    assert result["carbon_tonnes"] == EXPECTED_CARBON
    assert result["co2_equivalent"] == EXPECTED_CO2


def test_predict_without_flag_omits_carbon(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Carbon block is absent unless enable_carbon is set."""
    monkeypatch.setenv("CLIMATEVISION_ALLOW_DEV_KEY", "1")
    with patch(
        "climatevision.api.main.run_inference_from_gee",
        return_value=_deforestation_payload(),
    ):
        response = client.post(
            "/api/predict",
            json={
                "bbox": [-60.0, -15.0, -45.0, -5.0],
                "analysis_type": "deforestation",
            },
            headers={"X-API-Key": "cv_dev"},
        )
    assert response.status_code == 200
    assert "carbon_estimation" not in response.json()["result"]


def test_predict_with_flag_includes_carbon(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """enable_carbon adds a carbon_estimation block with correct figures."""
    monkeypatch.setenv("CLIMATEVISION_ALLOW_DEV_KEY", "1")
    with patch(
        "climatevision.api.main.run_inference_from_gee",
        return_value=_deforestation_payload(),
    ):
        response = client.post(
            "/api/predict",
            json={
                "bbox": [-60.0, -15.0, -45.0, -5.0],
                "analysis_type": "deforestation",
                "enable_carbon": True,
            },
            headers={"X-API-Key": "cv_dev"},
        )
    assert response.status_code == 200
    carbon = response.json()["result"]["carbon_estimation"]
    assert carbon["hectares_lost"] == EXPECTED_HECTARES
    assert carbon["carbon_tonnes"] == EXPECTED_CARBON
    assert carbon["co2_equivalent"] == EXPECTED_CO2


def test_report_endpoint_returns_impact_report(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /api/reports/{run_id} returns a complete impact report."""
    monkeypatch.setenv("CLIMATEVISION_ALLOW_DEV_KEY", "1")
    with patch(
        "climatevision.api.main.run_inference_from_gee",
        return_value=_deforestation_payload(),
    ):
        predict = client.post(
            "/api/predict",
            json={
                "bbox": [-60.0, -15.0, -45.0, -5.0],
                "analysis_type": "deforestation",
            },
            headers={"X-API-Key": "cv_dev"},
        )
    run_id = predict.json()["run_id"]

    report = client.get(f"/api/reports/{run_id}")
    assert report.status_code == 200
    body = report.json()

    assert body["run_id"] == run_id
    assert body["hectares_lost"] == EXPECTED_HECTARES
    assert body["carbon_tonnes"] == EXPECTED_CARBON
    assert body["co2_equivalent"] == EXPECTED_CO2
    assert body["region_bbox"] == [-60.0, -15.0, -45.0, -5.0]

    ci = body["confidence_interval"]
    assert {"lower", "upper", "uncertainty_pct"} <= ci.keys()
    # The Monte Carlo band is computed over CO2-equivalent, so the CO2 point
    # estimate must fall within it.
    assert ci["lower"] <= body["co2_equivalent"] <= ci["upper"]


def test_report_endpoint_unknown_run_returns_404(client: TestClient) -> None:
    """A report for a nonexistent run returns 404."""
    response = client.get("/api/reports/999999")
    assert response.status_code == 404
