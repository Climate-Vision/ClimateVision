"""
Tests for the SAR flood analysis type, its registry wiring, and API exposure.

The full GEE -> rasterio download path is not exercised here (it needs GDAL and
Earth Engine credentials); those are integration concerns. These tests cover the
analysis logic on numpy arrays plus discoverability through the registry/API.
"""
from __future__ import annotations

import numpy as np
import pytest

from climatevision.analysis import get_analysis_type, list_analysis_types
from climatevision.analysis.flooding_sar import FloodingSARAnalysis, FLOODED, PERMANENT_WATER


def _sar_scene(size: int = 24) -> np.ndarray:
    """(2, H, W) VV/VH dB scene with a central water blob (~ -26 dB VH)."""
    rng = np.random.default_rng(0)
    vv = rng.normal(-9.0, 1.0, (size, size)).astype(np.float32)
    vh = rng.normal(-15.0, 1.0, (size, size)).astype(np.float32)
    vh[6:18, 6:18] = -26.0
    vv[6:18, 6:18] = -20.0
    return np.stack([vv, vv * 0 + vh], axis=0)


class TestFloodingSARAnalysis:
    def test_preprocess_returns_two_band_db(self):
        scene = _sar_scene()
        out = FloodingSARAnalysis().preprocess(scene)
        assert out.shape == scene.shape  # (2, H, W) preserved

    def test_without_reference_water_is_flooded_and_flagged(self):
        scene = _sar_scene()
        analysis = FloodingSARAnalysis()
        pred, conf = analysis.run_inference(analysis.preprocess(scene))
        assert (pred == FLOODED).any()
        assert (pred == PERMANENT_WATER).sum() == 0  # cannot resolve permanent
        metrics = analysis.calculate_metrics(pred, scene.shape[-2:])
        assert metrics["permanent_flood_distinguished"] is False

    def test_with_reference_separates_permanent_and_flood(self):
        scene = _sar_scene()
        # Half the water blob is "normally water".
        ref = np.zeros(scene.shape[-2:], dtype=np.uint8)
        ref[6:12, 6:18] = 1
        analysis = FloodingSARAnalysis(permanent_water_ref=ref)
        pred, conf = analysis.run_inference(analysis.preprocess(scene))
        assert (pred == PERMANENT_WATER).any()
        assert (pred == FLOODED).any()
        metrics = analysis.calculate_metrics(pred, scene.shape[-2:])
        assert metrics["permanent_flood_distinguished"] is True

    def test_metrics_area_with_bbox(self):
        scene = _sar_scene()
        analysis = FloodingSARAnalysis()
        pred, _ = analysis.run_inference(analysis.preprocess(scene))
        metrics = analysis.calculate_metrics(pred, scene.shape[-2:], bbox=[36.7, -1.4, 37.0, -1.1])
        assert "flooded_area_km2" in metrics
        assert metrics["flooded_area_km2"] >= 0

    def test_alerts_critical(self):
        analysis = FloodingSARAnalysis()
        alerts = analysis.generate_alerts({"flooded_percentage": 30.0, "flooded_area_km2": 12.0})
        assert len(alerts) == 1
        assert alerts[0].severity.value == "critical"

    def test_full_analyze_pipeline(self):
        scene = _sar_scene()
        result = FloodingSARAnalysis().analyze(image=scene, bbox=[36.7, -1.4, 37.0, -1.1])
        assert result.success
        assert result.analysis_type == "flooding_sar"
        assert "flooded_percentage" in result.metrics


class TestRegistryAndDiscovery:
    def test_registered_in_registry(self):
        analysis = get_analysis_type("flooding_sar")
        assert analysis is not None
        assert isinstance(analysis, FloodingSARAnalysis)

    def test_listed_among_analysis_types(self):
        names = [t["name"] for t in list_analysis_types()]
        assert "flooding_sar" in names


class TestApiExposure:
    def test_health_ok_and_lists_flooding_sar(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"  # config entry keeps validation green
        assert "flooding_sar" in data["analysis_types"]

    def test_analysis_types_endpoint_includes_sar(self, client):
        resp = client.get("/api/analysis-types")
        assert resp.status_code == 200
        sar = next((t for t in resp.json() if t["name"] == "flooding_sar"), None)
        assert sar is not None
        assert sar["bands"] == ["VV", "VH"]

    def test_predict_flooding_sar_accepted_by_schema(self, client):
        """A flooding_sar request must reach auth (401), not be rejected as an
        invalid analysis_type (422) -- proving the schema accepts it."""
        resp = client.post(
            "/api/predict",
            json={
                "kind": "gee",
                "analysis_type": "flooding_sar",
                "bbox": [36.7, -1.4, 37.0, -1.1],
                "start_date": "2024-04-01",
                "end_date": "2024-04-10",
            },
        )
        assert resp.status_code == 401
