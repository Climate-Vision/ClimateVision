"""
Integration tests for the SAR flood inference pipeline (bbox -> result).

These run the real synthetic-fallback path end to end (GEE absent), exercising
rasterio GeoTIFF I/O, the ensemble, and the provenance/QA + road-impact blocks.
Requires rasterio; skipped if it is not installed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("rasterio")

from climatevision.inference.flood_pipeline import run_flood_inference_from_gee


BBOX = [36.7, -1.4, 37.0, -1.1]


def _run():
    return run_flood_inference_from_gee(
        bbox=BBOX, start_date="2024-04-01", end_date="2024-04-10",
    )


class TestFloodPipelineProvenance:
    def test_runs_and_tags_synthetic(self):
        res = _run()
        assert res["success"] is True
        assert res["analysis_type"] == "flooding_sar"
        # GEE is unavailable in tests -> synthetic fallback, explicitly tagged.
        assert res["is_synthetic"] is True
        assert res["quality"]["is_synthetic"] is True

    def test_quality_block_is_honest_without_reference(self):
        q = _run()["quality"]
        # No JRC reference in the test env -> split not made, flagged clearly.
        assert q["permanent_flood_distinguished"] is False
        assert q["permanent_water_reference_source"] == "none"
        assert q["validated_against_benchmark"] is False
        assert any("not yet validated" in n.lower() for n in q["notes"])
        assert any("synthetic" in n.lower() for n in q["notes"])

    def test_road_impact_degrades_when_osmnx_absent(self):
        res = _run()
        ri = res["metadata"]["road_impact"]
        # osmnx is not a test dependency; impact must degrade, not fabricate.
        assert ri["road_impact_available"] is False
        assert ri["affected_road_km"] is None
        assert res["quality"]["road_impact_available"] is False

    def test_inference_metrics_present(self):
        inf = _run()["inference"]
        assert "flooded_percentage" in inf
        assert "flooded_area_km2" in inf
