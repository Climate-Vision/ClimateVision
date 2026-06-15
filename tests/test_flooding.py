"""
Tests for flood detection analysis module.
"""
from __future__ import annotations

import numpy as np
import pytest

from climatevision.analysis.flooding import FloodingAnalysis


class TestFloodingAnalysis:
    def test_preprocess_normalizes_image(self):
        analysis = FloodingAnalysis()
        img = np.random.randint(0, 255, (256, 256, 3)).astype(np.float32)
        out = analysis.preprocess(img)
        assert out.dtype == np.float32
        assert out.shape == (256, 256, 3)
        assert out.max() <= 1.0

    def test_water_index_classification(self):
        analysis = FloodingAnalysis()
        # Create synthetic image: strong water signature
        img = np.zeros((256, 256, 3), dtype=np.float32)
        img[..., 0] = 0.8  # Green (high)
        img[..., 2] = 0.1  # SWIR (low)
        pred, conf = analysis._water_index_classification(img)
        assert pred.shape == (256, 256)
        assert 0 <= conf <= 1.0
        # Most pixels should be classified as water (1) or flooded (2)
        water_pixels = (pred == 1).sum() + (pred == 2).sum()
        assert water_pixels > 100

    def test_calculate_metrics(self):
        analysis = FloodingAnalysis()
        prediction = np.zeros((256, 256), dtype=np.int32)
        prediction[100:150, 100:150] = 2  # flooded patch
        bbox = [36.7, -1.4, 37.0, -1.1]
        metrics = analysis.calculate_metrics(prediction, (256, 256), bbox=bbox)
        assert "flooded_percentage" in metrics
        assert "flooded_area_km2" in metrics
        assert metrics["flooded_percentage"] > 0

    def test_generate_alerts_critical(self):
        analysis = FloodingAnalysis()
        metrics = {"flooded_percentage": 25.0, "flooded_area_km2": 10.0}
        alerts = analysis.generate_alerts(metrics)
        assert len(alerts) == 1
        assert alerts[0].severity.value == "critical"
        assert "Critical Flooding" in alerts[0].title

    def test_generate_alerts_warning(self):
        analysis = FloodingAnalysis()
        metrics = {"flooded_percentage": 8.0, "flooded_area_km2": 2.0}
        alerts = analysis.generate_alerts(metrics)
        assert len(alerts) == 1
        assert alerts[0].severity.value == "high"
        assert "Flooding Detected" in alerts[0].title

    def test_generate_alerts_no_alert(self):
        analysis = FloodingAnalysis()
        metrics = {"flooded_percentage": 1.0}
        alerts = analysis.generate_alerts(metrics)
        assert len(alerts) == 0

    def test_generate_alerts_rapid_expansion(self):
        analysis = FloodingAnalysis()
        prev = {"flooded_percentage": 5.0}
        curr = {"flooded_percentage": 20.0}
        alerts = analysis.generate_alerts(curr, previous_metrics=prev)
        alert_types = [a.alert_type for a in alerts]
        assert "rapid_flood_expansion" in alert_types
