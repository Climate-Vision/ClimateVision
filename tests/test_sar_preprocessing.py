"""
Tests for SAR preprocessing module.
"""
from __future__ import annotations

import numpy as np
import pytest

from climatevision.data.sar_preprocessing import (
    RefinedLeeSpeckleFilter,
    linear_to_db,
    db_to_linear,
    apply_slope_mask,
)


class TestRefinedLeeSpeckleFilter:
    def test_reduces_variance(self):
        rng = np.random.default_rng(42)
        # Create image with multiplicative speckle noise (SAR-like)
        base = rng.uniform(0.3, 0.8, (128, 128)).astype(np.float32)
        speckle = rng.gamma(1.0, 1.0, (128, 128)).astype(np.float32)
        image = base * speckle
        flt = RefinedLeeSpeckleFilter(window_size=7)
        filtered = flt(image)
        assert filtered.shape == image.shape
        assert filtered.dtype == np.float32
        # Variance should be reduced for speckled images
        assert filtered.var() < image.var()

    def test_3d_input(self):
        rng = np.random.default_rng(42)
        image = rng.normal(0.5, 0.2, (2, 128, 128)).astype(np.float32)
        flt = RefinedLeeSpeckleFilter(window_size=7)
        filtered = flt(image)
        assert filtered.shape == image.shape


class TestBackscatterConversion:
    def test_linear_to_db_roundtrip(self):
        linear = np.array([0.01, 0.1, 1.0, 10.0], dtype=np.float32)
        db = linear_to_db(linear)
        back = db_to_linear(db)
        np.testing.assert_allclose(back, linear, rtol=1e-5)


class TestSlopeMask:
    def test_masks_steep_slopes(self):
        image = np.ones((100, 100), dtype=np.float32)
        slope = np.zeros((100, 100), dtype=np.float32)
        slope[50:60, 50:60] = 20.0  # steep
        masked = apply_slope_mask(image, slope, max_slope_deg=15.0)
        assert np.isnan(masked[55, 55])
        assert not np.isnan(masked[10, 10])
