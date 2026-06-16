"""
Tests for permanent-water vs flood-water classification.
"""
from __future__ import annotations

import numpy as np
import pytest

from climatevision.analysis.flood_classification import (
    DRY_LAND,
    FLOODED,
    PERMANENT_WATER,
    classify_with_change,
    classify_with_reference,
    permanent_water_from_occurrence,
)
from climatevision.analysis.flooding_ensemble import EnsembleFloodPipeline


class TestClassifyWithReference:
    def test_splits_permanent_and_flood(self):
        water = np.array([[1, 1], [1, 0]], dtype=np.uint8)
        perm = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        out = classify_with_reference(water, perm)
        assert out[0, 0] == PERMANENT_WATER  # water + in reference
        assert out[0, 1] == FLOODED          # water, not in reference
        assert out[1, 0] == FLOODED          # water, not in reference
        assert out[1, 1] == DRY_LAND         # no water

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            classify_with_reference(np.zeros((4, 4)), np.zeros((2, 2)))


class TestClassifyWithChange:
    def test_change_separates_flood_from_permanent(self):
        pre = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        post = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        out = classify_with_change(pre, post)
        assert out[0, 0] == PERMANENT_WATER  # water before and after
        assert out[0, 1] == FLOODED          # appeared after
        assert out[1, 0] == DRY_LAND         # dry both
        assert out[1, 1] == DRY_LAND         # receded -> not flooded now


class TestOccurrence:
    def test_threshold(self):
        occ = np.array([[10.0, 60.0], [50.0, 0.0]])
        mask = permanent_water_from_occurrence(occ, threshold_pct=50.0)
        assert mask.tolist() == [[0, 1], [1, 0]]


class TestEnsembleIntegration:
    def test_reference_yields_three_class_output(self):
        # Low VH (dB) reads as water for the TUW/DLR detectors.
        post_vh = np.full((16, 16), -10.0, dtype=np.float32)
        post_vh[4:12, 4:12] = -26.0  # a water blob
        perm_ref = np.zeros((16, 16), dtype=np.uint8)
        perm_ref[4:8, 4:8] = 1  # half the blob is "normally water"

        out = EnsembleFloodPipeline().detect(post_vh, permanent_water_ref=perm_ref)
        classified = out["classified_mask"]
        assert classified is not None
        # Both permanent and flood pixels should be present in the blob.
        assert (classified == PERMANENT_WATER).any()
        assert (classified == FLOODED).any()

    def test_no_reference_returns_none_not_a_guess(self):
        post_vh = np.full((8, 8), -10.0, dtype=np.float32)
        out = EnsembleFloodPipeline().detect(post_vh)
        assert out["classified_mask"] is None
