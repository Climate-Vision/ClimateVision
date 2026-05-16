"""Smoke tests for analysis-aware Sentinel-2 band mapping."""
from __future__ import annotations

import pytest

from climatevision.data.band_mapping import (
    SCL_BAND,
    SENTINEL2_BAND_ORDER,
    get_band_indices,
    get_bands_for_analysis,
    get_bands_for_analysis_with_scl,
    get_model_config,
    is_analysis_enabled,
    list_enabled_analysis_types,
)


def test_sentinel2_band_order_has_13_bands():
    assert len(SENTINEL2_BAND_ORDER) == 13
    assert SENTINEL2_BAND_ORDER[0] == "B01"
    assert SENTINEL2_BAND_ORDER[-1] == "B12"


def test_deforestation_uses_four_bands():
    bands = get_bands_for_analysis("deforestation")
    assert len(bands) == 4
    assert set(bands) == {"B02", "B03", "B04", "B08"}


def test_flooding_uses_three_bands():
    bands = get_bands_for_analysis("flooding")
    assert len(bands) == 3
    assert "B11" in bands


def test_ice_melting_uses_swir():
    bands = get_bands_for_analysis("ice_melting")
    assert "B11" in bands


def test_scl_appended_for_cloud_masking():
    bands = get_bands_for_analysis_with_scl("deforestation")
    assert SCL_BAND in bands
    assert bands[-1] == SCL_BAND


def test_scl_not_duplicated():
    bands_with_scl = get_bands_for_analysis_with_scl("deforestation")
    bands_again = get_bands_for_analysis_with_scl("deforestation")
    assert bands_with_scl.count(SCL_BAND) == 1
    assert bands_again.count(SCL_BAND) == 1


def test_band_indices_resolve_correctly():
    indices = get_band_indices(["B04", "B03", "B02", "B08"])
    assert indices == [3, 2, 1, 7]


def test_band_indices_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown"):
        get_band_indices(["B99"])


def test_band_indices_rejects_scl_directly():
    with pytest.raises(ValueError, match="SCL"):
        get_band_indices([SCL_BAND])


def test_enabled_analysis_types_include_active_three():
    enabled = list_enabled_analysis_types()
    for name in ("deforestation", "ice_melting", "flooding"):
        assert name in enabled, f"{name} should be enabled"


def test_disabled_analysis_types():
    assert not is_analysis_enabled("drought")
    assert not is_analysis_enabled("wildfire")


def test_model_config_carries_channels_and_classes():
    cfg = get_model_config("flooding")
    assert cfg["in_channels"] == 3
    assert cfg["num_classes"] == 3
