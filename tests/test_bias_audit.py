"""Smoke tests for the regional bias audit framework."""
from __future__ import annotations

from climatevision.governance.bias_audit import SUPPORTED_REGIONS


def test_four_regions_supported():
    expected = {"amazon", "congo", "southeast_asia", "boreal"}
    assert set(SUPPORTED_REGIONS.keys()) == expected


def test_each_region_has_bbox_of_four_floats():
    for region, meta in SUPPORTED_REGIONS.items():
        bbox = meta["bbox"]
        assert len(bbox) == 4, f"{region} bbox must have 4 values"
        for v in bbox:
            assert isinstance(v, (int, float)), f"{region} bbox must be numeric"


def test_bbox_west_less_than_east():
    for region, meta in SUPPORTED_REGIONS.items():
        west, _south, east, _north = meta["bbox"]
        # Boreal spans the international date line, accept either ordering
        if region == "boreal":
            continue
        assert west < east, f"{region}: west {west} must be < east {east}"


def test_bbox_south_less_than_north():
    for region, meta in SUPPORTED_REGIONS.items():
        _west, south, _east, north = meta["bbox"]
        assert south < north, f"{region}: south {south} must be < north {north}"


def test_each_region_has_human_readable_metadata():
    for region, meta in SUPPORTED_REGIONS.items():
        assert "name" in meta and isinstance(meta["name"], str) and meta["name"]
        assert "description" in meta and isinstance(meta["description"], str)


def test_amazon_bbox_in_south_america():
    bbox = SUPPORTED_REGIONS["amazon"]["bbox"]
    west, south, east, north = bbox
    # Loosely: longitudes negative (western hemisphere), latitudes near equator
    assert west < 0 and east < 0
    assert -20 < south < north < 10


def test_congo_bbox_straddles_equator():
    bbox = SUPPORTED_REGIONS["congo"]["bbox"]
    _west, south, _east, north = bbox
    assert south < 0 < north
