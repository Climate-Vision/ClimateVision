"""Tests for governance.datasheet."""

from __future__ import annotations

import json

import pytest

from climatevision.governance.datasheet import (
    Datasheet,
    build_datasheet,
    generate,
    render_markdown,
    write_datasheet,
)


def _valid_manifest() -> dict:
    return {
        "name": "sentinel2-deforestation",
        "version": "1.0.0",
        "motivation": {
            "purpose": "Detect Amazon basin deforestation events from Sentinel-2.",
            "creators": "ClimateVision Data Pipeline team",
            "funding": "Self-funded open-source initiative.",
        },
        "composition": {
            "instances": "12,480 256x256 tiles",
            "labels": "Binary deforestation mask per tile",
            "splits": "70/15/15 train/val/test by spatial cluster",
            "label_source": "Hansen Global Forest Change v1.10",
        },
        "collection_process": {
            "source": "Sentinel-2 L2A via Google Earth Engine",
            "timeframe": "2020-01-01 to 2023-12-31",
            "consent": "Public open-data licence; no human subjects.",
        },
        "preprocessing": {
            "cloud_masking": "QA60 + s2cloudless threshold 0.4",
            "normalisation": "Per-band z-score against training set means",
            "augmentation": "Random flip / 90deg rotate at train time only",
        },
        "uses": {
            "intended_uses": [
                "Training U-Net segmentation models for deforestation detection.",
                "Evaluating fairness of detection across forest biomes.",
            ]
        },
        "distribution": {
            "license": "CC-BY-4.0 (derived data)",
            "redistribution": "Allowed with attribution; do not redistribute raw Sentinel-2 tiles.",
        },
    }


def test_build_datasheet_returns_typed_object():
    sheet = build_datasheet(_valid_manifest())
    assert isinstance(sheet, Datasheet)
    assert sheet.name == "sentinel2-deforestation"
    assert sheet.version == "1.0.0"
    assert sheet.motivation["purpose"].startswith("Detect")


def test_inappropriate_uses_default_when_omitted():
    sheet = build_datasheet(_valid_manifest())
    assert sheet.uses["inappropriate_uses"], "default inappropriate_uses should be populated"


def test_inappropriate_uses_respect_override():
    manifest = _valid_manifest()
    manifest["uses"]["inappropriate_uses"] = ["custom override"]
    sheet = build_datasheet(manifest)
    assert sheet.uses["inappropriate_uses"] == ["custom override"]


def test_maintenance_has_default():
    sheet = build_datasheet(_valid_manifest())
    assert "owner" in sheet.maintenance
    assert "update_cadence" in sheet.maintenance


def test_validate_rejects_missing_required_section():
    manifest = _valid_manifest()
    del manifest["motivation"]["purpose"]
    with pytest.raises(ValueError, match="motivation.purpose"):
        build_datasheet(manifest)


def test_validate_rejects_empty_required_field():
    manifest = _valid_manifest()
    manifest["composition"]["labels"] = ""
    with pytest.raises(ValueError, match="composition.labels"):
        build_datasheet(manifest)


def test_validate_rejects_missing_collection_timeframe():
    manifest = _valid_manifest()
    del manifest["collection_process"]["timeframe"]
    with pytest.raises(ValueError, match="collection_process.timeframe"):
        build_datasheet(manifest)


def test_render_markdown_includes_section_headings():
    sheet = build_datasheet(_valid_manifest())
    md = render_markdown(sheet)
    for heading in (
        "# Datasheet:",
        "## Motivation",
        "## Composition",
        "## Collection Process",
        "## Uses",
        "## Distribution",
        "## Maintenance",
    ):
        assert heading in md, f"missing heading: {heading}"


def test_render_markdown_renders_lists_as_bullets():
    sheet = build_datasheet(_valid_manifest())
    md = render_markdown(sheet)
    assert "- Training U-Net segmentation models" in md


def test_write_datasheet_round_trips_json(tmp_path):
    sheet = build_datasheet(_valid_manifest())
    paths = write_datasheet(sheet, output_dir=tmp_path)
    loaded = json.loads(paths["json"].read_text())
    assert loaded["name"] == sheet.name
    assert loaded["composition"]["splits"] == "70/15/15 train/val/test by spatial cluster"


def test_generate_end_to_end(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_valid_manifest()))
    paths = generate(manifest_path, output_dir=tmp_path / "out")
    assert paths["markdown"].exists()
    assert paths["json"].exists()
    assert "Datasheet:" in paths["markdown"].read_text()


def test_generate_loads_yaml(tmp_path):
    pytest.importorskip("yaml")
    import yaml

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(_valid_manifest()))
    paths = generate(manifest_path, output_dir=tmp_path / "out")
    assert paths["markdown"].exists()
