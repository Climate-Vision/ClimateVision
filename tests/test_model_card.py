"""Tests for governance.model_card."""

from __future__ import annotations

import json

import pytest

from climatevision.governance.model_card import (
    REQUIRED_METRICS,
    build_model_card,
    generate,
    render_markdown,
    write_model_card,
)


def _config():
    return {
        "model": {"name": "unet-deforestation", "version": "1.2.0"},
        "analysis_type": "deforestation",
        "training_data": {
            "regions": ["amazon", "congo"],
            "tile_count": 12000,
        },
        "evaluation_data": {"regions": ["southeast_asia"], "tile_count": 1500},
    }


def _metrics():
    return {"iou": 0.81, "f1": 0.86, "precision": 0.88, "recall": 0.85}


def test_build_card_uses_config_values():
    card = build_model_card(_config(), _metrics())
    assert card.name == "unet-deforestation"
    assert card.version == "1.2.0"
    assert card.analysis_type == "deforestation"
    assert card.metrics == _metrics()
    assert card.training_data["tile_count"] == 12000


def test_missing_metric_raises():
    bad = {"iou": 0.5, "f1": 0.5}
    with pytest.raises(ValueError):
        build_model_card(_config(), bad)


def test_required_metric_set_is_documented():
    assert set(REQUIRED_METRICS) <= set(_metrics())


def test_render_markdown_includes_all_sections():
    card = build_model_card(_config(), _metrics(), fairness_report={"score": 0.92})
    md = render_markdown(card)
    for heading in [
        "# Model Card:",
        "## Description",
        "## Intended Use",
        "## Training Data",
        "## Evaluation",
        "## Fairness",
        "## Limitations",
        "## Ethical Considerations",
        "## Contact",
    ]:
        assert heading in md
    assert "score" in md


def test_write_model_card_emits_md_and_json(tmp_path):
    card = build_model_card(_config(), _metrics())
    paths = write_model_card(card, output_dir=tmp_path)

    assert paths["markdown"].exists()
    assert paths["json"].exists()

    payload = json.loads(paths["json"].read_text())
    assert payload["version"] == "1.2.0"
    assert payload["metrics"]["iou"] == pytest.approx(0.81)


def test_generate_loads_files_from_disk(tmp_path):
    cfg_path = tmp_path / "config.json"
    metrics_path = tmp_path / "metrics.json"
    cfg_path.write_text(json.dumps(_config()))
    metrics_path.write_text(json.dumps(_metrics()))

    paths = generate(
        config=cfg_path,
        metrics=metrics_path,
        output_dir=tmp_path / "cards",
    )
    assert paths["markdown"].exists()
    assert paths["json"].exists()
