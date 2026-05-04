"""Tests for reports.llm_reporter."""

from __future__ import annotations

import json

import pytest

from climatevision.reports.llm_reporter import (
    ImpactReport,
    LLMReporter,
    ReportContext,
    generate_impact_report,
    render_template,
)


def _ctx():
    return ReportContext(
        region="amazon",
        period="2026-Q1",
        analysis_type="deforestation",
        carbon={"hectares": 1247.5, "carbon_tonnes": 4321.2, "ci_lower": 4000.0, "ci_upper": 4600.0},
        validation={"iou": 0.81, "f1": 0.87},
        shap={"top_bands": [{"band": "NIR", "importance": 0.42}, {"band": "Red", "importance": 0.31}]},
        fairness={"score": 0.93, "disparity_regions": []},
        run_id=12345,
    )


def test_headline_metric_uses_carbon_when_available():
    text = _ctx().headline_metric()
    assert "1,247.5 hectares" in text
    assert "4,321.2 tCO2e" in text


def test_template_renders_all_sections():
    md = render_template(_ctx())
    for heading in [
        "# Impact Report",
        "## Carbon Analytics",
        "## Validation",
        "## Explainability",
        "## Fairness",
    ]:
        assert heading in md


def test_template_skips_shap_when_disabled():
    md = render_template(_ctx(), include_shap=False)
    assert "## Explainability" not in md


def test_reporter_falls_back_to_template_without_llm():
    report = LLMReporter().generate(_ctx())
    assert report.provider == "template"
    assert "amazon" in report.body.lower()


def test_reporter_uses_provided_llm_callable():
    captured = {}

    def fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "Executive summary line.\n\n## Carbon Analytics\n- Hectares: 1247.5\n"

    report = LLMReporter(llm=fake_llm).generate(_ctx())
    assert report.provider == "llm"
    assert "Executive summary line." in report.summary
    assert "amazon" in captured["prompt"].lower()


def test_reporter_handles_llm_exception_gracefully():
    def boom(prompt: str) -> str:
        raise RuntimeError("provider down")

    report = LLMReporter(llm=boom).generate(_ctx())
    assert report.provider == "template"


def test_generate_impact_report_writes_to_disk(tmp_path):
    report = generate_impact_report(
        region="amazon",
        period="2026-Q1",
        analysis_type="deforestation",
        carbon={"hectares": 100.0, "carbon_tonnes": 350.0},
        validation={"iou": 0.7, "f1": 0.8},
        output_dir=tmp_path,
    )
    md_path = tmp_path / "amazon_2026-Q1_impact.md"
    json_path = tmp_path / "amazon_2026-Q1_impact.json"

    assert isinstance(report, ImpactReport)
    assert md_path.exists()
    assert json_path.exists()

    payload = json.loads(json_path.read_text())
    assert payload["context"]["region"] == "amazon"
    assert payload["provider"] == "template"
