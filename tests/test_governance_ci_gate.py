"""Tests for scripts.governance_ci_gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_GATE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "governance_ci_gate.py"
_spec = importlib.util.spec_from_file_location("governance_ci_gate", _GATE_PATH)
gate = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["governance_ci_gate"] = gate
_spec.loader.exec_module(gate)


def _good_metrics():
    return {"iou": 0.82, "f1": 0.86, "precision": 0.88, "recall": 0.85}


def _bad_metrics():
    return {"iou": 0.55, "f1": 0.60}


def _good_fairness():
    return {"score": 0.92, "disparity_regions": []}


def _bad_fairness():
    return {"score": 0.5, "disparity_regions": ["amazon", "congo"]}


def _security(high=0, critical=0):
    findings = [{"severity": "high"}] * high + [{"severity": "critical"}] * critical
    return {"findings": findings}


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_metrics_gate_passes_when_above_threshold(tmp_path):
    metrics_path = _write(tmp_path / "m.json", _good_metrics())
    passed, results = gate.run_gate(metrics_path, None, None, gate._merge_thresholds(None))
    assert passed
    assert all(r.passed for r in results)


def test_metrics_gate_fails_when_below_threshold(tmp_path):
    metrics_path = _write(tmp_path / "m.json", _bad_metrics())
    passed, _ = gate.run_gate(metrics_path, None, None, gate._merge_thresholds(None))
    assert not passed


def test_fairness_gate_fails_on_disparity(tmp_path):
    metrics = _write(tmp_path / "m.json", _good_metrics())
    fairness = _write(tmp_path / "f.json", _bad_fairness())
    passed, results = gate.run_gate(metrics, fairness, None, gate._merge_thresholds(None))
    assert not passed
    failed_names = {r.name for r in results if not r.passed}
    assert "fairness.score" in failed_names or "fairness.disparity_regions" in failed_names


def test_security_gate_fails_on_high_finding(tmp_path):
    metrics = _write(tmp_path / "m.json", _good_metrics())
    security = _write(tmp_path / "s.json", _security(high=1))
    passed, _ = gate.run_gate(metrics, None, security, gate._merge_thresholds(None))
    assert not passed


def test_thresholds_can_be_overridden(tmp_path):
    metrics = _write(tmp_path / "m.json", _bad_metrics())
    custom = {"metrics": {"iou": 0.5, "f1": 0.5}}
    passed, _ = gate.run_gate(metrics, None, None, gate._merge_thresholds(custom))
    assert passed


def test_render_summary_includes_pass_and_fail():
    results = [
        gate.GateResult(name="a", passed=True, detail="ok"),
        gate.GateResult(name="b", passed=False, detail="bad"),
    ]
    md = gate.render_summary(results)
    assert "Governance CI Gate — FAIL" in md
    assert "PASS" in md and "FAIL" in md


def test_main_exits_nonzero_on_failure(tmp_path):
    metrics = _write(tmp_path / "m.json", _bad_metrics())
    rc = gate.main(["--metrics", str(metrics)])
    assert rc == gate.EXIT_FAIL


def test_main_exits_zero_on_success(tmp_path):
    metrics = _write(tmp_path / "m.json", _good_metrics())
    fairness = _write(tmp_path / "f.json", _good_fairness())
    security = _write(tmp_path / "s.json", _security())
    summary_path = tmp_path / "out" / "summary.md"
    rc = gate.main([
        "--metrics", str(metrics),
        "--fairness", str(fairness),
        "--security", str(security),
        "--summary-out", str(summary_path),
    ])
    assert rc == gate.EXIT_OK
    assert summary_path.exists()
    assert "PASS" in summary_path.read_text()


def test_main_exits_bad_input_on_missing_metrics(tmp_path):
    rc = gate.main(["--metrics", str(tmp_path / "missing.json")])
    assert rc == gate.EXIT_BAD_INPUT
