"""Tests for api.admin operational endpoints.

Imports the admin module via importlib to avoid the broken
``climatevision.data`` package __init__ chain (irrelevant to admin).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "climatevision"
    / "api"
    / "admin.py"
)
_spec = importlib.util.spec_from_file_location("cv_api_admin", _PATH)
admin = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["cv_api_admin"] = admin
_spec.loader.exec_module(admin)


@pytest.fixture
def env(tmp_path, monkeypatch):
    audit = tmp_path / "audit.jsonl"
    alerts = tmp_path / "alerts.jsonl"
    monkeypatch.setattr(admin, "DEFAULT_AUDIT_LOG", audit)
    monkeypatch.setattr(admin, "DEFAULT_ALERT_LOG", alerts)
    return audit, alerts


def _now():
    return datetime.now(timezone.utc)


def _write_audit(path, entries):
    with path.open("a") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


def _make_audit_entry(minutes_ago: int, mean_conf: float, positive: float, error: bool = False):
    ts = _now() - timedelta(minutes=minutes_ago)
    return {
        "timestamp": ts.isoformat(),
        "model_version": "v1",
        "input_hash": "abc",
        "output_summary": {"mean_confidence": mean_conf, "positive_fraction": positive},
        "request_id": None,
        "user_id": None,
        "prev_hash": "0" * 64,
        "entry_hash": "x",
        "metadata": {},
        **({"error": "boom"} if error else {}),
    }


def _make_alert(minutes_ago: int, severity: str = "high"):
    ts = _now() - timedelta(minutes=minutes_ago)
    return {
        "alert_id": "id",
        "org_id": 1,
        "analysis_type": "deforestation",
        "region_bbox": [-60, -15, -45, 5],
        "severity": severity,
        "measured_value": 0.3,
        "threshold": 0.15,
        "summary": "test",
        "triggered_at": ts.isoformat(),
        "channels": ["log"],
    }


def _client():
    app = FastAPI()
    app.include_router(admin.router)
    return TestClient(app)


def test_reports_returns_zeros_for_empty_logs(env):
    client = _client()
    resp = client.get("/api/reports?window_hours=24")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_count"] == 0
    assert body["error_rate"] == 0.0
    assert body["mean_confidence"] is None
    assert body["alert_count"] == 0


def test_reports_aggregates_within_window(env):
    audit, alerts = env
    _write_audit(audit, [
        _make_audit_entry(minutes_ago=10, mean_conf=0.8, positive=0.2),
        _make_audit_entry(minutes_ago=30, mean_conf=0.9, positive=0.4),
        _make_audit_entry(minutes_ago=60 * 48, mean_conf=0.5, positive=0.1),  # outside
        _make_audit_entry(minutes_ago=20, mean_conf=0.7, positive=0.3, error=True),
    ])
    _write_audit(alerts, [_make_alert(15), _make_alert(60 * 48)])

    client = _client()
    body = client.get("/api/reports?window_hours=24").json()

    assert body["run_count"] == 3
    assert pytest.approx(body["error_rate"], rel=1e-3) == 1 / 3
    assert pytest.approx(body["mean_confidence"], rel=1e-3) == (0.8 + 0.9 + 0.7) / 3
    assert pytest.approx(body["positive_fraction_mean"], rel=1e-3) == (0.2 + 0.4 + 0.3) / 3
    assert body["alert_count"] == 1


def test_reports_rejects_zero_window(env):
    client = _client()
    resp = client.get("/api/reports?window_hours=0")
    assert resp.status_code == 422


def test_anomalies_lists_all_when_unfiltered(env):
    _, alerts = env
    _write_audit(alerts, [_make_alert(5, "high"), _make_alert(10, "medium")])
    body = _client().get("/api/anomalies").json()
    assert body["count"] == 2


def test_anomalies_filters_by_severity(env):
    _, alerts = env
    _write_audit(alerts, [_make_alert(5, "high"), _make_alert(10, "medium")])
    body = _client().get("/api/anomalies?severity=high").json()
    assert body["count"] == 1
    assert body["anomalies"][0]["severity"] == "high"


def test_anomalies_filters_by_window(env):
    _, alerts = env
    _write_audit(alerts, [_make_alert(5, "high"), _make_alert(60 * 48, "high")])
    body = _client().get("/api/anomalies?window_hours=1").json()
    assert body["count"] == 1


def test_anomalies_rejects_invalid_severity(env):
    resp = _client().get("/api/anomalies?severity=blah")
    assert resp.status_code == 422
