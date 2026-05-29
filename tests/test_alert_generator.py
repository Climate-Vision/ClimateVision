"""Tests for inference.alert_generator.

Imports the module via importlib to avoid the broken
``climatevision.inference.__init__`` -> data package chain.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "climatevision"
    / "inference"
    / "alert_generator.py"
)
_spec = importlib.util.spec_from_file_location("cv_alert_generator", _PATH)
ag = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["cv_alert_generator"] = ag
_spec.loader.exec_module(ag)


def _amazon_subscription(**overrides):
    base = dict(
        org_id=1,
        bbox=(-60.0, -15.0, -45.0, 5.0),
        analysis_type="deforestation",
        alert_threshold=0.15,
        channels=("log",),
        cooldown_minutes=60,
    )
    base.update(overrides)
    return ag.Subscription(**base)


def _frozen_clock(start: datetime):
    state = {"now": start}

    def clock():
        return state["now"]

    def advance(minutes: int):
        state["now"] = state["now"] + timedelta(minutes=minutes)

    return clock, advance


def test_alert_fires_when_threshold_exceeded(tmp_path):
    gen = ag.AlertGenerator(
        subscriptions=[_amazon_subscription()],
        alert_log_path=tmp_path / "alerts.jsonl",
        delivery={"log": ag.log_channel},
    )
    alerts = gen.evaluate(
        analysis_type="deforestation",
        bbox=(-55.0, -10.0, -50.0, 0.0),
        measured_value=0.30,
    )
    assert len(alerts) == 1
    assert alerts[0].severity == "high"  # 0.30 >= 0.15 * 2
    assert alerts[0].channels == ("log",)


def test_no_alert_below_threshold(tmp_path):
    gen = ag.AlertGenerator(
        subscriptions=[_amazon_subscription()],
        alert_log_path=tmp_path / "alerts.jsonl",
    )
    alerts = gen.evaluate(
        analysis_type="deforestation",
        bbox=(-55.0, -10.0, -50.0, 0.0),
        measured_value=0.10,
    )
    assert alerts == []


def test_subscription_filtered_by_analysis_type(tmp_path):
    gen = ag.AlertGenerator(
        subscriptions=[_amazon_subscription()],
        alert_log_path=tmp_path / "alerts.jsonl",
    )
    alerts = gen.evaluate(
        analysis_type="flooding",
        bbox=(-55.0, -10.0, -50.0, 0.0),
        measured_value=0.99,
    )
    assert alerts == []


def test_subscription_filtered_by_bbox_overlap(tmp_path):
    gen = ag.AlertGenerator(
        subscriptions=[_amazon_subscription()],
        alert_log_path=tmp_path / "alerts.jsonl",
    )
    # Disjoint bbox over Africa
    alerts = gen.evaluate(
        analysis_type="deforestation",
        bbox=(20.0, 0.0, 30.0, 10.0),
        measured_value=0.99,
    )
    assert alerts == []


def test_cooldown_suppresses_duplicates(tmp_path):
    start = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    clock, advance = _frozen_clock(start)
    gen = ag.AlertGenerator(
        subscriptions=[_amazon_subscription(cooldown_minutes=30)],
        alert_log_path=tmp_path / "alerts.jsonl",
        clock=clock,
    )
    first = gen.evaluate("deforestation", (-55.0, -10.0, -50.0, 0.0), 0.30)
    advance(10)
    second = gen.evaluate("deforestation", (-55.0, -10.0, -50.0, 0.0), 0.30)
    advance(40)
    third = gen.evaluate("deforestation", (-55.0, -10.0, -50.0, 0.0), 0.30)

    assert len(first) == 1
    assert second == []
    assert len(third) == 1


def test_severity_escalation():
    assert ag._classify_severity(0.20, 0.15) == "medium"
    assert ag._classify_severity(0.31, 0.15) == "high"
    assert ag._classify_severity(0.46, 0.15) == "critical"


def test_custom_channel_delivery_called(tmp_path):
    delivered: list = []

    def fake_webhook(alert):
        delivered.append(alert.alert_id)

    sub = _amazon_subscription(channels=("webhook",))
    gen = ag.AlertGenerator(
        subscriptions=[sub],
        alert_log_path=tmp_path / "alerts.jsonl",
        delivery={"webhook": fake_webhook},
    )
    alerts = gen.evaluate("deforestation", (-55.0, -10.0, -50.0, 0.0), 0.30)
    assert len(delivered) == 1
    assert delivered[0] == alerts[0].alert_id


def test_persisted_alerts_can_be_replayed(tmp_path):
    path = tmp_path / "alerts.jsonl"
    gen = ag.AlertGenerator(
        subscriptions=[_amazon_subscription()],
        alert_log_path=path,
    )
    gen.evaluate("deforestation", (-55.0, -10.0, -50.0, 0.0), 0.30)

    fresh = ag.AlertGenerator(alert_log_path=path)
    replayed = fresh.iter_alerts()
    assert len(replayed) == 1
    assert replayed[0].severity in {"medium", "high", "critical"}
