"""Tests for GET /api/runs/stats, the dashboard KPI summary."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import climatevision.db as db
from climatevision.api.main import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client over an empty database, so the counts are exact."""
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "stats.sqlite3")
    monkeypatch.setattr(db, "_INITIALIZED", False)
    return TestClient(create_app())


def _insert_run(status: str, analysis_type: str, created_at: datetime) -> None:
    timestamp = created_at.isoformat()
    with db.get_connection() as conn:
        conn.execute(
            "INSERT INTO runs (kind, status, analysis_type, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("demo", status, analysis_type, timestamp, timestamp),
        )


def _create_alert(organization_id: int, title: str) -> int:
    return db.create_organization_alert(
        organization_id=organization_id,
        alert_type="deforestation",
        title=title,
        message=f"{title} detected",
    )


def test_stats_of_an_empty_database(client: TestClient) -> None:
    response = client.get("/api/runs/stats")

    assert response.status_code == 200
    assert response.json() == {
        "total_runs": 0,
        "completed_last_7_days": 0,
        "by_status": {},
        "by_analysis_type": {},
        "alerts": {"total": 0, "unacknowledged": 0},
    }


def test_stats_count_organization_alerts(client: TestClient) -> None:
    """Alerts are stored in organization_alerts, where they are acknowledged."""
    org_id = db.create_organization(name="Stats NGO")["id"]
    first = _create_alert(org_id, "Clearing")
    _create_alert(org_id, "Burn scar")
    _create_alert(org_id, "Road")
    db.acknowledge_alert(first, acknowledged_by="ranger")

    response = client.get("/api/runs/stats")

    assert response.status_code == 200
    assert response.json()["alerts"] == {"total": 3, "unacknowledged": 2}


def test_stats_count_runs(client: TestClient) -> None:
    now = datetime.now(timezone.utc)
    _insert_run("completed", "deforestation", now)
    _insert_run("completed", "flooding", now - timedelta(days=30))
    _insert_run("failed", "deforestation", now)

    response = client.get("/api/runs/stats")

    assert response.status_code == 200
    stats = response.json()
    assert stats["total_runs"] == 3
    assert stats["by_status"] == {"completed": 2, "failed": 1}
    assert stats["by_analysis_type"] == {"deforestation": 2, "flooding": 1}
    assert stats["completed_last_7_days"] == 1
