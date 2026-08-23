"""Tests for the /ws/runs/{run_id} run status WebSocket."""

import asyncio
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from climatevision.api.main import WS_RUN_NOT_FOUND
from climatevision.api.run_events import (
    RunEventHub,
    build_status_event,
    run_event_hub,
)
from climatevision.db import get_connection


def _insert_run(status: str) -> int:
    """Insert a run with the given status and return its id."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO runs (kind, status, analysis_type, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "test",
                status,
                "deforestation",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ),
        )
        return int(cur.lastrowid)


def _insert_result(run_id: int, payload: dict) -> None:
    """Attach a stored result payload to a run."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO results (run_id, payload_json, mask_path, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, json.dumps(payload), None, "2026-01-01T00:00:00Z"),
        )


def test_unknown_run_is_reported_and_closed(client: TestClient) -> None:
    """A socket for a nonexistent run gets an error event, then a close."""
    with client.websocket_connect("/ws/runs/99999999") as ws:
        event = ws.receive_json()
        assert event["type"] == "error"
        assert event["run_id"] == 99999999
        assert "not found" in event["error"].lower()

        # Starlette surfaces the server-side close as a websocket.close message.
        closed = ws.receive()
        assert closed["type"] == "websocket.close"
        assert closed["code"] == WS_RUN_NOT_FOUND


def test_already_completed_run_sends_terminal_snapshot(client: TestClient) -> None:
    """Connecting to a finished run still yields a terminal event immediately.

    This is the late-attach case: the transition happened before the client
    connected, so it can only come from the initial database read.
    """
    run_id = _insert_run("completed")
    _insert_result(run_id, {"analysis_type": "deforestation", "deforested_pixels": 42})

    with client.websocket_connect(f"/ws/runs/{run_id}") as ws:
        event = ws.receive_json()

    assert event["type"] == "status"
    assert event["run_id"] == run_id
    assert event["status"] == "completed"
    assert event["result"]["deforested_pixels"] == 42


def test_failed_run_surfaces_error_not_result(client: TestClient) -> None:
    """A failed run reports its message under `error`, with no `result`."""
    run_id = _insert_run("failed")
    _insert_result(run_id, {"analysis_type": "deforestation", "error": "GEE timeout"})

    with client.websocket_connect(f"/ws/runs/{run_id}") as ws:
        event = ws.receive_json()

    assert event["status"] == "failed"
    assert event["error"] == "GEE timeout"
    assert "result" not in event


def test_failed_run_without_stored_message_still_reports_an_error(
    client: TestClient,
) -> None:
    """A failed run with no `error` in its payload gets a fallback message."""
    run_id = _insert_run("failed")
    _insert_result(run_id, {"analysis_type": "deforestation"})

    with client.websocket_connect(f"/ws/runs/{run_id}") as ws:
        event = ws.receive_json()

    assert event["status"] == "failed"
    assert event["error"] == "Inference failed"


def test_running_run_streams_transition_then_closes(client: TestClient) -> None:
    """A run still in progress streams its snapshot, then the terminal event."""
    run_id = _insert_run("running")

    with client.websocket_connect(f"/ws/runs/{run_id}") as ws:
        snapshot = ws.receive_json()
        assert snapshot["status"] == "running"
        assert "result" not in snapshot

        # Publishing on the app's event loop mirrors what the predict handler
        # does once inference finishes.
        portal = ws.portal  # type: ignore[attr-defined]
        portal.call(
            run_event_hub.publish,
            run_id,
            build_status_event(run_id, "completed", result={"deforested_pixels": 7}),
        )

        terminal = ws.receive_json()
        assert terminal["status"] == "completed"
        assert terminal["result"]["deforested_pixels"] == 7

        closed = ws.receive()
        assert closed["type"] == "websocket.close"


def test_running_run_ignores_events_for_other_runs(client: TestClient) -> None:
    """Events published for a different run are not delivered to this socket."""
    watched_id = _insert_run("running")
    other_id = _insert_run("running")

    with client.websocket_connect(f"/ws/runs/{watched_id}") as ws:
        assert ws.receive_json()["status"] == "running"

        portal = ws.portal  # type: ignore[attr-defined]
        portal.call(
            run_event_hub.publish,
            other_id,
            build_status_event(other_id, "completed", result={}),
        )
        portal.call(
            run_event_hub.publish,
            watched_id,
            build_status_event(watched_id, "failed", error="boom"),
        )

        event = ws.receive_json()

    # The first event received is the one for the watched run, proving the
    # other run's event was never queued onto this subscription.
    assert event["run_id"] == watched_id
    assert event["status"] == "failed"
    assert event["error"] == "boom"


def test_subscription_is_released_when_the_client_disconnects(
    client: TestClient,
) -> None:
    """Leaving the socket context drops the subscription, avoiding a leak."""
    run_id = _insert_run("running")

    with client.websocket_connect(f"/ws/runs/{run_id}") as ws:
        ws.receive_json()
        assert run_event_hub.subscriber_count(run_id) == 1

    # Give the server task a moment to unwind the context manager.
    for _ in range(50):
        if run_event_hub.subscriber_count(run_id) == 0:
            break
        import time

        time.sleep(0.01)

    assert run_event_hub.subscriber_count(run_id) == 0


# ===== Predict-to-hub wiring =====


def test_predict_publishes_a_completed_event(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful /api/predict publishes the run's terminal event."""
    monkeypatch.setenv("CLIMATEVISION_ALLOW_DEV_KEY", "1")
    published: list[tuple[int, dict]] = []

    async def record(run_id: int, event: dict) -> None:
        published.append((run_id, event))

    monkeypatch.setattr(run_event_hub, "publish", record)

    fake_result = {"inference": {"forest_percentage": 72.3}}
    with patch(
        "climatevision.api.main.run_inference_from_gee", return_value=fake_result
    ):
        response = client.post(
            "/api/predict",
            json={
                "bbox": [-60.0, -15.0, -45.0, -5.0],
                "start_date": "2023-01-01",
                "end_date": "2023-06-30",
                "analysis_type": "deforestation",
            },
            headers={"X-API-Key": "cv_dev"},
        )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    assert len(published) == 1
    published_run_id, event = published[0]
    assert published_run_id == run_id
    assert event["status"] == "completed"
    assert event["result"]["inference"]["forest_percentage"] == 72.3


def test_predict_publishes_a_failed_event_when_inference_raises(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed /api/predict publishes `failed` with the error message."""
    monkeypatch.setenv("CLIMATEVISION_ALLOW_DEV_KEY", "1")
    published: list[tuple[int, dict]] = []

    async def record(run_id: int, event: dict) -> None:
        published.append((run_id, event))

    monkeypatch.setattr(run_event_hub, "publish", record)

    with patch(
        "climatevision.api.main.run_inference_from_gee",
        side_effect=RuntimeError("GEE unavailable"),
    ):
        response = client.post(
            "/api/predict",
            json={
                "bbox": [-60.0, -15.0, -45.0, -5.0],
                "start_date": "2023-01-01",
                "end_date": "2023-06-30",
                "analysis_type": "deforestation",
            },
            headers={"X-API-Key": "cv_dev"},
        )

    assert response.status_code == 200
    assert len(published) == 1
    _, event = published[0]
    assert event["status"] == "failed"
    assert event["error"] == "GEE unavailable"
    assert "result" not in event


# ===== RunEventHub unit tests =====


def test_hub_delivers_to_every_subscriber_of_a_run() -> None:
    """All sockets watching the same run receive the event."""

    async def scenario() -> None:
        hub = RunEventHub()
        async with hub.subscribe(1) as first, hub.subscribe(1) as second:
            await hub.publish(1, {"type": "status", "status": "completed"})
            assert (await first.get())["status"] == "completed"
            assert (await second.get())["status"] == "completed"

    asyncio.run(scenario())


def test_hub_publish_to_a_run_with_no_subscribers_is_a_no_op() -> None:
    """Publishing to an unwatched run neither raises nor retains state."""

    async def scenario() -> None:
        hub = RunEventHub()
        await hub.publish(404, {"type": "status", "status": "completed"})
        assert hub.subscriber_count(404) == 0

    asyncio.run(scenario())


def test_hub_drops_events_for_a_subscriber_that_stops_reading() -> None:
    """A full queue is skipped so publishing never blocks inference."""

    async def scenario() -> None:
        hub = RunEventHub()
        async with hub.subscribe(1) as queue:
            # Fill well past the bound without ever draining the queue.
            for i in range(100):
                await hub.publish(1, {"type": "status", "seq": i})
            assert queue.full()
            # Publishing still returns rather than blocking forever.
            await asyncio.wait_for(hub.publish(1, {"type": "status", "seq": 100}), 1)

    asyncio.run(scenario())


def test_hub_forgets_a_run_once_its_last_subscriber_leaves() -> None:
    """The subscriber map does not grow once a run has no watchers."""

    async def scenario() -> None:
        hub = RunEventHub()
        async with hub.subscribe(7):
            assert hub.subscriber_count(7) == 1
        assert hub.subscriber_count(7) == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "kwargs,expected_keys",
    [
        ({}, {"type", "run_id", "status"}),
        ({"result": {"a": 1}}, {"type", "run_id", "status", "result"}),
        ({"error": "boom"}, {"type", "run_id", "status", "error"}),
    ],
)
def test_build_status_event_only_includes_provided_fields(
    kwargs: dict, expected_keys: set
) -> None:
    """Optional fields stay out of the payload unless explicitly supplied."""
    event = build_status_event(1, "completed", **kwargs)
    assert set(event) == expected_keys
