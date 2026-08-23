"""In-process publish/subscribe hub for run status events.

`/api/predict` and `/api/predict/upload` execute inference inline, so a run's
status transitions happen inside the same process that serves the WebSocket.
That makes a small in-memory hub sufficient: the predict handlers publish a
terminal event, and every socket currently watching that run receives it.

The hub deliberately does not persist anything. The WebSocket handler reads the
authoritative current state from the database when a client connects, and only
uses the hub for changes that happen *while* the client is attached. A client
that connects after a run has already finished still gets a terminal event from
that initial database read.

Note that this is per-process state. Under a multi-worker deployment a socket
served by worker A will not observe a transition published by worker B; such a
client falls back to the initial snapshot plus the frontend's polling path. A
cross-process broker (Redis pub/sub or Postgres LISTEN/NOTIFY) would be the
natural upgrade and is tracked separately rather than folded in here.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

# Bounded so a client that stops reading cannot grow the queue without limit.
_MAX_QUEUED_EVENTS = 32


class RunEventHub:
    """Fan-out of run status events to the sockets watching each run."""

    def __init__(self) -> None:
        self._subscribers: dict[int, set[asyncio.Queue[dict[str, Any]]]] = {}
        self._lock = asyncio.Lock()

    async def publish(self, run_id: int, event: dict[str, Any]) -> None:
        """Deliver an event to every subscriber of ``run_id``.

        Args:
            run_id: The run the event belongs to.
            event: The payload to deliver. Delivered as-is.
        """
        async with self._lock:
            queues = list(self._subscribers.get(run_id, ()))

        for queue in queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A subscriber that cannot keep up is skipped rather than
                # allowed to block inference from completing.
                logger.warning(
                    "Dropping run event for run %s: subscriber queue is full", run_id
                )

    @asynccontextmanager
    async def subscribe(
        self, run_id: int
    ) -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
        """Subscribe to ``run_id`` for the duration of the context.

        Args:
            run_id: The run to watch.

        Yields:
            A queue that receives every event published for the run.
        """
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_MAX_QUEUED_EVENTS)
        async with self._lock:
            self._subscribers.setdefault(run_id, set()).add(queue)
        try:
            yield queue
        finally:
            async with self._lock:
                subscribers = self._subscribers.get(run_id)
                if subscribers is not None:
                    subscribers.discard(queue)
                    # Keep the map from growing once a run has no watchers.
                    if not subscribers:
                        del self._subscribers[run_id]

    def subscriber_count(self, run_id: int) -> int:
        """Return how many sockets are currently watching ``run_id``."""
        return len(self._subscribers.get(run_id, ()))


# Module-level hub shared by the predict handlers and the WebSocket endpoint.
run_event_hub = RunEventHub()


def build_status_event(
    run_id: int,
    status: str,
    *,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build the wire payload for a run status event.

    Args:
        run_id: The run the event describes.
        status: The run's status, e.g. ``running``/``completed``/``failed``.
        result: The result payload, for a ``completed`` run.
        error: The failure message, for a ``failed`` run.

    Returns:
        The event dictionary sent over the WebSocket.
    """
    event: dict[str, Any] = {"type": "status", "run_id": run_id, "status": status}
    if result is not None:
        event["result"] = result
    if error is not None:
        event["error"] = error
    return event
