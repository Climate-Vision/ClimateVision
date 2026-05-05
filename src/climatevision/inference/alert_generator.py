"""
Deforestation alert generator for ClimateVision.

Watches for inference results that exceed a configurable threshold for
a given subscription (region, analysis_type, alert_threshold) and emits
notifications via pluggable channels (email, webhook, log).

Routing rules:

- Each `Subscription` defines a region (bbox), analysis type, threshold,
  and a list of channels to deliver to.
- A new prediction is matched against subscriptions by analysis type
  and whether its bbox overlaps the subscription bbox.
- Alerts are de-duplicated within a configurable cooldown window so a
  flapping signal does not page everyone every minute.

The generator does not perform delivery itself for non-loggable channels;
it returns delivery records that the caller (typically the alert worker
or `notification_router.deliver_pending`) is responsible for sending.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ALERT_LOG = _PROJECT_ROOT / "outputs" / "alerts" / "alerts.jsonl"

DeliveryFn = Callable[["Alert"], None]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Subscription:
    org_id: int
    bbox: tuple[float, float, float, float]
    analysis_type: str
    alert_threshold: float
    channels: tuple[str, ...] = ("log",)
    cooldown_minutes: int = 60


@dataclass
class Alert:
    alert_id: str
    org_id: int
    analysis_type: str
    region_bbox: tuple[float, float, float, float]
    severity: str
    measured_value: float
    threshold: float
    summary: str
    triggered_at: str
    channels: tuple[str, ...]


def _bbox_overlaps(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    a_min_x, a_min_y, a_max_x, a_max_y = a
    b_min_x, b_min_y, b_max_x, b_max_y = b
    return not (
        a_max_x < b_min_x
        or b_max_x < a_min_x
        or a_max_y < b_min_y
        or b_max_y < a_min_y
    )


def _classify_severity(measured: float, threshold: float) -> str:
    if measured >= threshold * 3:
        return "critical"
    if measured >= threshold * 2:
        return "high"
    return "medium"


class AlertGenerator:
    """
    Subscription-driven alert generator with cooldown deduplication.
    """

    def __init__(
        self,
        subscriptions: Optional[Iterable[Subscription]] = None,
        alert_log_path: Optional[Union[str, Path]] = None,
        delivery: Optional[dict[str, DeliveryFn]] = None,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._subscriptions: list[Subscription] = list(subscriptions or [])
        self.alert_log_path = Path(alert_log_path) if alert_log_path else _DEFAULT_ALERT_LOG
        self._delivery = dict(delivery or {})
        self._lock = threading.Lock()
        self._last_fired: dict[tuple[int, str], datetime] = {}
        self._clock = clock

    def add_subscription(self, sub: Subscription) -> None:
        self._subscriptions.append(sub)

    def register_channel(self, name: str, fn: DeliveryFn) -> None:
        self._delivery[name] = fn

    def _persist(self, alert: Alert) -> None:
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.alert_log_path.open("a") as fh:
            fh.write(json.dumps(asdict(alert)) + "\n")

    def _in_cooldown(self, sub: Subscription, now: datetime) -> bool:
        key = (sub.org_id, sub.analysis_type)
        last = self._last_fired.get(key)
        if last is None:
            return False
        return now - last < timedelta(minutes=sub.cooldown_minutes)

    def _matches(
        self,
        sub: Subscription,
        analysis_type: str,
        bbox: tuple[float, float, float, float],
        measured_value: float,
    ) -> bool:
        if sub.analysis_type != analysis_type:
            return False
        if not _bbox_overlaps(sub.bbox, bbox):
            return False
        return measured_value >= sub.alert_threshold

    def evaluate(
        self,
        analysis_type: str,
        bbox: tuple[float, float, float, float],
        measured_value: float,
        summary: str = "",
    ) -> list[Alert]:
        now = self._clock()
        alerts: list[Alert] = []

        for sub in self._subscriptions:
            if not self._matches(sub, analysis_type, bbox, measured_value):
                continue
            if self._in_cooldown(sub, now):
                logger.debug(
                    "Skipping alert for org=%s in cooldown", sub.org_id
                )
                continue

            alert = Alert(
                alert_id=str(uuid.uuid4()),
                org_id=sub.org_id,
                analysis_type=analysis_type,
                region_bbox=bbox,
                severity=_classify_severity(measured_value, sub.alert_threshold),
                measured_value=float(measured_value),
                threshold=float(sub.alert_threshold),
                summary=summary or (
                    f"{analysis_type} signal {measured_value:.3f} "
                    f"exceeded threshold {sub.alert_threshold:.3f}"
                ),
                triggered_at=now.isoformat(),
                channels=tuple(sub.channels),
            )
            self._last_fired[(sub.org_id, sub.analysis_type)] = now
            self._persist(alert)
            self._dispatch(alert)
            alerts.append(alert)

        if alerts:
            logger.info("Fired %d alert(s) for analysis=%s", len(alerts), analysis_type)
        return alerts

    def _dispatch(self, alert: Alert) -> None:
        for channel in alert.channels:
            fn = self._delivery.get(channel)
            if fn is None:
                logger.warning(
                    "No delivery handler registered for channel '%s' (alert=%s)",
                    channel,
                    alert.alert_id,
                )
                continue
            try:
                fn(alert)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Delivery on channel '%s' failed for alert=%s",
                    channel,
                    alert.alert_id,
                )

    def iter_alerts(self) -> list[Alert]:
        if not self.alert_log_path.exists():
            return []
        out: list[Alert] = []
        with self.alert_log_path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["region_bbox"] = tuple(row["region_bbox"])
                row["channels"] = tuple(row["channels"])
                out.append(Alert(**row))
        return out


def log_channel(alert: Alert) -> None:
    """Default 'log' channel — writes the alert summary at WARNING level."""
    logger.warning(
        "ALERT [%s] org=%s analysis=%s severity=%s value=%.3f >= %.3f :: %s",
        alert.alert_id,
        alert.org_id,
        alert.analysis_type,
        alert.severity,
        alert.measured_value,
        alert.threshold,
        alert.summary,
    )
