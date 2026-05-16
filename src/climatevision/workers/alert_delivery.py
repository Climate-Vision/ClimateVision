"""
Alert Delivery Worker

Background worker that processes undelivered alerts by attempting
delivery through the configured notification channels (email via SMTP
or HTTP webhook POST). Implements exponential-backoff retry logic.

Architecture:
- Runs as a FastAPI lifespan background task
- Polls for undelivered alerts every 60 seconds
- Max 3 delivery attempts per alert with exponential backoff
- Delivery status is persisted to the organization_alerts table

Configuration (via environment variables):
- SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_USE_TLS
- ALERT_DELIVERY_MAX_RETRIES (default: 3)
- ALERT_DELIVERY_POLL_INTERVAL_SECONDS (default: 60)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

import urllib.request
import urllib.error

from climatevision.db import (
    get_connection,
    get_pending_alerts,
    mark_alert_delivered,
    increment_delivery_attempt,
    mark_alert_failed,
)

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_POLL_INTERVAL_SECONDS = 60


def _get_smtp_config() -> dict:
    """Read SMTP configuration from environment variables."""
    return {
        "host": os.getenv("SMTP_HOST", "localhost"),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "username": os.getenv("SMTP_USERNAME", ""),
        "password": os.getenv("SMTP_PASSWORD", ""),
        "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() == "true",
        "from_email": os.getenv("SMTP_FROM_EMAIL", os.getenv("SMTP_USERNAME", "alerts@climatevision.org")),
    }


def _send_email(
    to_email: str,
    subject: str,
    body: str,
    smtp_config: dict,
) -> bool:
    """Send an email alert via SMTP. Returns True on success."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[ClimateVision Alert] {subject}"
        msg["From"] = smtp_config["from_email"]
        msg["To"] = to_email

        # Plain-text fallback + HTML version
        text_part = MIMEText(body, "plain", "utf-8")
        html_body = f"""\
<html>
  <body style="font-family: system-ui, sans-serif; max-width: 600px; margin: 0 auto;">
    <div style="background: #065f46; color: white; padding: 16px; border-radius: 8px 8px 0 0;">
      <h2 style="margin: 0;">🌍 ClimateVision Alert</h2>
    </div>
    <div style="padding: 16px; border: 1px solid #e5e7eb; border-top: none; border-radius: 0 0 8px 8px;">
      <h3>{subject}</h3>
      <pre style="white-space: pre-wrap; font-family: system-ui, sans-serif;">{body}</pre>
    </div>
  </body>
</html>"""
        html_part = MIMEText(html_body, "html", "utf-8")
        msg.attach(text_part)
        msg.attach(html_part)

        context = ssl.create_default_context()
        if smtp_config["use_tls"]:
            with smtplib.SMTP(smtp_config["host"], smtp_config["port"], timeout=15) as server:
                server.starttls(context=context)
                if smtp_config["username"]:
                    server.login(smtp_config["username"], smtp_config["password"])
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_config["host"], smtp_config["port"], timeout=15) as server:
                if smtp_config["username"]:
                    server.login(smtp_config["username"], smtp_config["password"])
                server.send_message(msg)

        logger.info("Email sent to %s: %s", to_email, subject)
        return True
    except Exception as exc:
        logger.warning("Email delivery failed to %s: %s", to_email, exc)
        return False


def _send_webhook(webhook_url: str, payload: dict) -> bool:
    """Send an alert via HTTP POST webhook. Returns True on success."""
    try:
        data = urllib.parse.urlencode(payload).encode("utf-8") if payload else b"{}"
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "ClimateVision-AlertDelivery/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            if 200 <= resp.status < 300:
                logger.info("Webhook delivered to %s (status %s)", webhook_url, resp.status)
                return True
            logger.warning("Webhook delivery to %s returned status %s", webhook_url, resp.status)
            return False
    except urllib.error.URLError as exc:
        logger.warning("Webhook delivery failed to %s: %s", webhook_url, exc)
        return False
    except Exception as exc:
        logger.warning("Webhook delivery failed to %s: %s", webhook_url, exc)
        return False


def _get_organization_contact_email(org_id: int) -> Optional[str]:
    """Get the contact email for an organization."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT contact_email FROM organizations WHERE id = ? AND active = 1",
            (org_id,),
        ).fetchone()
        if row and row["contact_email"]:
            return row["contact_email"]
    return None


def _get_subscription_webhook_url(sub_id: Optional[int]) -> Optional[str]:
    """Get the webhook URL for a subscription."""
    if sub_id is None:
        return None
    with get_connection() as conn:
        row = conn.execute(
            "SELECT webhook_url, notification_channel FROM organization_subscriptions WHERE id = ?",
            (sub_id,),
        ).fetchone()
        if row and row["webhook_url"] and row["notification_channel"] == "webhook":
            return row["webhook_url"]
    return None


class AlertDeliveryWorker:
    """
    Background worker that polls for undelivered alerts and attempts delivery.

    Runs as an asyncio task, checking for pending alerts at a configurable
    interval and attempting delivery with exponential-backoff retry logic.
    """

    def __init__(
        self,
        poll_interval_seconds: int | None = None,
        max_retries: int | None = None,
    ):
        self.poll_interval = poll_interval_seconds or int(
            os.getenv("ALERT_DELIVERY_POLL_INTERVAL_SECONDS", str(DEFAULT_POLL_INTERVAL_SECONDS))
        )
        self.max_retries = max_retries or int(
            os.getenv("ALERT_DELIVERY_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))
        )
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the background worker task."""
        if self._task is not None:
            logger.warning("AlertDeliveryWorker already running")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "AlertDeliveryWorker started (poll_interval=%ds, max_retries=%d)",
            self.poll_interval,
            self.max_retries,
        )

    async def stop(self) -> None:
        """Stop the background worker task gracefully."""
        if self._task is None:
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=30)
        except asyncio.TimeoutError:
            logger.warning("AlertDeliveryWorker did not stop within 30s, cancelling")
            self._task.cancel()
        self._task = None
        logger.info("AlertDeliveryWorker stopped")

    async def _run_loop(self) -> None:
        """Main polling loop — runs until stop_event is set."""
        while not self._stop_event.is_set():
            try:
                await self._process_pending_alerts()
            except Exception:
                logger.exception("Error in alert delivery worker loop")
            # Wait with periodic stop check
            for _ in range(self.poll_interval):
                if self._stop_event.is_set():
                    return
                await asyncio.sleep(1)

    async def _process_pending_alerts(self) -> None:
        """
        Fetch all pending alerts and attempt delivery for each.

        Delivery strategy:
        1. Fetch alerts with delivery_attempts < max_retries and delivered=0
        2. For each alert, check the associated subscription's notification channel
        3. Attempt email delivery if channel is 'email' and org has contact_email
        4. Attempt webhook delivery if channel is 'webhook' and subscription has webhook_url
        5. Update delivery status on success or increment attempts on failure
        """
        alerts = get_pending_alerts(max_attempts=self.max_retries)
        if not alerts:
            return

        smtp_config = _get_smtp_config()
        delivered_count = 0
        failed_count = 0

        for alert in alerts:
            alert_id = alert["id"]
            org_id = alert["organization_id"]
            sub_id = alert["subscription_id"]
            title = alert["title"]
            message = alert["message"]
            attempts = alert["delivery_attempts"]

            logger.info(
                "Processing alert #%s (attempt %s/%s): %s",
                alert_id,
                attempts + 1,
                self.max_retries,
                title,
            )

            # Exponential backoff: wait before retry
            if attempts > 0:
                delay = min(2 ** attempts, 300)  # cap at 5 minutes
                logger.info("Backoff delay for alert #%s: %ds", alert_id, delay)
                await asyncio.sleep(delay)

            success = False

            # Check subscription for notification channel preference
            channel = "email"  # default
            webhook_url = _get_subscription_webhook_url(sub_id)
            if webhook_url:
                channel = "webhook"

            if channel == "webhook" and webhook_url:
                payload = {
                    "alert_id": alert_id,
                    "organization_id": org_id,
                    "title": title,
                    "message": message,
                    "severity": alert["severity"],
                    "alert_type": alert["alert_type"],
                    "created_at": alert["created_at"],
                }
                try:
                    json_payload = json.dumps(payload)
                    req = urllib.request.Request(
                        webhook_url,
                        data=json_payload.encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "User-Agent": "ClimateVision-AlertDelivery/1.0",
                        },
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        if 200 <= resp.status < 300:
                            success = True
                except Exception as exc:
                    logger.warning("Webhook delivery failed for alert #%s: %s", alert_id, exc)
            else:
                # Email delivery
                contact_email = _get_organization_contact_email(org_id)
                if contact_email:
                    # Run blocking SMTP call in thread to avoid blocking event loop
                    success = await asyncio.to_thread(
                        _send_email,
                        contact_email,
                        title,
                        message,
                        smtp_config,
                    )
                else:
                    logger.warning(
                        "Alert #%s: no contact email for organization #%s, skipping email delivery",
                        alert_id,
                        org_id,
                    )

            if success:
                mark_alert_delivered(alert_id)
                delivered_count += 1
                logger.info("Alert #%s delivered successfully", alert_id)
            else:
                is_last_attempt = attempts + 1 >= self.max_retries
                if is_last_attempt:
                    mark_alert_failed(alert_id)
                    failed_count += 1
                    logger.warning(
                        "Alert #%s failed after %s attempts, marked as failed",
                        alert_id,
                        self.max_retries,
                    )
                else:
                    increment_delivery_attempt(alert_id)
                    logger.info(
                        "Alert #%s delivery attempt %s failed, will retry",
                        alert_id,
                        attempts + 1,
                    )

        if delivered_count or failed_count:
            logger.info(
                "Alert delivery cycle complete: %s delivered, %s failed",
                delivered_count,
                failed_count,
            )
