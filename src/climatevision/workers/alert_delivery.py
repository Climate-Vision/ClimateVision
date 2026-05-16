"""Alert delivery worker with SMTP and webhook channels.

Triggered via FastAPI BackgroundTasks on alert creation.
Retries up to 3 times with exponential backoff (60 s, 120 s).
"""

import logging
import os
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests

from climatevision.db import (
    get_alert,
    get_organization,
    get_subscription,
    increment_delivery_attempts,
    mark_alert_delivered,
)

logger = logging.getLogger(__name__)

# Exponential backoff delays in seconds between attempts.
_BACKOFF_DELAYS = [60, 120]


def _smtp_configured() -> bool:
    """Check whether the minimum SMTP environment variables are set."""
    return bool(os.getenv("SMTP_HOST") and os.getenv("SMTP_USER"))


def send_email_smtp(to_email: str, subject: str, body: str) -> bool:
    """Send an alert email via SMTP using environment credentials.

    Args:
        to_email: Recipient address.
        subject: Email subject line.
        body: Plain-text body.

    Returns:
        True if the SMTP server accepted the message, otherwise False.
    """
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    from_addr = os.getenv("SMTP_FROM", "alerts@climatevision.dev")

    if not host or not user:
        logger.warning("SMTP not configured — skipping email delivery")
        return False

    msg = MIMEMultipart()
    msg["From"] = from_addr
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls()
            if password:
                server.login(user, password)
            server.send_message(msg)
        logger.info("Email delivered to %s", to_email)
        return True
    except Exception:
        logger.exception("Email delivery failed for %s", to_email)
        return False


def send_webhook(url: str, payload: dict) -> bool:
    """POST an alert payload to a webhook URL.

    Args:
        url: Webhook endpoint.
        payload: JSON-serializable dict with alert data.

    Returns:
        True when the endpoint responds with a 2xx status, otherwise False.
    """
    try:
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code < 400:
            logger.info("Webhook accepted by %s", url)
            return True
        logger.warning("Webhook rejected by %s — status %s", url, resp.status_code)
        return False
    except Exception:
        logger.exception("Webhook delivery failed for %s", url)
        return False


def _build_email_body(alert: dict) -> str:
    """Compose a plain-text email body from an alert row."""
    lines = [
        f"Alert: {alert['title']}",
        f"Type: {alert['alert_type']}",
        f"Severity: {alert['severity']}",
        "",
        alert["message"],
        "",
    ]
    if alert.get("details"):
        lines.append(f"Details: {alert['details']}")
    return "\n".join(lines)


def _build_webhook_payload(alert: dict, org_id: int) -> dict:
    """Build the JSON payload sent to webhook endpoints."""
    return {
        "alert_id": alert["id"],
        "organization_id": org_id,
        "alert_type": alert["alert_type"],
        "severity": alert["severity"],
        "title": alert["title"],
        "message": alert["message"],
        "details": alert.get("details"),
        "created_at": alert["created_at"],
    }


def process_alert_delivery(alert_id: int) -> None:
    """Deliver an alert via its configured channel with retries.

    Reads the alert and its linked subscription/organization from the
    database, determines the notification channel, and attempts delivery
    up to three times with exponential backoff.

    Args:
        alert_id: Primary key of the alert to deliver.
    """
    alert_row = get_alert(alert_id)
    if alert_row is None:
        logger.error("Alert %s not found", alert_id)
        return

    alert = dict(alert_row)

    if alert["delivered"]:
        logger.info("Alert %s already delivered — skipping", alert_id)
        return

    org_row = get_organization(alert["organization_id"])
    if org_row is None:
        logger.error("Organization %s for alert %s not found", alert["organization_id"], alert_id)
        return

    org = dict(org_row)

    subscription = None
    if alert["subscription_id"] is not None:
        sub_row = get_subscription(alert["subscription_id"])
        if sub_row is not None:
            subscription = dict(sub_row)

    channel = "email"
    if subscription:
        channel = subscription["notification_channel"]

    for attempt in range(3):
        success = False

        if channel == "email":
            contact = org.get("contact_email")
            if contact:
                body = _build_email_body(dict(alert))
                success = send_email_smtp(
                    to_email=contact,
                    subject=f"[ClimateVision Alert] {alert['title']}",
                    body=body,
                )
            else:
                logger.warning(
                    "Organization %s has no contact_email — skipping email delivery",
                    org["id"],
                )
                return

        elif channel == "webhook":
            webhook_url = None
            if subscription:
                webhook_url = subscription.get("webhook_url")
            if webhook_url:
                payload = _build_webhook_payload(dict(alert), org["id"])
                success = send_webhook(url=webhook_url, payload=payload)
            else:
                logger.warning(
                    "Subscription for alert %s has no webhook_url — skipping webhook delivery",
                    alert_id,
                )
                return

        elif channel == "api":
            # API delivery is implicit — the alert exists in the DB and is
            # already queryable via the REST endpoints.
            logger.info("API channel — alert %s is already queryable", alert_id)
            mark_alert_delivered(alert_id)
            return

        else:
            logger.warning("Unknown notification channel '%s' for alert %s", channel, alert_id)
            return

        if success:
            mark_alert_delivered(alert_id)
            logger.info("Alert %s delivered successfully on attempt %d", alert_id, attempt + 1)
            return

        # Record the failed attempt.
        increment_delivery_attempts(alert_id)
        logger.warning("Alert %s delivery attempt %d failed", alert_id, attempt + 1)

        # Exponential backoff before the next retry (if any remain).
        if attempt < 2:
            delay = _BACKOFF_DELAYS[attempt]
            logger.info("Retrying alert %s in %d seconds", alert_id, delay)
            time.sleep(delay)

    logger.error("Alert %s delivery failed after 3 attempts", alert_id)
