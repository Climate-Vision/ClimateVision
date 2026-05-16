"""Tests for alert delivery worker and pending endpoint."""

from unittest import mock

import pytest
from fastapi.testclient import TestClient

from climatevision.workers.alert_delivery import (
    process_alert_delivery,
    send_email_smtp,
    send_webhook,
)


class TestPendingEndpoint:
    """Integration tests for GET /api/organizations/{org_id}/alerts/pending."""

    @pytest.fixture
    def org(self, client: TestClient) -> dict:
        """Create and return a test organization."""
        response = client.post(
            "/api/organizations",
            json={
                "name": "Test NGO",
                "type": "ngo",
                "contact_email": "test@example.com",
            },
            headers={"X-API-Key": "cv_dev"},
        )
        assert response.status_code == 200
        return response.json()

    def test_pending_endpoint_returns_undelivered(
        self, client: TestClient, org: dict
    ) -> None:
        """Pending endpoint should return only undelivered alerts."""
        org_id = org["id"]

        with mock.patch("climatevision.api.main.process_alert_delivery"):
            resp = client.post(
                f"/api/organizations/{org_id}/alerts",
                json={
                    "alert_type": "deforestation",
                    "severity": "high",
                    "title": "Forest loss detected",
                    "message": "20% forest cover lost.",
                },
                headers={"X-API-Key": "cv_dev"},
            )
            assert resp.status_code == 200
            alert = resp.json()

        resp = client.get(
            f"/api/organizations/{org_id}/alerts/pending",
            headers={"X-API-Key": "cv_dev"},
        )
        assert resp.status_code == 200
        pending = resp.json()
        assert len(pending) == 1
        assert pending[0]["id"] == alert["id"]
        assert pending[0]["delivered"] is False

        client.post(
            f"/api/alerts/{alert['id']}/deliver",
            headers={"X-API-Key": "cv_dev"},
        )

        resp = client.get(
            f"/api/organizations/{org_id}/alerts/pending",
            headers={"X-API-Key": "cv_dev"},
        )
        assert resp.status_code == 200
        pending = resp.json()
        assert len(pending) == 0

    def test_create_alert_triggers_background_delivery(
        self, client: TestClient, org: dict
    ) -> None:
        """Creating an alert should enqueue a BackgroundTask."""
        org_id = org["id"]

        with mock.patch("climatevision.api.main.process_alert_delivery") as mock_deliver:
            resp = client.post(
                f"/api/organizations/{org_id}/alerts",
                json={
                    "alert_type": "flooding",
                    "severity": "critical",
                    "title": "Flood alert",
                    "message": "Severe flooding detected.",
                },
                headers={"X-API-Key": "cv_dev"},
            )
            assert resp.status_code == 200

        mock_deliver.assert_called_once()
        alert_id = mock_deliver.call_args[0][0]
        assert isinstance(alert_id, int)


class TestEmailDelivery:
    """Unit tests for SMTP email delivery."""

    def test_email_delivery_success(self) -> None:
        """SMTP configured and server accepts the message."""
        env = {
            "SMTP_HOST": "smtp.example.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "user",
            "SMTP_PASS": "pass",
            "SMTP_FROM": "from@example.com",
        }
        with mock.patch.dict("os.environ", env, clear=False), mock.patch(
            "climatevision.workers.alert_delivery.smtplib.SMTP"
        ) as mock_smtp:
            instance = mock_smtp.return_value.__enter__.return_value
            result = send_email_smtp("to@example.com", "Subject", "Body")
            assert result is True
            instance.starttls.assert_called_once()
            instance.login.assert_called_once_with("user", "pass")
            instance.send_message.assert_called_once()

    def test_email_delivery_skips_when_not_configured(self) -> None:
        """When SMTP_HOST is unset, the function returns False gracefully."""
        with mock.patch.dict("os.environ", {"SMTP_HOST": ""}, clear=False):
            result = send_email_smtp("to@example.com", "Subject", "Body")
            assert result is False


class TestWebhookDelivery:
    """Unit tests for HTTP webhook delivery."""

    def test_webhook_delivery_success(self) -> None:
        """Webhook endpoint returns 2xx."""
        with mock.patch(
            "climatevision.workers.alert_delivery.requests.post"
        ) as mock_post:
            mock_post.return_value.status_code = 200
            result = send_webhook("https://example.com/hook", {"key": "value"})
            assert result is True
            mock_post.assert_called_once_with(
                "https://example.com/hook",
                json={"key": "value"},
                timeout=30,
            )

    def test_webhook_delivery_failure(self) -> None:
        """Webhook endpoint returns 5xx."""
        with mock.patch(
            "climatevision.workers.alert_delivery.requests.post"
        ) as mock_post:
            mock_post.return_value.status_code = 500
            result = send_webhook("https://example.com/hook", {"key": "value"})
            assert result is False


class TestProcessAlertDelivery:
    """Unit tests for the main delivery orchestrator."""

    def test_skip_already_delivered(self) -> None:
        """Alerts already marked delivered should not be re-processed."""
        alert = {"id": 1, "delivered": 1, "organization_id": 1, "subscription_id": None}

        with mock.patch(
            "climatevision.workers.alert_delivery.get_alert", return_value=alert
        ), mock.patch(
            "climatevision.workers.alert_delivery.send_email_smtp"
        ) as mock_email:
            process_alert_delivery(1)
            mock_email.assert_not_called()

    def test_delivery_retry_on_failure(self) -> None:
        """Failed delivery increments attempts and retries with backoff."""
        alert = {
            "id": 1,
            "organization_id": 1,
            "subscription_id": 1,
            "alert_type": "deforestation",
            "severity": "high",
            "title": "Test",
            "message": "Msg",
            "details": None,
            "created_at": "2024-01-01T00:00:00",
            "delivered": 0,
        }
        org = {"id": 1, "contact_email": "test@example.com"}
        sub = {"id": 1, "notification_channel": "email", "webhook_url": None}

        with (
            mock.patch(
                "climatevision.workers.alert_delivery.get_alert", return_value=alert
            ),
            mock.patch(
                "climatevision.workers.alert_delivery.get_organization",
                return_value=org,
            ),
            mock.patch(
                "climatevision.workers.alert_delivery.get_subscription",
                return_value=sub,
            ),
            mock.patch(
                "climatevision.workers.alert_delivery.send_email_smtp",
                return_value=False,
            ) as mock_email,
            mock.patch(
                "climatevision.workers.alert_delivery.increment_delivery_attempts"
            ) as mock_incr,
            mock.patch(
                "climatevision.workers.alert_delivery.time.sleep"
            ) as mock_sleep,
            mock.patch(
                "climatevision.workers.alert_delivery.mark_alert_delivered"
            ) as mock_mark,
        ):
            process_alert_delivery(1)

            assert mock_email.call_count == 3
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(60)
            mock_sleep.assert_any_call(120)
            assert mock_incr.call_count == 3
            mock_mark.assert_not_called()

    def test_max_retries_exhausted(self) -> None:
        """After 3 failures, alert remains undelivered."""
        alert = {
            "id": 1,
            "organization_id": 1,
            "subscription_id": 1,
            "alert_type": "deforestation",
            "severity": "high",
            "title": "Test",
            "message": "Msg",
            "details": None,
            "created_at": "2024-01-01T00:00:00",
            "delivered": 0,
        }
        org = {"id": 1, "contact_email": "test@example.com"}
        sub = {"id": 1, "notification_channel": "email", "webhook_url": None}

        with (
            mock.patch(
                "climatevision.workers.alert_delivery.get_alert", return_value=alert
            ),
            mock.patch(
                "climatevision.workers.alert_delivery.get_organization",
                return_value=org,
            ),
            mock.patch(
                "climatevision.workers.alert_delivery.get_subscription",
                return_value=sub,
            ),
            mock.patch(
                "climatevision.workers.alert_delivery.send_email_smtp",
                return_value=False,
            ),
            mock.patch(
                "climatevision.workers.alert_delivery.increment_delivery_attempts"
            ) as mock_incr,
            mock.patch("climatevision.workers.alert_delivery.time.sleep"),
            mock.patch(
                "climatevision.workers.alert_delivery.mark_alert_delivered"
            ) as mock_mark,
        ):
            process_alert_delivery(1)

            assert mock_incr.call_count == 3
            mock_mark.assert_not_called()

    def test_exponential_backoff_timing(self) -> None:
        """Verify backoff delays are 60 s and 120 s."""
        alert = {
            "id": 1,
            "organization_id": 1,
            "subscription_id": 1,
            "alert_type": "deforestation",
            "severity": "high",
            "title": "Test",
            "message": "Msg",
            "details": None,
            "created_at": "2024-01-01T00:00:00",
            "delivered": 0,
        }
        org = {"id": 1, "contact_email": "test@example.com"}
        sub = {"id": 1, "notification_channel": "email", "webhook_url": None}

        with (
            mock.patch(
                "climatevision.workers.alert_delivery.get_alert", return_value=alert
            ),
            mock.patch(
                "climatevision.workers.alert_delivery.get_organization",
                return_value=org,
            ),
            mock.patch(
                "climatevision.workers.alert_delivery.get_subscription",
                return_value=sub,
            ),
            mock.patch(
                "climatevision.workers.alert_delivery.send_email_smtp",
                return_value=False,
            ),
            mock.patch(
                "climatevision.workers.alert_delivery.increment_delivery_attempts"
            ),
            mock.patch(
                "climatevision.workers.alert_delivery.time.sleep"
            ) as mock_sleep,
            mock.patch("climatevision.workers.alert_delivery.mark_alert_delivered"),
        ):
            process_alert_delivery(1)

            delays = [call[0][0] for call in mock_sleep.call_args_list]
            assert delays == [60, 120]
