"""
API-key authentication: DB-backed validation + secure-by-default dev bypass.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from climatevision.api.auth import APIKeyAuth
from climatevision.api.main import create_app
from climatevision.db import create_organization, get_connection, init_db


def _fresh_org_key() -> str:
    try:
        init_db()
    except Exception:
        pass
    return create_organization(name="AuthTestNGO", org_type="ngo", contact_email="a@e.org")["api_key"]


class TestValidateKey:
    def test_real_db_key_authenticates(self):
        key = _fresh_org_key()
        org = APIKeyAuth().validate_key(key)
        assert org is not None
        assert org["name"] == "AuthTestNGO"

    def test_unknown_key_rejected(self):
        assert APIKeyAuth().validate_key("cv_not_a_real_key") is None

    def test_non_cv_prefix_rejected(self):
        assert APIKeyAuth().validate_key("totally-bogus") is None

    def test_deactivated_org_key_rejected(self):
        key = _fresh_org_key()
        with get_connection() as conn:
            conn.execute("UPDATE organizations SET active = 0 WHERE api_key = ?", (key,))
        # Use a fresh handler so the 5-min cache doesn't mask deactivation.
        assert APIKeyAuth().validate_key(key) is None


class TestDevKeyGate:
    def test_cv_dev_rejected_by_default(self, monkeypatch):
        monkeypatch.delenv("CLIMATEVISION_ALLOW_DEV_KEY", raising=False)
        assert APIKeyAuth().validate_key("cv_dev") is None

    def test_cv_dev_allowed_when_flag_set(self, monkeypatch):
        monkeypatch.setenv("CLIMATEVISION_ALLOW_DEV_KEY", "1")
        org = APIKeyAuth().validate_key("cv_dev")
        assert org is not None and org.get("demo") is True


class TestEndToEndAuth:
    def test_predict_with_real_key(self):
        key = _fresh_org_key()
        client = TestClient(create_app())
        r = client.post(
            "/api/predict",
            headers={"X-API-Key": key},
            json={
                "kind": "gee",
                "analysis_type": "flooding_sar",
                "bbox": [36.7, -1.4, 37.0, -1.1],
                "start_date": "2024-04-01",
                "end_date": "2024-04-10",
            },
        )
        assert r.status_code == 200

    def test_predict_rejects_cv_dev_without_flag(self, monkeypatch):
        monkeypatch.delenv("CLIMATEVISION_ALLOW_DEV_KEY", raising=False)
        client = TestClient(create_app())
        r = client.post(
            "/api/predict",
            headers={"X-API-Key": "cv_dev"},
            json={"kind": "gee", "analysis_type": "flooding_sar",
                  "bbox": [36.7, -1.4, 37.0, -1.1],
                  "start_date": "2024-04-01", "end_date": "2024-04-10"},
        )
        assert r.status_code == 401
