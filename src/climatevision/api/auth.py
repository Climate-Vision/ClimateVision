"""
API Key Authentication for ClimateVision API.

Provides secure API key validation and organization-based
access control for all protected endpoints.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyAuth:
    """
    API Key authentication handler with organization context.

    Validates API keys and extracts organization information
    for request-scoped access control.
    """

    def __init__(self, db_connection=None):
        self._db = db_connection
        self._key_cache: dict[str, dict] = {}

    def generate_api_key(self, org_id: int, org_name: str) -> str:
        """
        Generate a new API key for an organization.

        Args:
            org_id: Organization ID
            org_name: Organization name

        Returns:
            New API key string (prefix + random bytes)
        """
        prefix = "cv_"
        random_part = secrets.token_urlsafe(32)
        api_key = f"{prefix}{random_part}"

        logger.info(
            "api_key_generated",
            extra={
                "org_id": org_id,
                "org_name": org_name,
                "key_prefix": api_key[:8],
            }
        )

        return api_key

    def hash_key(self, api_key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def validate_key(self, api_key: str) -> Optional[dict]:
        """
        Validate an API key and return organization context.

        Args:
            api_key: The API key to validate

        Returns:
            Organization dict if valid, None otherwise
        """
        if not api_key or not api_key.startswith("cv_"):
            return None

        # Development bypass — only when explicitly enabled. Off by default so
        # production is secure even if CLIMATEVISION_ALLOW_DEV_KEY is unset.
        if api_key == "cv_dev":
            if os.environ.get("CLIMATEVISION_ALLOW_DEV_KEY", "0") == "1":
                return {
                    "id": 0,
                    "name": "Development",
                    "demo": True,
                }
            logger.warning("cv_dev key rejected: dev bypass disabled in this environment.")
            return None

        # Check cache first
        key_hash = self.hash_key(api_key)
        if key_hash in self._key_cache:
            cached = self._key_cache[key_hash]
            if cached.get("expires_at", datetime.max) > datetime.utcnow():
                return cached.get("org")

        # Would query database in production
        # For now, return None to indicate key not found
        return None

    def revoke_key(self, api_key: str) -> bool:
        """
        Revoke an API key.

        Args:
            api_key: The API key to revoke

        Returns:
            True if revoked successfully
        """
        key_hash = self.hash_key(api_key)

        if key_hash in self._key_cache:
            del self._key_cache[key_hash]

        logger.info(
            "api_key_revoked",
            extra={"key_prefix": api_key[:8] if api_key else "unknown"}
        )

        return True


# Singleton instance
_auth_handler: Optional[APIKeyAuth] = None


def get_auth_handler() -> APIKeyAuth:
    """Get or create the API key auth handler."""
    global _auth_handler
    if _auth_handler is None:
        _auth_handler = APIKeyAuth()
    return _auth_handler


async def require_api_key(
    request: Request,
    api_key: Optional[str] = Security(API_KEY_HEADER)
) -> dict:
    """
    FastAPI dependency for requiring API key authentication.

    Usage:
        @app.get("/protected")
        async def protected_endpoint(org: dict = Depends(require_api_key)):
            return {"org_id": org["id"]}
    """
    if not api_key:
        logger.warning(
            "auth_failed",
            extra={
                "reason": "missing_api_key",
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
            }
        )
        raise HTTPException(
            status_code=401,
            detail="API key required. Include X-API-Key header."
        )

    auth = get_auth_handler()
    org = auth.validate_key(api_key)

    if not org:
        logger.warning(
            "auth_failed",
            extra={
                "reason": "invalid_api_key",
                "key_prefix": api_key[:8] if len(api_key) >= 8 else "short",
                "path": request.url.path,
            }
        )
        raise HTTPException(
            status_code=401,
            detail="Invalid API key."
        )

    # Attach org context to request state
    request.state.organization = org

    logger.info(
        "auth_success",
        extra={
            "org_id": org.get("id"),
            "org_name": org.get("name"),
            "path": request.url.path,
        }
    )

    return org


async def optional_api_key(
    request: Request,
    api_key: Optional[str] = Security(API_KEY_HEADER)
) -> Optional[dict]:
    """
    FastAPI dependency for optional API key authentication.

    Returns organization context if valid key provided, None otherwise.
    Does not raise exceptions for missing/invalid keys.
    """
    if not api_key:
        return None

    auth = get_auth_handler()
    return auth.validate_key(api_key)
