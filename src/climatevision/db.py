"""
ClimateVision Database Module

Manages SQLite database for storing:
- Analysis runs and results
- Organization (NGO) data and subscriptions
- Alerts and notifications
"""

import secrets
import sqlite3
from pathlib import Path
from typing import Optional, Any
from datetime import datetime, timezone

from climatevision.config import Config

_DB_PATH: Optional[Path] = None
_INITIALIZED = False


def get_db_path() -> Path:
    """Get the path to the SQLite database file."""
    global _DB_PATH
    if _DB_PATH is None:
        db_dir = Config.PROJECT_ROOT / "outputs"
        db_dir.mkdir(parents=True, exist_ok=True)
        _DB_PATH = db_dir / "climatevision.sqlite3"
    return _DB_PATH


def get_connection() -> sqlite3.Connection:
    """Create a new database connection with foreign keys enabled."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _utc_now_iso() -> str:
    """Get current UTC time as ISO format string."""
    return datetime.now(timezone.utc).isoformat()


def generate_api_key() -> str:
    """Generate a secure API key for organizations."""
    return f"cv_{secrets.token_urlsafe(32)}"


def init_db() -> None:
    """Initialize the database schema with all required tables."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    with get_connection() as conn:
        # ===== Core Analysis Tables =====
        
        # Runs table - stores analysis run metadata
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                analysis_type TEXT NOT NULL DEFAULT 'deforestation',
                bbox TEXT NULL,
                start_date TEXT NULL,
                end_date TEXT NULL,
                organization_id INTEGER NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(organization_id) REFERENCES organizations(id) ON DELETE SET NULL
            )
            """
        )

        # Results table - stores inference results
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                mask_path TEXT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
            )
            """
        )

        # Legacy alerts table (kept for backward compatibility)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                delivered INTEGER NOT NULL,
                target TEXT NULL,
                detail TEXT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
            )
            """
        )

        # ===== Organization (NGO) Tables =====
        
        # Organizations table - stores NGO/partner information
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS organizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'ngo',
                description TEXT NULL,
                logo_url TEXT NULL,
                website_url TEXT NULL,
                contact_email TEXT NULL,
                contact_phone TEXT NULL,
                address TEXT NULL,
                regions_of_interest TEXT NULL,
                alert_preferences TEXT NULL,
                api_key TEXT UNIQUE,
                api_key_created_at TEXT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        # Organization subscriptions - regions monitored by organizations
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS organization_subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                organization_id INTEGER NOT NULL,
                name TEXT NULL,
                description TEXT NULL,
                bbox TEXT NOT NULL,
                analysis_types TEXT NOT NULL DEFAULT '["deforestation"]',
                alert_threshold REAL NOT NULL DEFAULT 5.0,
                notification_channel TEXT NOT NULL DEFAULT 'email',
                webhook_url TEXT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                last_checked_at TEXT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(organization_id) REFERENCES organizations(id) ON DELETE CASCADE
            )
            """
        )

        # Organization alerts - alerts sent to organizations
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS organization_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                organization_id INTEGER NOT NULL,
                subscription_id INTEGER NULL,
                run_id INTEGER NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'medium',
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                details TEXT NULL,
                delivered INTEGER NOT NULL DEFAULT 0,
                delivery_attempts INTEGER NOT NULL DEFAULT 0,
                delivered_at TEXT NULL,
                acknowledged INTEGER NOT NULL DEFAULT 0,
                acknowledged_at TEXT NULL,
                acknowledged_by TEXT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
                FOREIGN KEY(subscription_id) REFERENCES organization_subscriptions(id) ON DELETE SET NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE SET NULL
            )
            """
        )

        # Organization members - users belonging to organizations
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS organization_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                organization_id INTEGER NOT NULL,
                email TEXT NOT NULL,
                name TEXT NULL,
                role TEXT NOT NULL DEFAULT 'member',
                active INTEGER NOT NULL DEFAULT 1,
                invited_at TEXT NOT NULL,
                joined_at TEXT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
                UNIQUE(organization_id, email)
            )
            """
        )

        # Organization reports - generated reports for organizations
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS organization_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                organization_id INTEGER NOT NULL,
                subscription_id INTEGER NULL,
                report_type TEXT NOT NULL DEFAULT 'summary',
                format TEXT NOT NULL DEFAULT 'json',
                title TEXT NOT NULL,
                description TEXT NULL,
                parameters TEXT NULL,
                file_path TEXT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                error_message TEXT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT NULL,
                FOREIGN KEY(organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
                FOREIGN KEY(subscription_id) REFERENCES organization_subscriptions(id) ON DELETE SET NULL
            )
            """
        )

        # ===== Migrations for existing databases =====

        existing_run_cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        if "analysis_type" not in existing_run_cols:
            conn.execute(
                "ALTER TABLE runs ADD COLUMN analysis_type TEXT NOT NULL DEFAULT 'deforestation'"
            )
        if "organization_id" not in existing_run_cols:
            conn.execute(
                "ALTER TABLE runs ADD COLUMN organization_id INTEGER NULL"
            )

        # ===== Indexes for Performance =====

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_analysis_type ON runs(analysis_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_organization ON runs(organization_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_org_subscriptions_org ON organization_subscriptions(organization_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_org_alerts_org ON organization_alerts(organization_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_org_alerts_delivered ON organization_alerts(delivered)"
        )

    _INITIALIZED = True


# ===== Organization CRUD Operations =====

def create_organization(
    name: str,
    org_type: str = "ngo",
    description: Optional[str] = None,
    contact_email: Optional[str] = None,
    website_url: Optional[str] = None,
    regions_of_interest: Optional[list] = None,
) -> dict[str, Any]:
    """Create a new organization and return its data with API key."""
    api_key = generate_api_key()
    now = _utc_now_iso()
    
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO organizations (
                name, type, description, contact_email, website_url,
                regions_of_interest, api_key, api_key_created_at,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                org_type,
                description,
                contact_email,
                website_url,
                str(regions_of_interest) if regions_of_interest else None,
                api_key,
                now,
                now,
                now,
            ),
        )
        org_id = cursor.lastrowid
    
    return {
        "id": org_id,
        "name": name,
        "type": org_type,
        "api_key": api_key,
        "created_at": now,
    }


def get_organization(org_id: int) -> Optional[sqlite3.Row]:
    """Get an organization by ID."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM organizations WHERE id = ?", (org_id,)
        ).fetchone()


def get_organization_by_api_key(api_key: str) -> Optional[sqlite3.Row]:
    """Get an organization by API key."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM organizations WHERE api_key = ? AND active = 1",
            (api_key,),
        ).fetchone()


def list_organizations(
    active_only: bool = True,
    org_type: Optional[str] = None,
    limit: int = 100,
) -> list[sqlite3.Row]:
    """List organizations with optional filtering."""
    query = "SELECT * FROM organizations WHERE 1=1"
    params: list = []
    
    if active_only:
        query += " AND active = 1"
    if org_type:
        query += " AND type = ?"
        params.append(org_type)
    
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    
    with get_connection() as conn:
        return conn.execute(query, params).fetchall()


# ===== Subscription CRUD Operations =====

def create_subscription(
    organization_id: int,
    bbox: list[float],
    name: Optional[str] = None,
    analysis_types: Optional[list[str]] = None,
    alert_threshold: float = 5.0,
    notification_channel: str = "email",
    webhook_url: Optional[str] = None,
) -> dict[str, Any]:
    """Create a new subscription for an organization."""
    import json
    now = _utc_now_iso()
    
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO organization_subscriptions (
                organization_id, name, bbox, analysis_types,
                alert_threshold, notification_channel, webhook_url,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                organization_id,
                name,
                json.dumps(bbox),
                json.dumps(analysis_types or ["deforestation"]),
                alert_threshold,
                notification_channel,
                webhook_url,
                now,
                now,
            ),
        )
        sub_id = cursor.lastrowid
    
    return {
        "id": sub_id,
        "organization_id": organization_id,
        "bbox": bbox,
        "created_at": now,
    }


def get_subscriptions_for_organization(
    organization_id: int,
    active_only: bool = True,
) -> list[sqlite3.Row]:
    """Get all subscriptions for an organization."""
    query = "SELECT * FROM organization_subscriptions WHERE organization_id = ?"
    params: list = [organization_id]
    
    if active_only:
        query += " AND active = 1"
    
    query += " ORDER BY created_at DESC"
    
    with get_connection() as conn:
        return conn.execute(query, params).fetchall()


# ===== Alert Operations =====

def create_organization_alert(
    organization_id: int,
    alert_type: str,
    title: str,
    message: str,
    severity: str = "medium",
    subscription_id: Optional[int] = None,
    run_id: Optional[int] = None,
    details: Optional[str] = None,
) -> int:
    """Create a new alert for an organization."""
    now = _utc_now_iso()
    
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO organization_alerts (
                organization_id, subscription_id, run_id,
                alert_type, severity, title, message, details,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                organization_id,
                subscription_id,
                run_id,
                alert_type,
                severity,
                title,
                message,
                details,
                now,
            ),
        )
        return cursor.lastrowid


def get_alerts_for_organization(
    organization_id: int,
    undelivered_only: bool = False,
    unacknowledged_only: bool = False,
    limit: int = 50,
) -> list[sqlite3.Row]:
    """Get alerts for an organization with optional filtering."""
    query = "SELECT * FROM organization_alerts WHERE organization_id = ?"
    params: list = [organization_id]
    
    if undelivered_only:
        query += " AND delivered = 0"
    if unacknowledged_only:
        query += " AND acknowledged = 0"
    
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    
    with get_connection() as conn:
        return conn.execute(query, params).fetchall()


def acknowledge_alert(alert_id: int, acknowledged_by: Optional[str] = None) -> bool:
    """Mark an alert as acknowledged."""
    now = _utc_now_iso()
    
    with get_connection() as conn:
        cursor = conn.execute(
            """
            UPDATE organization_alerts
            SET acknowledged = 1, acknowledged_at = ?, acknowledged_by = ?
            WHERE id = ?
            """,
            (now, acknowledged_by, alert_id),
        )
        return cursor.rowcount > 0


def mark_alert_delivered(alert_id: int) -> bool:
    """Mark an alert as delivered."""
    now = _utc_now_iso()
    
    with get_connection() as conn:
        cursor = conn.execute(
            """
            UPDATE organization_alerts
            SET delivered = 1, delivered_at = ?, delivery_attempts = delivery_attempts + 1
            WHERE id = ?
            """,
            (now, alert_id),
        )
        return cursor.rowcount > 0


def get_alert(alert_id: int) -> Optional[sqlite3.Row]:
    """Get a single alert by ID."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM organization_alerts WHERE id = ?", (alert_id,)
        ).fetchone()


def get_subscription(sub_id: int) -> Optional[sqlite3.Row]:
    """Get a single subscription by ID."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM organization_subscriptions WHERE id = ?", (sub_id,)
        ).fetchone()


def get_pending_alerts(
    organization_id: int,
    limit: int = 50,
) -> list[sqlite3.Row]:
    """Get undelivered alerts for an organization."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT * FROM organization_alerts
            WHERE organization_id = ? AND delivered = 0
            ORDER BY created_at DESC LIMIT ?
            """,
            (organization_id, limit),
        ).fetchall()


def increment_delivery_attempts(alert_id: int) -> bool:
    """Increment the delivery attempts counter for an alert."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            UPDATE organization_alerts
            SET delivery_attempts = delivery_attempts + 1
            WHERE id = ?
            """,
            (alert_id,),
        )
        return cursor.rowcount > 0
