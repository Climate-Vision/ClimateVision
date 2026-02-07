import sqlite3
from pathlib import Path
from typing import Optional

from climatevision.config import Config

_DB_PATH: Optional[Path] = None
_INITIALIZED = False


def get_db_path() -> Path:
    global _DB_PATH
    if _DB_PATH is None:
        db_dir = Config.PROJECT_ROOT / "outputs"
        db_dir.mkdir(parents=True, exist_ok=True)
        _DB_PATH = db_dir / "climatevision.sqlite3"
    return _DB_PATH


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                bbox TEXT NULL,
                start_date TEXT NULL,
                end_date TEXT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

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

    _INITIALIZED = True
