"""SQLite bootstrap helpers for benchmark storage."""

from __future__ import annotations

from pathlib import Path

from sqlite_utils import Database

SCHEMA_PATH = Path(__file__).resolve().with_name("db.sql")


def open_benchmark_db(db_path: Path) -> Database:
    resolved_path = db_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    db = Database(resolved_path)
    db.conn.execute("PRAGMA foreign_keys = ON")
    db.conn.execute("PRAGMA busy_timeout = 5000")
    db.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    db.conn.commit()
    return db
