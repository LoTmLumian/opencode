from __future__ import annotations

"""SQLite persistence layer (mirrors ``packages/core/src/database/database.ts`` + drizzle schema).

Learning-oriented rewrite: instead of an ORM (drizzle in the original) we use
Python's standard-library ``sqlite3`` with WAL journal mode and hand-written
SQL. Three tables mirror the original's schema:

- ``sessions``  — one row per session, holding its mutable metadata.
- ``messages``  — the append-only transcript of a session.
- ``events``    — bus/SSE events that need durability.

The ``data`` columns of ``messages``/``events`` are stored as JSON strings so
we can persist arbitrary Pydantic models without a schema migration per type.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

__all__ = ["Database"]


class Database:
    """A thin wrapper around a single SQLite connection.

    WAL mode is enabled so concurrent readers (e.g. the server streaming a
    session transcript) never block the writer (the agent loop), and so
    crash recovery is more robust.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # Ensure the parent directory exists before sqlite tries to create the
        # file, so we can persist to nested paths without error.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        # Row factory gives us dict-like access by column name.
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def _migrate(self) -> None:
        """Create tables if they do not exist (idempotent bootstrap)."""
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id               TEXT PRIMARY KEY,
                title            TEXT NOT NULL,
                agent            TEXT,
                model            TEXT,             -- JSON-serialized ModelRef
                parent_id        TEXT,
                permission       TEXT,             -- JSON-serialized permission rules
                created          REAL NOT NULL,
                updated          REAL NOT NULL,
                time_suspended   REAL,             -- set while a runner claims it
                resume_attempts  INT  NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                type        TEXT NOT NULL,
                data        TEXT NOT NULL,         -- JSON-serialized message
                time        REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, time);

            CREATE TABLE IF NOT EXISTS events (
                id          TEXT PRIMARY KEY,
                type        TEXT NOT NULL,
                session_id  TEXT,
                data        TEXT NOT NULL,         -- JSON-serialized event payload
                seq         INT,                   -- monotonic ordering per session
                time        REAL NOT NULL
            );
            """
        )
        self.conn.commit()

    # -- generic JSON helpers ---------------------------------------------

    @staticmethod
    def _dumps(data: Any) -> str:
        """Serialize an object to a JSON string (Pydantic models dump via dict)."""
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        return json.dumps(data, ensure_ascii=False, default=str)

    @staticmethod
    def _loads(raw: str) -> Any:
        """Deserialize a JSON string back to a Python object."""
        return json.loads(raw)

    def close(self) -> None:
        """Flush and close the underlying connection."""
        try:
            self.conn.commit()
        finally:
            self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- convenience accessors --------------------------------------------

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        """Run a parameterized statement and return the cursor."""
        return self.conn.execute(sql, params)

    def commit(self) -> None:
        self.conn.commit()

    def now(self) -> float:
        """Current wall-clock time, used to stamp rows consistently."""
        return time.time()
