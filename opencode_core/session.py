from __future__ import annotations

"""Session storage + crash-recoverable claim/release (mirrors ``packages/core/src/session/store.ts``).

The `SessionStore` is the persistence facade over `Database`. Its headline
feature is the **claim / release** protocol used for crash recovery:

- A session is only processed by one runner at a time. Before running, a
  runner *claims* the session, writing a ``time_suspended`` timestamp and
  incrementing ``resume_attempts``.
- If the runner crashes (or is killed) mid-flight, the timestamp is never
  cleared. When a new runner starts and finds a stale claim (older than
  ``stale_claim_ms``), it treats the previous owner as dead and reclaims the
  session - this is how the original project resumes interrupted work.
- ``release`` clears ``time_suspended`` when a run finishes cleanly.

Also provided here: the default `system_prompt` used to prime the model.
"""

import json
import time
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, Field, TypeAdapter

from opencode_schema.model import ModelRef
from opencode_schema.session import (
    AssistantMessage,
    SessionMessage,
    SessionSummary,
    SystemMessage,
    ToolMessage,
    UserMessage,
    new_session_id,
)

from .database import Database

__all__ = ["SessionRecord", "SessionStore", "system_prompt"]


# Discriminated union over the concrete message subclasses, keyed on ``type``.
# This is what lets us restore the correct subclass (user/assistant/tool/system)
# when loading a stored transcript row.
_MessageUnion = Annotated[
    UserMessage | AssistantMessage | SystemMessage | ToolMessage,
    Field(discriminator="type"),
]
_message_adapter: TypeAdapter = TypeAdapter(_MessageUnion)


@dataclass
class SessionRecord:
    """A loaded session row (mirrors the original ``Session.Info``)."""

    id: str
    title: str
    agent: str | None
    model: ModelRef | None
    parent_id: str | None
    permission: list[Any]  # JSON-serialized permission rules
    created: float
    updated: float

    @property
    def model_str(self) -> str | None:
        """The model as a ``provider/model`` string, if any."""
        return str(self.model) if self.model is not None else None


class SessionStore:
    """Persistence for sessions and their message transcripts.

    Args:
        db: The underlying `Database` connection.
        stale_claim_ms: A claim older than this many milliseconds is considered
            stale (the previous owner is presumed dead) and can be reclaimed.
    """

    def __init__(self, db: Database, *, stale_claim_ms: float = 60_000) -> None:
        self.db = db
        self.stale_claim_ms = stale_claim_ms

    # -- sessions ---------------------------------------------------------

    def create(
        self,
        *,
        title: str,
        agent: str | None = None,
        model: str | ModelRef | None = None,
        parent_id: str | None = None,
        permission: list[Any] | None = None,
    ) -> str:
        """Insert a new session row and return its id.

        ``model`` may be given as a ``"provider/model"`` string or a `ModelRef`;
        it is persisted as JSON. ``permission`` (a list of rules) is persisted
        as JSON too.
        """
        sid = new_session_id()
        now = self.db.now()
        model_ref = ModelRef.parse(model) if isinstance(model, str) else model
        self.db.execute(
            """
            INSERT INTO sessions
                (id, title, agent, model, parent_id, permission, created, updated,
                 time_suspended, resume_attempts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 0)
            """,
            (
                sid,
                title,
                agent,
                self.db._dumps(model_ref) if model_ref is not None else None,
                parent_id,
                self.db._dumps(permission) if permission is not None else None,
                now,
                now,
            ),
        )
        self.db.commit()
        return sid

    def get(self, session_id: str) -> SessionRecord | None:
        """Load a single session by id, or ``None`` if it does not exist."""
        row = self.db.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list(self, limit: int = 50) -> list[SessionSummary]:
        """Return the most recently updated sessions, newest first."""
        rows = self.db.execute(
            "SELECT * FROM sessions ORDER BY updated DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_summary(row) for row in rows]

    def update_metadata(
        self,
        session_id: str,
        *,
        title: str | None = None,
        agent: str | None = None,
        model: str | ModelRef | None = None,
    ) -> None:
        """Patch a session's mutable metadata (title / agent / model).

        Only the fields explicitly passed (not ``None``) are updated. The
        ``updated`` timestamp is always bumped.
        """
        if title is not None:
            self.db.execute(
                "UPDATE sessions SET title = ?, updated = ? WHERE id = ?",
                (title, self.db.now(), session_id),
            )
        if agent is not None:
            self.db.execute(
                "UPDATE sessions SET agent = ?, updated = ? WHERE id = ?",
                (agent, self.db.now(), session_id),
            )
        if model is not None:
            model_ref = ModelRef.parse(model) if isinstance(model, str) else model
            self.db.execute(
                "UPDATE sessions SET model = ?, updated = ? WHERE id = ?",
                (self.db._dumps(model_ref), self.db.now(), session_id),
            )
        self.db.commit()

    # -- messages ---------------------------------------------------------

    def add_message(self, session_id: str, message: SessionMessage) -> None:
        """Persist a `SessionMessage` to the session transcript.

        The whole message (a Pydantic model) is serialized to JSON into the
        ``data`` column so we never lose fields during a round-trip.
        """
        self.db.execute(
            "INSERT INTO messages (id, session_id, type, data, time) VALUES (?, ?, ?, ?, ?)",
            (message.id, session_id, message.type, self.db._dumps(message), message.time),
        )
        # A new message counts as session activity.
        self.db.execute(
            "UPDATE sessions SET updated = ? WHERE id = ?",
            (self.db.now(), session_id),
        )
        self.db.commit()

    def messages(self, session_id: str) -> list[SessionMessage]:
        """Return the full transcript of a session, oldest first."""
        rows = self.db.execute(
            "SELECT data FROM messages WHERE session_id = ? ORDER BY time ASC",
            (session_id,),
        ).fetchall()
        return [self._decode_message(self.db._loads(row["data"])) for row in rows]

    # -- claim / release (crash recovery) --------------------------------

    def claim(self, session_id: str) -> bool:
        """Attempt to take ownership of a session.

        Succeeds (returns ``True``) when the session is either currently
        unclaimed or its existing claim is stale. On success we write a fresh
        ``time_suspended`` and bump ``resume_attempts``. Fails (returns
        ``False``) when another live runner holds the claim - the caller should
        back off and not process the session.

        The read-then-write is not fully atomic across processes, but within a
        single SQLite connection it is serialized by the database lock, which
        matches the original's intent for a single-node setup.
        """
        row = self.db.execute(
            "SELECT time_suspended FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            # No such session: nothing to claim.
            return False

        suspended = row["time_suspended"]
        if suspended is not None:
            age_ms = (self.db.now() - suspended) * 1000.0
            if age_ms < self.stale_claim_ms:
                # A recent claim exists and has not gone stale -> someone else
                # is actively processing this session.
                return False

        # Either unclaimed or stale: take ownership.
        self.db.execute(
            """
            UPDATE sessions
            SET time_suspended = ?, resume_attempts = resume_attempts + 1
            WHERE id = ?
            """,
            (self.db.now(), session_id),
        )
        self.db.commit()
        return True

    def release(self, session_id: str) -> None:
        """Clear the claim, marking the session as no longer being processed."""
        self.db.execute(
            "UPDATE sessions SET time_suspended = NULL WHERE id = ?", (session_id,)
        )
        self.db.commit()

    def is_claimed(self, session_id: str) -> bool:
        """Return True if the session currently has a non-stale claim."""
        row = self.db.execute(
            "SELECT time_suspended FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None or row["time_suspended"] is None:
            return False
        return (self.db.now() - row["time_suspended"]) * 1000.0 < self.stale_claim_ms

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _row_to_record(row: Any) -> SessionRecord:
        """Convert a session row into a `SessionRecord`."""
        return SessionRecord(
            id=row["id"],
            title=row["title"],
            agent=row["agent"],
            model=SessionStore._decode_model(row["model"]),
            parent_id=row["parent_id"],
            permission=json.loads(row["permission"]) if row["permission"] else [],
            created=row["created"],
            updated=row["updated"],
        )

    @staticmethod
    def _row_to_summary(row: Any) -> SessionSummary:
        """Convert a session row into a lightweight `SessionSummary`."""
        return SessionSummary(
            id=row["id"],
            title=row["title"],
            agent=row["agent"],
            model=SessionStore._decode_model(row["model"]),
            created=row["created"],
            updated=row["updated"],
        )

    @staticmethod
    def _decode_model(raw: str | None) -> ModelRef | None:
        """Deserialize a JSON-encoded model column into a `ModelRef` (or None)."""
        if not raw:
            return None
        return ModelRef(**json.loads(raw))

    @staticmethod
    def _decode_message(data: dict[str, Any]) -> SessionMessage:
        """Deserialize a stored message dict into its concrete `SessionMessage`.

        `SessionMessage` is a discriminated union keyed on ``type``; a Pydantic
        ``TypeAdapter`` restores the right subclass (user/assistant/tool/system).
        """
        return _message_adapter.validate_python(data)


def system_prompt(session: SessionRecord | None = None) -> str:
    """Default system prompt for the agent.

    Describes the agent as an AI coding assistant that can inspect and modify
    files, run commands, and delegate subtasks to sub-agents. ``session`` is
    accepted for forward-compatibility (e.g. injecting the session title) but
    is not currently required.
    """
    _ = session  # reserved for future use (e.g. injecting session context)
    return (
        "You are a helpful AI coding assistant. You can use tools to inspect "
        "and modify files, search the codebase, and run shell commands. "
        "When a task requires several steps, work through it methodically and "
        "report what you did. Use the available tools whenever they would help "
        "you complete the user's request more reliably."
    )
