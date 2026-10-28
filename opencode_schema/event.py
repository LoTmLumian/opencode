from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from .model import ModelRef

__all__ = [
    "Event",
    "EventMessage",
    "EventType",
    "MessageUpdated",
    "SessionUpdated",
    "Snapshot",
]


def _event_id() -> str:
    return "evt_" + uuid.uuid4().hex


# Event types emitted on the bus / SSE feed.
EventType = Literal[
    "message.created",
    "message.updated",
    "message.part.updated",
    "session.idle",
    "session.updated",
    "session.error",
    "permission.ask",
    "permission.resolve",
    "model.updated",
    "snapshot.created",
]


class Event(BaseModel):
    """A bus event (mirrors `packages/core/src/bus.ts` Event)."""

    id: str = Field(default_factory=_event_id)
    type: EventType
    session_id: str | None = None
    # Payload is kept as raw JSON to allow forward-compatible evolution.
    payload: dict[str, Any] = Field(default_factory=dict)
    # Provider metadata namespace echoed back verbatim.
    native: dict[str, Any] = Field(default_factory=dict)
    time: float = Field(default_factory=time.time)


class MessageUpdated(Event):
    """Fired when a message is created or updated."""

    type: Literal["message.updated"] = "message.updated"


class SessionUpdated(Event):
    """Fired when session metadata changes (title, agent, model)."""

    type: Literal["session.updated"] = "session.updated"


# Alias used by the SSE feed and client SDKs.
class EventMessage(BaseModel):
    """Wire format for a single event on the SSE stream."""

    event: EventType
    data: dict[str, Any]

    @classmethod
    def from_event(cls, event: Event) -> "EventMessage":
        return cls(event=event.type, data=event.model_dump())


class Snapshot(BaseModel):
    """A session snapshot for crash recovery (mirrors `Session.Snapshot`)."""

    id: str = Field(default_factory=lambda: "snap_" + uuid.uuid4().hex)
    session_id: str
    # Serialized session state at snapshot time.
    data: dict[str, Any] = Field(default_factory=dict)
    time: float = Field(default_factory=time.time)
