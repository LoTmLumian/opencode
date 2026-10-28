from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from .model import ModelRef

__all__ = [
    "SessionID",
    "MessageID",
    "ToolCallID",
    "SessionMessage",
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "ToolMessage",
    "ToolState",
    "SessionSummary",
]


def _new_id(prefix: str) -> str:
    """Generate a unique ID with the given prefix (mirrors opencode's `ascending()`)."""
    return f"{prefix}_{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# ID types (branded strings in the original; plain string aliases here)
# ---------------------------------------------------------------------------


def new_session_id() -> str:
    return _new_id("ses")


def new_message_id() -> str:
    return _new_id("msg")


def new_tool_call_id() -> str:
    return _new_id("call")


# ---------------------------------------------------------------------------
# Tool execution state (mirrors `SessionMessage.ToolState`)
# ---------------------------------------------------------------------------


class ToolState(BaseModel):
    """Lifecycle state of a tool call within an assistant message."""

    status: Literal["streaming", "running", "completed", "error"]
    # Input arguments snapshot (valid once running).
    input: dict[str, Any] = Field(default_factory=dict)
    # Rendered output (valid once completed).
    content: str | None = None
    # Error message (valid once errored).
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallPart(BaseModel):
    """A tool invocation inside an assistant message (mirrors `AssistantTool`)."""

    type: Literal["tool_call"] = "tool_call"
    id: str = Field(default_factory=new_tool_call_id)
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    state: ToolState | None = None


class TextPart(BaseModel):
    """A text chunk inside an assistant message (mirrors `AssistantText`)."""

    type: Literal["text"] = "text"
    text: str


# ---------------------------------------------------------------------------
# Session messages (mirrors `SessionMessage.Info` union)
# ---------------------------------------------------------------------------


class SessionMessage(BaseModel):
    """A persisted message in a session transcript.

    The original models this as a tagged union discriminated by `type`;
    Pydantic's discriminated unions provide the same behaviour.
    """

    id: str = Field(default_factory=new_message_id)
    type: str
    time: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserMessage(SessionMessage):
    """A message from the user (mirrors `SessionMessage.User`)."""

    type: Literal["user"] = "user"
    text: str
    files: list[str] = Field(default_factory=list)
    agents: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)


class AssistantMessage(SessionMessage):
    """A model response (mirrors `SessionMessage.Assistant`)."""

    type: Literal["assistant"] = "assistant"
    agent: str
    model: ModelRef
    content: list[TextPart | ToolCallPart] = Field(default_factory=list)
    finish_reason: str | None = None
    error: str | None = None
    # Raw provider-native fields echoed verbatim.
    native: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)


class SystemMessage(SessionMessage):
    """A system-generated message (mirrors `SessionMessage.System`)."""

    type: Literal["system"] = "system"
    text: str
    description: str | None = None


class ToolMessage(SessionMessage):
    """A tool execution record (mirrors `SessionMessage.Tool` in v1)."""

    type: Literal["tool"] = "tool"
    tool_call_id: str
    tool_name: str
    output: str
    is_error: bool = False


# Runtime convenience: a lightweight summary of a session for listing.
class SessionSummary(BaseModel):
    id: str
    title: str
    agent: str | None = None
    model: ModelRef | None = None
    created: float
    updated: float
