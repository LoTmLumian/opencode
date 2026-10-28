from __future__ import annotations

"""opencode_schema — data contract layer.

Mirrors the original `packages/schema` package: pure data types shared by
core, ai, server, cli and sdk with no runtime dependencies between the
layers. Everything here is provider-agnostic and persistence-agnostic.
"""

from .agent import AgentDefinition, AgentMode, AgentPrompt
from .event import (
    Event,
    EventMessage,
    EventType,
    MessageUpdated,
    SessionUpdated,
    Snapshot,
)
from .model import (
    ChatMessage,
    ContentBlock,
    FileContent,
    FinishReason,
    MessageRole,
    ModelRef,
    TextBlock,
    TextContent,
    TokenUsage,
    ToolCallBlock,
    ToolContent,
    ToolResultBlock,
)
from .permission import (
    PermissionDecision,
    PermissionMode,
    PermissionRequest,
    PermissionResult,
    PermissionRule,
)
from .provider import AuthType, ProviderConfig, ProviderDefinition
from .session import (
    AssistantMessage,
    SessionMessage,
    SessionSummary,
    SystemMessage,
    TextPart,
    ToolCallPart,
    ToolMessage,
    ToolState,
    UserMessage,
    new_message_id,
    new_session_id,
    new_tool_call_id,
)
from .tool import (
    ToolCallError,
    ToolContext,
    ToolDefinition,
    ToolExecutor,
    ToolInputSchema,
    ToolResult,
)

__all__ = [
    "AgentDefinition",
    "AgentMode",
    "AgentPrompt",
    "AssistantMessage",
    "AuthType",
    "ChatMessage",
    "ContentBlock",
    "Event",
    "EventMessage",
    "EventType",
    "FileContent",
    "FinishReason",
    "MessageRole",
    "MessageUpdated",
    "ModelRef",
    "PermissionDecision",
    "PermissionMode",
    "PermissionRequest",
    "PermissionResult",
    "PermissionRule",
    "ProviderConfig",
    "ProviderDefinition",
    "SessionMessage",
    "SessionSummary",
    "SessionUpdated",
    "Snapshot",
    "SystemMessage",
    "TextBlock",
    "TextContent",
    "TextPart",
    "TokenUsage",
    "ToolCallBlock",
    "ToolCallPart",
    "ToolContext",
    "ToolDefinition",
    "ToolCallError",
    "ToolExecutor",
    "ToolInputSchema",
    "ToolMessage",
    "ToolResult",
    "ToolResultBlock",
    "ToolState",
    "UserMessage",
    "new_message_id",
    "new_session_id",
    "new_tool_call_id",
]
