from __future__ import annotations

"""The AI layer: model-agnostic request/response types, streaming clients,
and the multi-step agent loop (a Python port of the opencode ``packages/ai``
TypeScript package).
"""

from .client import (
    LLMClient,
    LLMEvent,
    LLMRequest,
    LLMResponse,
    MessageStart,
    MessageStop,
    ProviderError,
    RawEvent,
    StepFinish,
    StepStart,
    TextDelta,
    ToolCallDelta,
    is_provider_error,
)
from .llm import ExecutableTool, LoopOptions, generate, loop, stream, to_definitions
from .messages import Message, TextPart, ToolCallPart, ToolChoice, ToolDefinition, ToolResultPart
from .providers.base import build_client

__all__ = [
    "LLMRequest",
    "LLMResponse",
    "LLMClient",
    "LLMEvent",
    "StepStart",
    "StepFinish",
    "MessageStart",
    "TextDelta",
    "ToolCallDelta",
    "MessageStop",
    "ProviderError",
    "RawEvent",
    "is_provider_error",
    "Message",
    "ToolDefinition",
    "ToolChoice",
    "TextPart",
    "ToolCallPart",
    "ToolResultPart",
    "generate",
    "stream",
    "loop",
    "LoopOptions",
    "ExecutableTool",
    "to_definitions",
    "build_client",
]
