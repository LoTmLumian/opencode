from __future__ import annotations

from typing import Any

from opencode_schema.provider import ProviderConfig

from ..client import LLMClient
from ..messages import Message, ToolDefinition

__all__ = ["ProtocolError", "build_client", "make_client"]


class ProtocolError(RuntimeError):
    """Raised when a provider violates its wire protocol."""

    def __init__(self, message: str, *, status: int | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


def build_client(config: ProviderConfig, http: Any = None) -> LLMClient:
    """Build the `LLMClient` implementation for a `ProviderConfig`.

    The OpenAI-compatible protocol covers both native OpenAI and any
    base URL exposing a `/chat/completions` endpoint. Anthropic uses the
    Messages API. This is the extension point for new protocols.
    """
    if config.protocol == "anthropic":
        from .anthropic import AnthropicClient

        return AnthropicClient(config, http=http)
    # Default to the OpenAI-compatible chat protocol.
    from .openai import OpenAIClient

    return OpenAIClient(config, http=http)


# Backwards-compatible alias.
make_client = build_client


def to_openai_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Serialize `ToolDefinition`s into OpenAI function-tool objects."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
        )
    return result


def to_anthropic_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Serialize `ToolDefinition`s into Anthropic tool objects."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        result.append({"name": tool.name, "description": tool.description, "input_schema": tool.inputSchema})
    return result


def message_to_openai(message: Message) -> dict[str, Any]:
    """Convert a `Message` into the OpenAI wire format.

    Assistant tool calls are split into a parallel `tool_calls` array and a
    text content; tool results map to the ``role: "tool"`` message form.
    """
    if message.role == "tool":
        part = message.content[0] if message.content else None
        return {
            "role": "tool",
            "tool_call_id": getattr(part, "id", ""),
            "content": getattr(part, "result", ""),
        }

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for part in message.content:
        if part.type == "text":
            text_parts.append(part.text)
        elif part.type == "tool_call":
            tool_calls.append(
                {
                    "id": part.id,
                    "type": "function",
                    "function": {"name": part.name, "arguments": _dump_args(part)},
                }
            )
    body: dict[str, Any] = {"role": message.role}
    if text_parts:
        body["content"] = "\n".join(text_parts)
    elif message.role == "assistant":
        # OpenAI requires assistant content to be present when tool_calls exist.
        body["content"] = "" if tool_calls else None
    if tool_calls:
        body["tool_calls"] = tool_calls
    if message.name:
        body["name"] = message.name
    return body


def _dump_args(part: Any) -> str:
    import json

    if isinstance(part.arguments, dict):
        return json.dumps(part.arguments)
    return part.arguments
