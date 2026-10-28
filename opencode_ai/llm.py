from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from .client import (
    LLMClient,
    LLMEvent,
    LLMRequest,
    LLMResponse,
    StepFinish,
    StepStart,
    is_provider_error,
)
from .messages import Message, ToolCallPart, ToolDefinition, ToolResultPart
from .providers.base import build_client

__all__ = ["generate", "stream", "loop", "LoopOptions", "ExecutableTool", "to_definitions"]


# ---------------------------------------------------------------------------
# Executable tools
# ---------------------------------------------------------------------------


class ExecutableTool:
    """A runnable tool: a name plus a callable that turns an input dict into a result."""

    def __init__(self, name: str, execute: Callable[[dict[str, Any]], Awaitable[Any]]) -> None:
        self.name = name
        self.execute = execute


def to_definitions(tools: list[ExecutableTool]) -> list[ToolDefinition]:
    """Build `ToolDefinition`s from a list of `ExecutableTool`s.

    The tool description is taken from the underlying callable's docstring
    when available, otherwise left empty.
    """
    definitions: list[ToolDefinition] = []
    for tool in tools:
        doc = getattr(tool.execute, "__doc__", None)
        description = doc.strip() if isinstance(doc, str) and doc.strip() else ""
        definitions.append(ToolDefinition(name=tool.name, description=description))
    return definitions


# ---------------------------------------------------------------------------
# Single-shot helpers
# ---------------------------------------------------------------------------


def _client_for(request: LLMRequest) -> LLMClient:
    """Construct an `LLMClient` for the request's provider.

    The provider id selects the wire protocol (``anthropic`` vs the default
    OpenAI-compatible protocol) and the API key environment variable. Base
    URL is left to the client's default when not otherwise configured.
    """
    from opencode_schema.provider import ProviderConfig

    provider = (request.provider or "").lower()
    if provider == "anthropic":
        config = ProviderConfig(
            id="anthropic",
            protocol="anthropic",
            api_key_env="ANTHROPIC_API_KEY",
        )
    else:
        config = ProviderConfig(
            id=provider or "openai",
            protocol="openai",
            api_key_env="OPENAI_API_KEY",
        )
    return build_client(config)


async def generate(
    request: LLMRequest,
    client: LLMClient | None = None,
) -> LLMResponse:
    """Run a single request and return the assembled `LLMResponse`."""
    if client is None:
        client = _client_for(request)
    return await client.generate(request)


async def stream(
    request: LLMRequest,
    client: LLMClient | None = None,
) -> AsyncIterator[LLMEvent]:
    """Stream the events for a single request."""
    if client is None:
        client = _client_for(request)
    async for event in client.stream(request):
        yield event


# ---------------------------------------------------------------------------
# Multi-step agent loop
# ---------------------------------------------------------------------------


class LoopOptions:
    """Configuration knobs for the agent loop (currently a placeholder)."""

    def __init__(self, *, max_steps: int | None = None) -> None:
        self.max_steps = max_steps


async def _run_tool(tool_map: dict[str, ExecutableTool], call: dict[str, Any]) -> ToolResultPart:
    """Execute one tool call, wrapping the outcome in a `ToolResultPart`."""
    name = call.get("name", "")
    tool_id = call.get("id", "")
    arguments = call.get("arguments", "{}")
    if isinstance(arguments, str):
        try:
            args: dict[str, Any] = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            args = {}
    else:
        args = arguments or {}

    tool = tool_map.get(name)
    if tool is None:
        return ToolResultPart(id=tool_id, name=name, result=f"Unknown tool: {name}", is_error=True)
    try:
        result = await tool.execute(args)
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        return ToolResultPart(id=tool_id, name=name, result=text, is_error=False)
    except Exception as exc:  # noqa: BLE001 - surface any tool failure to the model
        return ToolResultPart(id=tool_id, name=name, result=str(exc), is_error=True)


def _extract_tool_calls(response: LLMResponse) -> list[dict[str, Any]]:
    """Collect tool calls from a response, covering both `toolCalls` and message content."""
    calls: list[dict[str, Any]] = list(response.toolCalls or [])
    if not calls:
        for part in response.message.content:
            if isinstance(part, ToolCallPart):
                calls.append({"id": part.id, "name": part.name, "arguments": part.arguments})
    return calls


async def loop(
    request: LLMRequest,
    tools: list[ExecutableTool],
    client: LLMClient | None = None,
    *,
    until: Callable[[list[LLMResponse]], bool] | None = None,
) -> AsyncIterator[LLMEvent]:
    """Multi-step agent loop (mirrors `LLM.loop` in the original TS project).

    Each step streams the request, yields its events, then extracts any tool
    calls, executes them concurrently, appends the results as ``tool``
    messages, and recurses until no tool calls remain or ``until`` returns
    True.
    """
    if client is None:
        client = _client_for(request)
    tool_map = {tool.name: tool for tool in tools}
    history: list[LLMResponse] = []

    async def step(req: LLMRequest, index: int) -> AsyncIterator[LLMEvent]:
        nonlocal history
        events: list[LLMEvent] = []
        # Mark the beginning of a step and stream the client events.
        start = StepStart(step=index)
        events.append(start)
        yield start
        async for event in client.stream(req):
            if event.step is None:
                event.step = index
            events.append(event)
            yield event

        response = LLMResponse.from_events(events)
        if response is not None:
            history.append(response)
        yield StepFinish(step=index)

        # Termination conditions.
        if response is None:
            return
        if is_provider_error(events):
            return
        calls = _extract_tool_calls(response)
        if not calls:
            return
        if until is not None and until(history):
            return

        # Execute all tool calls concurrently.
        results = await asyncio.gather(*(_run_tool(tool_map, call) for call in calls))
        tool_messages = [Message.tool(part) for part in results]

        # Recurse with the tool results appended to the conversation.
        next_request = LLMRequest.update(req, {"messages": [*req.messages, *tool_messages]})
        async for event in step(next_request, index + 1):
            yield event

    async for event in step(request, 0):
        yield event
