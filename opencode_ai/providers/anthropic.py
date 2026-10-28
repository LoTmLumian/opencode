from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from opencode_schema.provider import ProviderConfig

from ..client import LLMClient, LLMEvent, LLMRequest, ProviderError, TextDelta, ToolCallDelta
from .base import ProtocolError, to_anthropic_tools

__all__ = ["AnthropicClient"]


class AnthropicClient(LLMClient):
    """Anthropic Messages API client (``/v1/messages``, SSE stream).

    Uses the native Anthropic wire format: ``content`` blocks of type
    ``text`` / ``tool_use`` / ``tool_result``.
    """

    def __init__(self, config: ProviderConfig, http: httpx.AsyncClient | None = None) -> None:
        self.config = config
        base = (config.base_url or "https://api.anthropic.com").rstrip("/")
        self._base = base
        headers: dict[str, str] = {"Content-Type": "application/json"}
        key = config.api_key()
        if key:
            headers["x-api-key"] = key
        headers.update(config.options.get("headers") or {})
        self._http = http or httpx.AsyncClient(base_url=base, headers=headers, timeout=httpx.Timeout(120.0))

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        body = self._build_body(request)
        url = self.config.options.get("messages_path", "/v1/messages")
        try:
            async with self._http.stream("POST", url, json=body) as resp:
                if resp.status_code >= 400:
                    text = await resp.aread()
                    raise ProtocolError(
                        f"Anthropic request failed ({resp.status_code}): {text[:500]!r}",
                        status=resp.status_code,
                        body=text[:500],
                    )
                event_type = None
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                        continue
                    if line.startswith("data:") and line[5:].strip():
                        try:
                            frame = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            continue
                        event = self._parse_event(event_type, frame)
                        if event is not None:
                            yield event
        except httpx.HTTPError as exc:
            yield ProviderError(message=f"Anthropic transport error: {exc}")

    # -- internals --------------------------------------------------------

    def _build_body(self, request: LLMRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [],
            "stream": True,
            "max_tokens": request.generation.get("max_tokens", 4096),
        }
        if request.tools:
            body["tools"] = to_anthropic_tools(request.tools)
        # System prompt goes in the top-level "system" field.
        if request.system:
            body["system"] = "".join(str(p) for p in request.system)
        for message in request.messages:
            body["messages"].append(self._message_to_anthropic(message))
        return body

    def _message_to_anthropic(self, message: Any) -> dict[str, Any]:
        """Convert a `Message` into the Anthropic wire format."""
        if message.role == "tool":
            part = message.content[0] if message.content else None
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": getattr(part, "id", ""),
                        "content": getattr(part, "result", ""),
                    }
                ],
            }
        blocks: list[dict[str, Any]] = []
        for part in message.content:
            if part.type == "text":
                blocks.append({"type": "text", "text": part.text})
            elif part.type == "tool_call":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": part.id,
                        "name": part.name,
                        "input": part.args,
                    }
                )
        return {"role": message.role, "content": blocks}

    def _parse_event(self, event_type: str | None, frame: dict[str, Any]) -> LLMEvent | None:
        """Map an Anthropic SSE event to an `LLMEvent`."""
        if event_type == "content_block_delta":
            delta = frame.get("delta") or {}
            if delta.get("type") == "text_delta":
                return TextDelta(text=delta.get("text", ""))
            if delta.get("type") == "input_json_delta":
                return ToolCallDelta(
                    id="",
                    name="",
                    arguments=delta.get("partial_json", ""),
                )
        return None
