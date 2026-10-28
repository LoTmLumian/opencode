from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from opencode_schema.provider import ProviderConfig

from ..client import LLMClient, LLMEvent, LLMRequest, ProviderError, TextDelta, ToolCallDelta
from .base import ProtocolError, message_to_openai, to_openai_tools

__all__ = ["OpenAIClient"]


class OpenAIClient(LLMClient):
    """OpenAI-compatible chat client (`/chat/completions`, SSE stream).

    Covers native OpenAI and any provider exposing the same endpoint
    (DeepSeek, Qwen, OpenRouter, local vLLM/Ollama, ...). The route id is
    ``openai`` or ``openai-compatible`` — the wire format is identical.
    """

    def __init__(self, config: ProviderConfig, http: httpx.AsyncClient | None = None) -> None:
        self.config = config
        base = (config.base_url or "https://api.openai.com/v1").rstrip("/")
        self._base = base
        headers: dict[str, str] = {"Content-Type": "application/json"}
        key = config.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        headers.update(config.options.get("headers") or {})
        self._http = http or httpx.AsyncClient(base_url=base, headers=headers, timeout=httpx.Timeout(120.0))

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        body = self._build_body(request)
        url = self.config.options.get("chat_path", "/chat/completions")
        try:
            async with self._http.stream("POST", url, json=body) as resp:
                if resp.status_code >= 400:
                    text = await resp.aread()
                    raise ProtocolError(
                        f"OpenAI request failed ({resp.status_code}): {text[:500]!r}",
                        status=resp.status_code,
                        body=text[:500],
                    )
                async for line in resp.aiter_lines():
                    frame = self._parse_sse_line(line)
                    if frame is None:
                        continue
                    event = self._parse_frame(frame)
                    if event is not None:
                        yield event
        except httpx.HTTPError as exc:
            yield ProviderError(message=f"OpenAI transport error: {exc}")

    # -- internals --------------------------------------------------------

    def _build_body(self, request: LLMRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.tools:
            body["tools"] = to_openai_tools(request.tools)
            body["tool_choice"] = self._tool_choice(request)
        if request.generation:
            body.update(request.generation)
        # System prompt is folded into messages as the first element.
        if request.system:
            body["messages"].append({"role": "system", "content": "".join(str(p) for p in request.system)})
        body["messages"].extend(message_to_openai(m) for m in request.messages)
        # OpenAI rejects empty content arrays; keep only non-empty.
        body["messages"] = [m for m in body["messages"] if m.get("content") is not None or m.get("tool_calls")]
        return body

    def _tool_choice(self, request: LLMRequest) -> Any:
        choice = request.toolChoice
        if choice is None:
            return "auto"
        if choice.type == "none":
            return "none"
        if choice.type == "required":
            return "required"
        if choice.type == "tool" and choice.name:
            return {"type": "function", "function": {"name": choice.name}}
        return "auto"

    def _parse_sse_line(self, line: str) -> dict[str, Any] | None:
        """Parse one SSE line into a JSON frame dict, or None to skip."""
        if not line or not line.startswith("data:"):
            return None
        data = line[len("data:"):].strip()
        if not data:
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None

    def _parse_frame(self, frame: dict[str, Any]) -> LLMEvent | None:
        """Map one OpenAI chat-completion chunk to an `LLMEvent`."""
        choices = frame.get("choices") or []
        if not choices:
            # A usage-only chunk (final frame with stream_options include_usage).
            return None
        delta = choices[0].get("delta") or {}
        if delta.get("content"):
            return TextDelta(text=delta["content"])
        if delta.get("tool_calls"):
            call = delta["tool_calls"][0]
            fn = call.get("function") or {}
            return ToolCallDelta(
                id=call.get("id", ""),
                name=fn.get("name", ""),
                arguments=fn.get("arguments", ""),
            )
        return None
