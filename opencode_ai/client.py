from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel, Field

from .messages import Message, ToolChoice, ToolDefinition

__all__ = [
    "LLMEvent",
    "LLMResponse",
    "LLMRequest",
    "LLMClient",
    "StepStart",
    "StepFinish",
    "MessageStart",
    "TextDelta",
    "ToolCallDelta",
    "MessageStop",
    "ProviderError",
]


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class LLMRequest(BaseModel):
    """The canonical request passed to every route (mirrors `LLM.LLMRequest`)."""

    id: str | None = None
    model: str = ""
    provider: str = ""
    system: list[Any] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    tools: list[ToolDefinition] = Field(default_factory=list)
    toolChoice: ToolChoice | None = None
    generation: dict[str, Any] = Field(default_factory=dict)
    providerOptions: dict[str, Any] = Field(default_factory=dict)
    http: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def update(cls, request: "LLMRequest", patch: dict[str, Any]) -> "LLMRequest":
        """Return a copy of `request` with `patch` merged in."""
        if not patch:
            return request
        return cls(**{**request.model_dump(), **patch})


# ---------------------------------------------------------------------------
# Stream events (mirrors `LLM.LLMEvent` union)
# ---------------------------------------------------------------------------


class LLMEvent(BaseModel):
    """Base class for events emitted during a streaming response."""

    type: str
    step: int | None = None


class StepStart(LLMEvent):
    """A new loop step begins (mirrors `LLMEvent.StepStart`)."""

    type: Literal["step-start"] = "step-start"  # type: ignore[assignment]


class StepFinish(LLMEvent):
    """A loop step ends (mirrors `LLMEvent.StepFinish`)."""

    type: Literal["step-finish"] = "step-finish"  # type: ignore[assignment]


class MessageStart(LLMEvent):
    """A message begins streaming (mirrors `LLMEvent.MessageStart`)."""

    type: Literal["message-start"] = "message-start"  # type: ignore[assignment]


class TextDelta(LLMEvent):
    """A text chunk arrives (mirrors `LLMEvent.TextDelta`)."""

    type: Literal["text-delta"] = "text-delta"  # type: ignore[assignment]
    text: str


class ToolCallDelta(LLMEvent):
    """A tool-call chunk arrives (mirrors `LLMEvent.ToolCallDelta`)."""

    type: Literal["tool-call-delta"] = "tool-call-delta"  # type: ignore[assignment]
    id: str
    name: str
    arguments: str


class MessageStop(LLMEvent):
    """A message finishes streaming (mirrors `LLMEvent.MessageStop`)."""

    type: Literal["message-stop"] = "message-stop"  # type: ignore[assignment]


class ProviderError(LLMEvent):
    """The provider reported an error (mirrors `LLMEvent.ProviderError`)."""

    type: Literal["provider-error"] = "provider-error"  # type: ignore[assignment]
    message: str


class RawEvent(LLMEvent):
    """A raw provider frame, for debugging (mirrors `LLMEvent.Raw`)."""

    type: Literal["raw"] = "raw"  # type: ignore[assignment]
    data: Any = None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class LLMResponse(BaseModel):
    """A completed model response (mirrors `LLM.LLMResponse`)."""

    id: str | None = None
    message: Message
    toolCalls: list[Any] = Field(default_factory=list)
    finishReason: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    providerMetadata: dict[str, Any] = Field(default_factory=dict)
    raw: list[Any] = Field(default_factory=list)

    @classmethod
    def from_events(cls, events: list[LLMEvent]) -> "LLMResponse | None":
        """Assemble a `LLMResponse` from a stream of `LLMEvent`s."""
        text_parts: list[str] = []
        calls: list[dict[str, Any]] = []
        current_call: dict[str, Any] | None = None
        finish: str | None = None
        usage: dict[str, Any] = {}
        raw: list[Any] = []
        for event in events:
            if isinstance(event, RawEvent):
                raw.append(event.data)
            elif isinstance(event, TextDelta):
                text_parts.append(event.text)
            elif isinstance(event, ToolCallDelta):
                if current_call is None or current_call.get("id") != event.id:
                    current_call = {"id": event.id, "name": event.name, "arguments": ""}
                    calls.append(current_call)
                current_call["arguments"] += event.arguments
            elif isinstance(event, ProviderError):
                # The caller inspects `events` for provider errors; mark finish.
                finish = "error"
        if not text_parts and not calls and finish is None:
            return None
        content: list[Any] = []
        if text_parts:
            content.append({"type": "text", "text": "".join(text_parts)})
        for call in calls:
            content.append({"type": "tool_call", **call})
        return cls(
            message=Message(role="assistant", content=content),
            toolCalls=calls,
            finishReason=finish,
            usage=usage,
            raw=raw,
        )


# ---------------------------------------------------------------------------
# Client interface
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """Provider-agnostic streaming client.

    A concrete implementation knows how to speak one wire protocol
    (OpenAI-compatible or Anthropic Messages) and is selected by the
    provider's `protocol` field. `stream` returns an async iterator of
    `LLMEvent`s; `generate` collects them into a single `LLMResponse`.
    """

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        """Yield model events for a request."""
        raise NotImplementedError

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Collect a streaming response into a single `LLMResponse`."""
        events: list[LLMEvent] = []
        async for event in self.stream(request):
            events.append(event)
        response = LLMResponse.from_events(events)
        if response is None:
            raise RuntimeError("Provider returned an empty response")
        return response


def is_provider_error(events: list[LLMEvent]) -> bool:
    """Return True if any event in the list is a provider error."""
    return any(isinstance(event, ProviderError) for event in events)
