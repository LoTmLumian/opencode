from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter, model_validator

__all__ = [
    "ModelRef",
    "parse_model_ref",
    "MessageRole",
    "ContentBlock",
    "TextBlock",
    "ToolCallBlock",
    "ToolResultBlock",
    "ChatMessage",
    "TokenUsage",
    "FinishReason",
    "ToolContent",
    "TextContent",
    "FileContent",
    "ToolResult",
]


# ---------------------------------------------------------------------------
# Model reference
#
# Mirrors `packages/schema/src/model.ts` ``Model.Ref``: a model is addressed as
# ``"<providerID>/<modelID>"`` with an optional ``#variant`` suffix.
# ---------------------------------------------------------------------------


class ModelRef(BaseModel):
    """A fully-qualified model reference ``providerID/modelID`` (optionally ``#variant``).

    Examples:
        ``openai/gpt-4o``, ``anthropic/claude-sonnet-4``, ``openai/gpt-4o#default``
    """

    providerID: str
    id: str
    variant: str | None = None

    @classmethod
    def parse(cls, input: str) -> "ModelRef":
        """Parse a model reference string into a ``ModelRef``.

        Raises:
            ValueError: If the string is not a valid ``providerID/modelID[#variant]``.
        """
        provider_end = input.find("/")
        if provider_end <= 0:
            raise ValueError(f"Invalid model reference: {input}")
        provider_id = input[:provider_end]
        variant_start = input.find("#", provider_end + 1)
        model_id = input[provider_end + 1 : variant_start if variant_start != -1 else None]
        variant = input[variant_start + 1 :] if variant_start != -1 else None
        if not model_id or "#" in provider_id or (variant is not None and (not variant or "#" in variant)):
            raise ValueError(f"Invalid model reference: {input}")
        return cls(providerID=provider_id, id=model_id, variant=variant)

    @model_validator(mode="before")
    @classmethod
    def _coerce_string(cls, value: object) -> object:
        # Allow constructing directly from a "provider/model" string.
        if isinstance(value, str):
            return cls.parse(value).model_dump()
        return value

    def __str__(self) -> str:
        base = f"{self.providerID}/{self.id}"
        return f"{base}#{self.variant}" if self.variant else base


# Convenience alias used across the codebase.
ModelRefAdapter = TypeAdapter(ModelRef)


# ---------------------------------------------------------------------------
# LLM chat message (provider-independent, maps onto the Route body)
# ---------------------------------------------------------------------------

MessageRole = Literal["user", "assistant", "system"]


class ContentBlock(BaseModel):
    """Base class for a single content block inside a message."""

    type: str


class TextBlock(ContentBlock):
    """Plain text content."""

    type: Literal["text"] = "text"
    text: str


class ToolCallBlock(ContentBlock):
    """An assistant request to invoke a tool."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, object] = Field(default_factory=dict)


class ToolResultBlock(ContentBlock):
    """The result of a previously requested tool call, fed back to the model."""

    type: Literal["tool_result"] = "tool_result"
    id: str
    content: str = ""
    is_error: bool = False


ContentBlockAdapter = TypeAdapter(ContentBlock)


class ChatMessage(BaseModel):
    """A single message in the model conversation window.

    ``role`` + ``content`` follows the OpenAI chat convention; block-level
    content is a superset used when tools are present.
    """

    role: MessageRole
    content: str | list[ContentBlock] = ""
    name: str | None = None


# ---------------------------------------------------------------------------
# Model response metadata
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    """Token accounting for a single model call (mirrors ``TokenUsage.Info``)."""

    input: int = 0
    output: int = 0
    total: int = 0


FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error", "unknown"]


# ---------------------------------------------------------------------------
# Tool content + result (mirrors ``packages/schema/src/tool.ts``)
# ---------------------------------------------------------------------------


class ToolContent(BaseModel):
    """Base class for tool result content blocks (``Tool.Content`` union)."""

    type: str


class TextContent(ToolContent):
    """A text chunk returned from a tool execution."""

    type: Literal["text"] = "text"
    text: str


class FileContent(ToolContent):
    """A file reference returned from a tool execution (e.g. an image path)."""

    type: Literal["file"] = "file"
    uri: str
    mime: str
    name: str | None = None


class ToolResult(BaseModel):
    """The structured result of a tool execution (mirrors ``Tool.Result``).

    ``output`` is the typed, machine-readable value; ``content`` is the
    human/model-facing rendering passed back to the LLM.
    """

    output: object | None = None
    content: str | list[ToolContent] | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @classmethod
    def text(cls, text: str, **metadata: object) -> "ToolResult":
        """Convenience constructor for a plain-text result."""
        return cls(content=text, metadata=metadata)

    def render(self) -> str:
        """Render the result as a single string for the LLM."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return "\n".join(
                block.text if isinstance(block, TextContent) else f"file: {block.uri}" for block in self.content
            )
        if self.content is None and self.output is not None:
            return str(self.output)
        return ""
