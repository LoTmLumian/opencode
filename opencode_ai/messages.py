from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "MessageRole",
    "ContentPart",
    "TextPart",
    "ToolCallPart",
    "ToolResultPart",
    "Message",
    "ToolDefinition",
    "ToolChoice",
]


MessageRole = Literal["system", "user", "assistant", "tool"]


class ContentPart(BaseModel):
    """Base class for a message content part (mirrors `LLM.ContentPart`)."""

    type: str


class TextPart(ContentPart):
    """Plain text content (mirrors `LLM.TextPart`)."""

    type: Literal["text"] = "text"
    text: str


class ToolCallPart(ContentPart):
    """An assistant request to call a tool (mirrors `LLM.ToolCallPart`)."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    # JSON string of the arguments, or a pre-parsed dict.
    arguments: str | dict[str, Any]

    @property
    def args(self) -> dict[str, Any]:
        """The parsed arguments dict."""
        if isinstance(self.arguments, dict):
            return self.arguments
        try:
            return json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return {}


class ToolResultPart(ContentPart):
    """The result of a tool call, fed back to the model (mirrors `LLM.ToolResultPart`)."""

    type: Literal["tool_result"] = "tool_result"
    id: str
    name: str | None = None
    result: str = ""
    is_error: bool = False


# Discriminated union so a ``content`` list restores the concrete part type
# (text / tool_call / tool_result) instead of silently degrading to the base.
ContentPartUnion = Annotated[
    TextPart | ToolCallPart | ToolResultPart,
    Field(discriminator="type"),
]


class Message(BaseModel):
    """A model-agnostic conversation message (mirrors `LLM.Message`).

    `content` is a list of `ContentPart`; convenience constructors accept
    plain strings and normalize them.
    """

    role: MessageRole
    content: list[ContentPartUnion] = Field(default_factory=list)
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    providerMetadata: dict[str, Any] = Field(default_factory=dict)
    native: dict[str, Any] = Field(default_factory=dict)

    # -- convenience constructors ----------------------------------------
    @classmethod
    def text(cls, value: str) -> TextPart:
        return TextPart(text=value)

    @classmethod
    def make(cls, input: "Message | dict[str, Any]") -> "Message":
        if isinstance(input, cls):
            return input
        return cls(**input)

    @classmethod
    def user(cls, content: str | ContentPart | list[ContentPart]) -> "Message":
        return cls(role="user", content=cls._content(content))

    @classmethod
    def assistant(cls, content: str | ContentPart | list[ContentPart]) -> "Message":
        return cls(role="assistant", content=cls._content(content))

    @classmethod
    def system(cls, content: str | TextPart | list[TextPart]) -> "Message":
        parts = [cls.text(content)] if isinstance(content, str) else (list(content) if isinstance(content, list) else [content])
        return cls(role="system", content=parts)

    @classmethod
    def tool(cls, result: ToolResultPart | dict[str, Any]) -> "Message":
        part = result if isinstance(result, ToolResultPart) else ToolResultPart(**result)
        return cls(role="tool", content=[part])

    @staticmethod
    def _content(input: str | ContentPart | list[ContentPart]) -> list[ContentPart]:
        if isinstance(input, str):
            return [TextPart(text=input)]
        if isinstance(input, list):
            return list(input)
        return [input]

    @classmethod
    def from_openai(cls, raw: dict[str, Any]) -> "Message":
        """Parse an OpenAI wire-format message into a `Message`."""
        role = raw["role"]
        content = raw.get("content", "")
        parts: list[ContentPart] = []
        if isinstance(content, str):
            if content:
                parts.append(TextPart(text=content))
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    parts.append(TextPart(text=block.get("text", "")))
                # image_url blocks are preserved as native data for now.
        if role == "assistant" and raw.get("tool_calls"):
            for call in raw["tool_calls"]:
                parts.append(
                    ToolCallPart(
                        id=call["id"],
                        name=call["function"]["name"],
                        arguments=call["function"].get("arguments", "{}"),
                    )
                )
        return cls(role=role, content=parts, providerMetadata={"openai": raw})

    @classmethod
    def from_anthropic(cls, raw: dict[str, Any]) -> "Message":
        """Parse an Anthropic wire-format message into a `Message`."""
        role = raw.get("role", "user")
        parts: list[ContentPart] = []
        content = raw.get("content", [])
        if isinstance(content, str):
            parts.append(TextPart(text=content))
        elif isinstance(content, list):
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    parts.append(TextPart(text=block.get("text", "")))
                elif btype == "tool_use":
                    parts.append(
                        ToolCallPart(
                            id=block["id"],
                            name=block["name"],
                            arguments=json.dumps(block.get("input", {})),
                        )
                    )
                elif btype == "tool_result":
                    result = block.get("content")
                    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                    parts.append(
                        ToolResultPart(id=block.get("tool_use_id", ""), result=text, is_error=bool(block.get("is_error")))
                    )
        return cls(role=role, content=parts, providerMetadata={"anthropic": raw})


class ToolDefinition(BaseModel):
    """A tool the model may call (mirrors `LLM.ToolDefinition`)."""

    type: Literal["tool"] = "tool"
    name: str
    description: str
    inputSchema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    outputSchema: dict[str, Any] | None = None

    @classmethod
    def make(cls, input: "ToolDefinition | dict[str, Any]") -> "ToolDefinition":
        if isinstance(input, cls):
            return input
        return cls(**input)


class ToolChoice(BaseModel):
    """Tool selection policy (mirrors `LLM.ToolChoice`)."""

    type: Literal["auto", "none", "required", "tool"] = "auto"
    name: str | None = None
    disableParallelToolUse: bool | None = None

    @classmethod
    def named(cls, name: str) -> "ToolChoice":
        return cls(type="tool", name=name)

    @classmethod
    def make(cls, input: "ToolChoice | str | dict[str, Any] | None") -> "ToolChoice | None":
        if input is None:
            return None
        if isinstance(input, cls):
            return input
        if isinstance(input, str):
            return cls(type=input if input in ("auto", "none", "required") else "tool", name=None if input in ("auto", "none", "required") else input)
        return cls(**input)
