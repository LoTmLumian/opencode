from __future__ import annotations

from typing import Any, Callable, Protocol

from pydantic import BaseModel, Field

__all__ = [
    "ToolInputSchema",
    "ToolDefinition",
    "ToolContext",
    "ToolExecutor",
    "ToolResult",
]


# ---------------------------------------------------------------------------
# Tool schema (mirrors `Tool.Info` in `packages/schema/src/tool.ts`)
# ---------------------------------------------------------------------------


class ToolInputSchema(BaseModel):
    """JSON Schema describing a tool's input parameters.

    The original uses Effect's `Schema`; here we use plain JSON Schema dicts
    generated from Pydantic models via `TypeAdapter.json_schema()`.
    """

    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    additionalProperties: bool | dict[str, Any] = False


class ToolDefinition(BaseModel):
    """Static metadata for a tool (name, description, input schema).

    This is the schema-level contract; runtime execution lives in
    `opencode_core.tool` via `ToolExecutor`.
    """

    name: str
    description: str
    input_schema: ToolInputSchema
    # Optional namespace for grouping / permission scoping.
    namespace: str | None = None
    # Permission action id (defaults to the tool name).
    permission: str | None = None
    # Whether this tool is only available in codemode.
    codemode: bool = False

    def to_openai_schema(self) -> dict[str, Any]:
        """Serialize to an OpenAI function-tool schema dict."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema.model_dump(exclude_none=True),
            },
        }


class ToolContext(BaseModel):
    """Per-execution context handed to a tool (mirrors `Tool.Context`)."""

    session_id: str
    agent: str
    message_id: str
    call_id: str


class ToolResult(BaseModel):
    """Result of executing a tool (mirrors `Tool.Result`)."""

    output: Any = None
    content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def text(cls, text: str, **metadata: Any) -> "ToolResult":
        return cls(content=text, metadata=metadata)

    def render(self) -> str:
        """Render the result as a single string to feed back to the LLM."""
        return self.content if self.content is not None else str(self.output)


class ToolExecutor(Protocol):
    """Runtime signature for a tool's execute function."""

    async def __call__(self, input: dict[str, Any], context: ToolContext) -> ToolResult: ...


class ToolCallError(Exception):
    """Raised when a tool execution fails; surfaces as a failed tool result."""

    def __init__(self, message: str, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metadata = metadata or {}
