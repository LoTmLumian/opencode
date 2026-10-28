from __future__ import annotations

"""Tool registry and service (mirrors `packages/core/src/tool.ts`).

This module owns the *runtime* side of the tool system: a registry where
tools are registered, looked up, updated and removed (`ToolEditor`), and a
thin service wrapper that executes tool calls while never letting a tool
error crash the agent loop (`ToolService`).

The static metadata contract (name / description / input schema) lives in
`opencode_schema.tool.ToolDefinition`; this module bridges from Pydantic
input models to those definitions and from there to the LLM layer's own
`opencode_ai.messages.ToolDefinition`.
"""

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, TypeAdapter, create_model

from opencode_schema.tool import (
    ToolCallError,
    ToolContext,
    ToolDefinition,
    ToolExecutor,
    ToolInputSchema,
    ToolResult,
)

__all__ = [
    "RegisteredTool",
    "ToolRegistrationError",
    "ToolEditor",
    "ToolService",
]


# ---------------------------------------------------------------------------
# Registered tool
# ---------------------------------------------------------------------------


@dataclass
class RegisteredTool:
    """A single registered tool: its static schema plus its executor.

    Attributes:
        definition: The static `ToolDefinition` (name / description / input
            schema / permission / codemode) handed to the model.
        execute: The async callable `(input, context) -> ToolResult`.
        id: Stable identifier for the tool, defaults to its name.
    """

    definition: ToolDefinition
    execute: ToolExecutor
    id: str


class ToolRegistrationError(Exception):
    """Raised when a tool cannot be registered (bad input model, dup name...)."""


# ---------------------------------------------------------------------------
# Tool editor (registry)
# ---------------------------------------------------------------------------


class ToolEditor:
    """Tool registry (mirrors `Tool.Editor`).

    Holds the mapping name -> `RegisteredTool` and offers the standard CRUD
    surface (`add` / `list` / `get` / `update` / `remove`). When a tool is
    added its input model is normalized into a `ToolDefinition` so consumers
    (LLM layer, server) only ever deal with schema objects.
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._namespaces: dict[str, Any] = {}

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _build_input_schema(input_model: Any) -> ToolInputSchema:
        """Convert an input model into a `ToolInputSchema`.

        `input_model` may be:
          * a Pydantic `BaseModel` subclass  -> derive JSON schema via TypeAdapter
          * a plain `dict` (raw JSON Schema) -> wrap as-is
          * `None`                           -> an empty object schema

        Raises `ToolRegistrationError` for anything else so registration fails
        loudly and early rather than at execution time.
        """
        if input_model is None:
            return ToolInputSchema()
        if isinstance(input_model, type) and issubclass(input_model, BaseModel):
            schema = TypeAdapter(input_model).json_schema()
        elif isinstance(input_model, dict):
            schema = input_model
        else:
            raise ToolRegistrationError(
                f"input_model must be a pydantic BaseModel, a JSON-schema dict, "
                f"or None; got {type(input_model).__name__}"
            )
        return ToolInputSchema(
            type=schema.get("type", "object"),
            properties=schema.get("properties", {}),
            required=list(schema.get("required", [])),
            additionalProperties=schema.get("additionalProperties", False),
        )

    # -- CRUD -------------------------------------------------------------

    def add(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel] | dict[str, Any] | None = None,
        *,
        execute: ToolExecutor | None = None,
        permission: str | None = None,
        codemode: bool = False,
    ) -> RegisteredTool:
        """Register a new tool.

        Args:
            name: Unique tool name (used by the model to invoke it).
            description: Human/LLM-facing description of what the tool does.
            input_model: Pydantic model, raw JSON-schema dict, or None.
            execute: Async `(input: dict, context: ToolContext) -> ToolResult`.
                Required for a usable tool; tools without an executor are
                registered but will return an error when executed.
            permission: Optional permission action id (defaults to `name`).
            codemode: Whether the tool is only available in codemode.

        Returns the newly registered `RegisteredTool`.
        """
        if name in self._tools:
            raise ToolRegistrationError(f"Tool already registered: {name}")
        definition = ToolDefinition(
            name=name,
            description=description,
            input_schema=self._build_input_schema(input_model),
            permission=permission or name,
            codemode=codemode,
        )
        registered = RegisteredTool(definition=definition, execute=execute or _noop_execute, id=name)
        self._tools[name] = registered
        return registered

    def list(self) -> list[RegisteredTool]:
        """Return all registered tools in insertion order."""
        return list(self._tools.values())

    def get(self, name: str) -> RegisteredTool | None:
        """Look up a tool by name; returns None if not registered."""
        return self._tools.get(name)

    def update(self, name: str, update_fn: Callable[[RegisteredTool], RegisteredTool]) -> RegisteredTool:
        """Apply `update_fn` to the registered tool and store the result.

        Useful for wrapping an executor with permission checks or logging.
        """
        current = self.get(name)
        if current is None:
            raise ToolRegistrationError(f"Unknown tool: {name}")
        updated = update_fn(current)
        self._tools[name] = updated
        return updated

    def remove(self, name: str) -> None:
        """Remove a tool by name (no-op if it is not registered)."""
        self._tools.pop(name, None)

    # -- LLM bridge -------------------------------------------------------

    def to_llm_definitions(self) -> list[Any]:
        """Return the tools in the AI layer's `ToolDefinition` format.

        This is the format consumed by `opencode_ai` (the model-facing layer),
        distinct from the schema-layer `ToolDefinition`. We import lazily to
        avoid a hard import cycle between core and ai at module load time.
        """
        from opencode_ai.messages import ToolDefinition as LLMToolDefinition

        out: list[Any] = []
        for tool in self._tools.values():
            out.append(
                LLMToolDefinition(
                    name=tool.definition.name,
                    description=tool.definition.description,
                    inputSchema=tool.definition.input_schema.model_dump(exclude_none=True),
                )
            )
        return out


async def _noop_execute(input: dict[str, Any], context: ToolContext) -> ToolResult:
    """Fallback executor for tools registered without a real implementation."""
    return ToolResult(content="Tool has no executor configured.", metadata={"error": True})


# ---------------------------------------------------------------------------
# Tool service
# ---------------------------------------------------------------------------


class ToolService:
    """Tool service (mirrors `Tool.Service`).

    A thin wrapper over a `ToolEditor`: it exposes the list of registered
    tools and executes tool calls by name. Execution is deliberately
    fail-safe — any error (unknown tool, `ToolCallError`, or arbitrary
    exception) is converted into an error `ToolResult` so the calling agent
    loop never crashes on a bad tool call.
    """

    def __init__(self, editor: ToolEditor | None = None) -> None:
        self.editor = editor or ToolEditor()

    def list(self) -> list[RegisteredTool]:
        return self.editor.list()

    async def execute(self, name: str, input: dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute a tool call by name.

        Never raises: all failure modes are normalized into a `ToolResult`
        flagged with `metadata["error"] = True`.
        """
        tool = self.editor.get(name)
        if tool is None:
            return ToolResult(content=f"Unknown tool: {name}", metadata={"error": True})
        try:
            return await tool.execute(input, context)
        except ToolCallError as exc:
            return ToolResult(content=str(exc), metadata={"error": True, **exc.metadata})
        except Exception as exc:  # noqa: BLE001 - tool errors must not crash the loop
            return ToolResult(content=f"Tool {name} failed: {exc}", metadata={"error": True})
