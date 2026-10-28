from __future__ import annotations

"""Sub-agent tool (mirrors `tool/plugin/subagent.ts`).

A sub-agent is a nested agent that can be spawned to work on a focused task.
The real spawning logic lives in the session layer; this module provides the
`subagent` tool definition and a placeholder execution path. A runner
callback can be injected (via `set_runner`) so the session layer can wire in
the actual implementation without this module importing the session layer.
"""

from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from opencode_core.tool import ToolEditor
from opencode_schema.tool import ToolCallError, ToolContext, ToolResult


class SubagentInput(BaseModel):
    """Parameters for the `subagent` tool."""

    agent: str = Field(..., description="The agent type/name to spawn.")
    description: str = Field(..., description="Short description of the subtask.")
    prompt: str = Field(..., description="The instruction/prompt for the sub-agent.")
    model: str | None = Field(None, description="Optional model override for the sub-agent.")
    session_id: str | None = Field(None, description="Optional session id to associate with the sub-agent.")
    background: bool = Field(False, description="Whether to run the sub-agent in the background.")


# Injectable runner callback: (input, context) -> str (the sub-agent's result).
# Set by the session layer to provide the real implementation.
_Runner: Callable[[dict[str, Any], ToolContext], Awaitable[str]] | None = None


def set_runner(runner: Callable[[dict[str, Any], ToolContext], Awaitable[str]] | None) -> None:
    """Inject (or clear) the sub-agent runner used by `execute_subagent`."""
    global _Runner
    _Runner = runner


async def execute_subagent(input: dict[str, Any], context: ToolContext) -> ToolResult:
    """Spawn (or describe) a sub-agent.

    If a runner is injected, it is invoked and its text result returned.
    Otherwise a placeholder message is returned explaining that the actual
    sub-agent execution is handled by the session layer.
    """
    req = SubagentInput.model_validate(input)

    if _Runner is not None:
        try:
            result = await _Runner(input, context)
        except Exception as exc:  # noqa: BLE001 - surface as tool error
            raise ToolCallError(f"Sub-agent failed: {exc}") from exc
        return ToolResult.text(result, agent=req.agent, background=req.background)

    return ToolResult.text(
        f"Sub-agent requested (agent={req.agent}, background={req.background}). "
        f"Execution is handled by the session layer; no runner is configured here.",
        agent=req.agent,
        description=req.description,
        background=req.background,
        placeholder=True,
    )


def register(editor: ToolEditor) -> None:
    """Register the `subagent` tool with the editor."""
    editor.add(
        "subagent",
        "Spawn a sub-agent to work on a focused task and return its result.",
        SubagentInput,
        execute=execute_subagent,
    )
