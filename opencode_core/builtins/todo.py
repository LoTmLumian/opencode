from __future__ import annotations

"""In-memory task/todo list (mirrors `tool/plugin/todo.ts`).

A simple task tracker the agent can use to keep an ordered list of outstanding
work. State is held in a process-wide dict so it survives across tool calls
within a session, but it is intentionally not persisted to disk.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from opencode_core.tool import ToolEditor
from opencode_schema.tool import ToolCallError, ToolContext, ToolResult


class TodoItem(BaseModel):
    """A single tracked task."""

    id: str
    subject: str
    status: Literal["pending", "in_progress", "completed"] = "pending"
    owner: str | None = None


class TodoInput(BaseModel):
    """Parameters for the `todo` tool."""

    action: Literal["list", "add", "update", "delete"] = Field(
        ..., description="The operation to perform on the task list."
    )
    id: str | None = Field(None, description="Task id, required for update/delete.")
    subject: str | None = Field(None, description="Task subject, required for add.")
    status: Literal["pending", "in_progress", "completed"] | None = Field(
        None, description="New status, used with update."
    )
    tasks: list[TodoItem] | None = Field(
        None, description="Bulk list of tasks to replace the whole list (list/add support)."
    )


# Process-wide store keyed by session so different sessions keep separate lists.
_STORE: dict[str, list[TodoItem]] = {}


def _next_id(items: list[TodoItem]) -> str:
    return str(max((int(t.id) for t in items if t.id.isdigit()), default=0) + 1)


async def execute_todo(input: dict[str, Any], context: ToolContext) -> ToolResult:
    """Dispatch on `input["action"]` against the session's task list."""
    req = TodoInput.model_validate(input)
    items = _STORE.setdefault(context.session_id, [])

    action = req.action
    if action == "list":
        return ToolResult.text(
            _render(items) if items else "No tasks",
            tasks=[t.model_dump() for t in items],
            count=len(items),
        )

    if action == "add":
        new_items = req.tasks
        if not new_items:
            if req.subject is None:
                raise ToolCallError("add requires a `subject` or `tasks`")
            new_items = [TodoItem(id=_next_id(items), subject=req.subject, owner=context.agent)]
        for item in new_items:
            if not item.id:
                item.id = _next_id(items)
            items.append(item)
        return ToolResult.text(f"Added {len(new_items)} task(s)", count=len(items))

    if action == "update":
        if req.id is None:
            raise ToolCallError("update requires an `id`")
        for item in items:
            if item.id == req.id:
                if req.subject is not None:
                    item.subject = req.subject
                if req.status is not None:
                    item.status = req.status
                return ToolResult.text(f"Updated task {req.id}", task=item.model_dump())
        raise ToolCallError(f"Task not found: {req.id}")

    if action == "delete":
        if req.id is None:
            raise ToolCallError("delete requires an `id`")
        for i, item in enumerate(items):
            if item.id == req.id:
                removed = items.pop(i)
                return ToolResult.text(f"Deleted task {req.id}", task=removed.model_dump())
        raise ToolCallError(f"Task not found: {req.id}")

    raise ToolCallError(f"Unknown action: {action}")


def _render(items: list[TodoItem]) -> str:
    """Render the task list as readable text."""
    lines = []
    for item in items:
        lines.append(f"[{item.status}] #{item.id} {item.subject}")
    return "\n".join(lines)


def register(editor: ToolEditor) -> None:
    """Register the `todo` tool with the editor."""
    editor.add(
        "todo",
        "Manage an in-memory task list: list, add, update, or delete tasks.",
        TodoInput,
        execute=execute_todo,
    )
