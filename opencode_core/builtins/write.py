from __future__ import annotations

"""Write content to a file (mirrors `tool/plugin/write.ts`)."""

import pathlib
from typing import Any

from pydantic import BaseModel, Field

from opencode_core.tool import ToolEditor
from opencode_schema.tool import ToolCallError, ToolContext, ToolResult


class WriteInput(BaseModel):
    """Parameters for the `write` tool."""

    path: str = Field(..., description="Path to the file to write.")
    content: str = Field(..., description="The full content to write to the file.")


async def execute_write(input: dict[str, Any], context: ToolContext) -> ToolResult:
    """Write `input["content"]` to `input["path"]`.

    Parent directories are created on demand. The returned result reports the
    number of bytes written so the model can confirm the write succeeded.
    """
    req = WriteInput.model_validate(input)
    path = pathlib.Path(req.path)

    try:
        # Ensure the parent directory exists before writing.
        path.parent.mkdir(parents=True, exist_ok=True)
        data = req.content.encode("utf-8")
        path.write_bytes(data)
    except OSError as exc:
        raise ToolCallError(f"Failed to write {req.path}: {exc}") from exc

    return ToolResult.text(
        f"Wrote {len(data)} bytes to {req.path}",
        path=str(path),
        bytes=len(data),
    )


def register(editor: ToolEditor) -> None:
    """Register the `write` tool with the editor."""
    editor.add(
        "write",
        "Write content to a file, creating parent directories as needed.",
        WriteInput,
        execute=execute_write,
    )
