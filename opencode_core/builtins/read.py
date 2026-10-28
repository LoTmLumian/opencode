from __future__ import annotations

"""Read a file from the filesystem (mirrors `tool/plugin/read.ts`)."""

import pathlib
from typing import Any

from pydantic import BaseModel, Field

from opencode_core.tool import ToolEditor
from opencode_schema.tool import ToolCallError, ToolContext, ToolResult


class ReadInput(BaseModel):
    """Parameters for the `read` tool."""

    path: str = Field(..., description="Path to the file to read.")
    offset: int | None = Field(
        None, description="Number of lines to skip from the start (0-based)."
    )
    limit: int | None = Field(
        None, description="Maximum number of lines to return. Omit to read the whole file."
    )


async def execute_read(input: dict[str, Any], context: ToolContext) -> ToolResult:
    """Read `input["path"]`, honoring optional `offset` / `limit` line ranges.

    The file is decoded as UTF-8 (with a graceful fallback to the filesystem
    encoding on decoding errors). Errors are surfaced as `ToolCallError` so the
    `ToolService` can render them as failed tool results rather than crashing.
    """
    req = ReadInput.model_validate(input)
    path = pathlib.Path(req.path)

    if not path.exists():
        raise ToolCallError(f"File not found: {req.path}")
    if not path.is_file():
        raise ToolCallError(f"Not a file: {req.path}")

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ToolCallError(f"Failed to read {req.path}: {exc}") from exc

    # Decode text; fall back to the local filesystem encoding if UTF-8 fails so
    # binary-ish or legacy-encoded files still yield something useful.
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode(errors="replace")

    lines = text.splitlines(keepends=True)

    start = req.offset or 0
    if req.limit is not None:
        selected = lines[start : start + req.limit]
    else:
        selected = lines[start:]

    content = "".join(selected)
    return ToolResult.text(
        content,
        path=str(path),
        lines=len(selected),
        total_lines=len(lines),
    )


def register(editor: ToolEditor) -> None:
    """Register the `read` tool with the editor."""
    editor.add(
        "read",
        "Read a file from the filesystem. Handles text files; use glob for directory listing.",
        ReadInput,
        execute=execute_read,
    )
