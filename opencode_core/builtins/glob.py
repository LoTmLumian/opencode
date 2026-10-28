from __future__ import annotations

"""File pattern matching (mirrors `tool/plugin/glob.ts`)."""

import pathlib
from typing import Any

from pydantic import BaseModel, Field

from opencode_core.tool import ToolEditor
from opencode_schema.tool import ToolCallError, ToolContext, ToolResult


class GlobInput(BaseModel):
    """Parameters for the `glob` tool."""

    pattern: str = Field(..., description="Glob pattern to match against file paths.")
    path: str | None = Field(
        None, description="Base directory to search from. Defaults to the current working directory."
    )


async def execute_glob(input: dict[str, Any], context: ToolContext) -> ToolResult:
    """Return the list of files matching `pattern` under `path` (default cwd)."""
    req = GlobInput.model_validate(input)
    base = pathlib.Path(req.path) if req.path else pathlib.Path.cwd()

    if not base.exists():
        raise ToolCallError(f"Directory not found: {base}")
    if not base.is_dir():
        raise ToolCallError(f"Not a directory: {base}")

    # `rglob` treats the pattern relative to the base; we use it to walk the
    # tree and filter by match against the pattern.
    try:
        matches = sorted(str(p) for p in base.rglob(req.pattern) if p.is_file())
    except OSError as exc:
        raise ToolCallError(f"Glob failed: {exc}") from exc

    return ToolResult.text(
        "\n".join(matches) if matches else "No files matched",
        files=matches,
        count=len(matches),
    )


def register(editor: ToolEditor) -> None:
    """Register the `glob` tool with the editor."""
    editor.add(
        "glob",
        "List files matching a glob pattern, recursively from an optional base directory.",
        GlobInput,
        execute=execute_glob,
    )
