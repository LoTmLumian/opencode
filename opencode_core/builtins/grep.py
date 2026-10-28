from __future__ import annotations

"""Regular-expression search across files (mirrors `tool/plugin/grep.ts`)."""

import pathlib
import re
from typing import Any

from pydantic import BaseModel, Field

from opencode_core.tool import ToolEditor
from opencode_schema.tool import ToolCallError, ToolContext, ToolResult


class GrepInput(BaseModel):
    """Parameters for the `grep` tool."""

    pattern: str = Field(..., description="Regular expression to search for.")
    path: str | None = Field(
        None, description="Base directory to search. Defaults to the current working directory."
    )
    glob: str | None = Field(
        None, description="Optional glob to restrict which files are searched (e.g. '*.py')."
    )


async def execute_grep(input: dict[str, Any], context: ToolContext) -> ToolResult:
    """Search files under `path` for `pattern` and return `file:line:content` hits."""
    req = GrepInput.model_validate(input)
    base = pathlib.Path(req.path) if req.path else pathlib.Path.cwd()

    if not base.exists():
        raise ToolCallError(f"Path not found: {base}")

    try:
        regex = re.compile(req.pattern)
    except re.error as exc:
        raise ToolCallError(f"Invalid regular expression: {exc}") from exc

    # Collect candidate files: either the base is a single file, or we walk the
    # tree, optionally filtering by `glob`.
    files: list[pathlib.Path]
    if base.is_file():
        files = [base]
    else:
        files = [p for p in base.rglob("*") if p.is_file()]
        if req.glob:
            files = [p for p in files if p.match(req.glob)]

    hits: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue  # skip unreadable files rather than aborting the search
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                hits.append(f"{path}:{lineno}:{line}")

    return ToolResult.text(
        "\n".join(hits) if hits else "No matches found",
        matches=hits,
        count=len(hits),
    )


def register(editor: ToolEditor) -> None:
    """Register the `grep` tool with the editor."""
    editor.add(
        "grep",
        "Search file contents for a regular expression and return file:line:content matches.",
        GrepInput,
        execute=execute_grep,
    )
