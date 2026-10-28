from __future__ import annotations

"""Exact string replacement in a file (mirrors `tool/plugin/edit.ts`).

The `edit` tool replaces one exact occurrence of `old_string` with
`new_string`. To avoid ambiguity it requires `old_string` to appear exactly
once; if it is missing or appears multiple times the tool reports an error so
the model can refine its request.
"""

import pathlib
from typing import Any

from pydantic import BaseModel, Field

from opencode_core.tool import ToolEditor
from opencode_schema.tool import ToolCallError, ToolContext, ToolResult


class EditInput(BaseModel):
    """Parameters for the `edit` tool."""

    path: str = Field(..., description="Path to the file to edit.")
    old_string: str = Field(..., description="The exact text to search for and replace.")
    new_string: str = Field(..., description="The replacement text.")


async def execute_edit(input: dict[str, Any], context: ToolContext) -> ToolResult:
    """Replace the unique occurrence of `old_string` with `new_string`.

    Mirrors the original project's exact-match semantics: the old string must
    be present and occur exactly once, otherwise we raise `ToolCallError`
    rather than guessing at the intended edit.
    """
    req = EditInput.model_validate(input)
    path = pathlib.Path(req.path)

    if not path.exists():
        raise ToolCallError(f"File not found: {req.path}")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ToolCallError(f"Failed to read {req.path}: {exc}") from exc
    except UnicodeDecodeError:
        raise ToolCallError(f"File is not valid UTF-8: {req.path}") from None

    count = text.count(req.old_string)
    if count == 0:
        raise ToolCallError(f"old_string not found in {req.path}")
    if count > 1:
        raise ToolCallError(
            f"old_string is ambiguous: found {count} occurrences in {req.path}. "
            f"Provide a more specific old_string."
        )

    new_text = text.replace(req.old_string, req.new_string)
    try:
        path.write_text(new_text, encoding="utf-8")
    except OSError as exc:
        raise ToolCallError(f"Failed to write {req.path}: {exc}") from exc

    return ToolResult.text(
        f"Replaced 1 occurrence in {req.path}",
        path=str(path),
        replacements=1,
    )


def register(editor: ToolEditor) -> None:
    """Register the `edit` tool with the editor."""
    editor.add(
        "edit",
        "Replace an exact string in a file with a new string. The old string must match exactly once.",
        EditInput,
        execute=execute_edit,
    )
