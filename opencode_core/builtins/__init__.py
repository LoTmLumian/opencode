from __future__ import annotations

"""Built-in tools (mirrors `packages/core/src/tool/plugin/*`).

Each module in this package defines one tool and exposes a `register(editor)`
function. `register_all` wires every built-in tool into a `ToolEditor`.
"""

from opencode_core.tool import ToolEditor

__all__ = ["register_all"]


def register_all(editor: ToolEditor) -> None:
    """Register all built-in tools onto `editor`."""
    from . import edit, glob, grep, read, shell, subagent, todo, webfetch, write

    for mod in (read, write, edit, glob, grep, shell, todo, webfetch, subagent):
        mod.register(editor)
