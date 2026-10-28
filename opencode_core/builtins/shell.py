from __future__ import annotations

"""Execute a shell command (mirrors `tool/plugin/shell.ts`)."""

import asyncio
import subprocess
from typing import Any

from pydantic import BaseModel, Field

from opencode_core.tool import ToolEditor
from opencode_schema.tool import ToolCallError, ToolContext, ToolResult


class ShellInput(BaseModel):
    """Parameters for the `shell` tool."""

    command: str = Field(..., description="The shell command to execute.")
    timeout: int | None = Field(
        None, description="Timeout in seconds. If the command exceeds it, it is killed."
    )


async def execute_shell(input: dict[str, Any], context: ToolContext) -> ToolResult:
    """Run `input["command"]` via the system shell and return its exit code + output.

    Uses `shell=True` (required on Windows to run commands like `dir`) and
    captures stdout and stderr together. A timeout kills the process and is
    reported as an error result.
    """
    req = ShellInput.model_validate(input)

    # Run the command in a threadpool so we do not block the event loop while
    # waiting for the subprocess.
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, _run_command, req.command, req.timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolCallError(f"Command timed out after {req.timeout}s: {req.command}") from exc
    except OSError as exc:
        raise ToolCallError(f"Failed to run command: {exc}") from exc

    output = result["stdout"] + result["stderr"]
    return ToolResult.text(
        output if output else "(no output)",
        exit_code=result["exit_code"],
        timed_out=result["timed_out"],
    )


def _run_command(command: str, timeout: int | None) -> dict[str, Any]:
    """Synchronous helper that spawns the command and captures its output."""
    # `shell=True` lets commands work on both Windows (cmd) and Unix (sh); the
    # actual shell is whatever `subprocess` picks for the platform.
    proc = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
        "timed_out": False,
    }


def register(editor: ToolEditor) -> None:
    """Register the `shell` tool with the editor."""
    editor.add(
        "shell",
        "Execute a shell command and return its exit code and output.",
        ShellInput,
        execute=execute_shell,
    )
