"""Verify the code examples in docs/USAGE.md actually run.

Offline: injects a fake LLM client for the SDK/loop examples.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opencode_ai.client import MessageStop, StepFinish, StepStart, TextDelta
from opencode_core.permission import PermissionBlockedError, PermissionService
from opencode_core.tool import ToolEditor, ToolService
from opencode_schema.permission import PermissionRequest, PermissionRule
from opencode_schema.tool import ToolContext, ToolResult


class FakeClient:
    async def stream(self, request):
        yield StepStart()
        yield TextDelta(text="ok")
        yield StepFinish()
        yield MessageStop()


async def verify_permission() -> None:
    svc = PermissionService()
    svc.add_rule(PermissionRule(action="write", resource="secrets/*", mode="deny"))
    try:
        await svc.assert_not_denied(PermissionRequest(action="write", resource="secrets/key.txt"))
        raise AssertionError("should have been blocked")
    except PermissionBlockedError:
        print("permission: blocked OK")

    async def handler(req):
        return "allow"

    svc2 = PermissionService(ask_handler=handler)
    result = await svc2.assert_(PermissionRequest(action="webfetch", resource="https://example.com"))
    assert result.decision == "allow"
    print("permission: ask+allow OK")


async def verify_custom_tool() -> None:
    from pydantic import BaseModel, Field

    class GreetInput(BaseModel):
        name: str = Field(..., description="name")

    async def execute_greet(input: dict, context: ToolContext) -> ToolResult:
        return ToolResult.text(f"Hello, {input['name']}!")

    editor = ToolEditor()
    editor.add("greet", "Greet someone by name", GreetInput, execute=execute_greet)
    svc = ToolService(editor)
    result = await svc.execute(
        "greet", {"name": "world"},
        ToolContext(session_id="s", agent="a", message_id="m", call_id="c"),
    )
    assert result.render() == "Hello, world!", result.render()
    print("custom tool: OK")


async def verify_sdk_local() -> None:
    from opencode_sdk import OpenCode

    tmp = tempfile.mkdtemp()
    oc = OpenCode(db_path=Path(tmp) / "sdk.db")
    oc.local.loop = oc.local.loop.__class__(
        store=oc.local.store,
        tools=oc.local.tool_service,
        bus=oc.local.bus,
        permission=oc.local.permission,
        client_factory=lambda provider: FakeClient(),
    )
    s = await oc.session("demo")
    await oc.prompt(s.session_id, "hello")
    msgs = await s.messages()
    assert [m["type"] for m in msgs] == ["user", "assistant"], msgs
    await oc.close()
    print("sdk local: OK")


async def main() -> None:
    await verify_permission()
    await verify_custom_tool()
    await verify_sdk_local()
    print("ALL DOC EXAMPLES OK")


if __name__ == "__main__":
    asyncio.run(main())

