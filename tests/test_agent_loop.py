"""End-to-end AgentLoop tests driven by a fake LLM client.

The fake client lets us verify the whole loop -- prompt -> streaming ->
tool execution -> tool-result feedback -> final answer -- without any
network. It also proves transcripts persist to SQLite and events fan out
on the bus.
"""

import pytest

from opencode_ai.client import LLMClient, TextDelta, ToolCallDelta
from opencode_core.agent import AgentLoop
from opencode_core.bus import Bus
from opencode_core.builtins import register_all
from opencode_core.database import Database
from opencode_core.permission import PermissionService
from opencode_core.session import SessionStore
from opencode_core.tool import ToolEditor, ToolService
from opencode_schema.session import AssistantMessage, ToolMessage

from .conftest import ScriptedClient


@pytest.fixture
def harness(tmp_path):
    db = Database(tmp_path / "agent.db")
    store = SessionStore(db)
    editor = ToolEditor()
    register_all(editor)
    tools = ToolService(editor)
    bus = Bus()
    permission = PermissionService()

    collected: list = []

    async def on_event(event):
        collected.append(event)

    bus.subscribe(None, on_event)

    def make_loop(client: LLMClient) -> AgentLoop:
        return AgentLoop(
            store=store,
            tools=tools,
            bus=bus,
            permission=permission,
            client_factory=lambda provider: client,
        )

    return {
        "store": store,
        "bus": bus,
        "events": collected,
        "make_loop": make_loop,
    }


async def test_single_turn_text(harness, tmp_path):
    """A plain text answer: one assistant message persisted, no tool calls."""
    client = ScriptedClient([[TextDelta(text="Hello there!")]])
    loop = harness["make_loop"](client)
    sid = harness["store"].create(title="t", model="openai/gpt-4o")

    await loop.run(sid, "say hi")

    msgs = harness["store"].messages(sid)
    assert [m.type for m in msgs] == ["user", "assistant"]
    assistant = msgs[-1]
    assert assistant.content[0].text == "Hello there!"
    assert not harness["store"].is_claimed(sid)  # released after run


async def test_multi_step_tool_loop(harness, tmp_path):
    """Full loop: tool call -> executed -> result fed back -> final answer."""
    target = tmp_path / "data.txt"
    target.write_text("payload-42", encoding="utf-8")

    import json

    client = ScriptedClient(
        [
            [
                TextDelta(text="Let me read that file."),
                ToolCallDelta(
                    id="call_1",
                    name="read",
                    arguments=json.dumps({"path": str(target)}),
                ),
            ],
            [TextDelta(text="The file contains payload-42.")],
        ]
    )
    loop = harness["make_loop"](client)
    sid = harness["store"].create(title="t", model="openai/gpt-4o")

    await loop.run(sid, "what's in the file?")

    msgs = harness["store"].messages(sid)
    types = [m.type for m in msgs]
    assert types == ["user", "assistant", "tool", "assistant"]

    # First assistant message carries the tool call.
    tool_call_msg = msgs[1]
    assert isinstance(tool_call_msg, AssistantMessage)
    assert tool_call_msg.content[1].name == "read"

    # The tool message holds the real file content.
    tool_msg = msgs[2]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_name == "read"
    assert "payload-42" in tool_msg.output
    assert tool_msg.is_error is False

    # Final assistant message answers.
    final = msgs[3]
    assert isinstance(final, AssistantMessage)
    assert final.content[0].text == "The file contains payload-42."

    # Events were fanned out on the bus.
    assert any(e.type == "message.updated" for e in harness["events"])


async def test_provider_error_persisted(harness):
    """A provider error becomes an error assistant message, not a crash."""
    from opencode_ai.client import ProviderError

    client = ScriptedClient([[ProviderError(message="boom")]])
    loop = harness["make_loop"](client)
    sid = harness["store"].create(title="t")

    await loop.run(sid, "hi")

    msgs = harness["store"].messages(sid)
    assert msgs[-1].type == "assistant"
    assert msgs[-1].error == "boom"
    assert msgs[-1].finish_reason == "error"


async def test_permission_denied_tool_not_executed(harness, tmp_path):
    """A denied tool call yields a "Permission denied" tool message."""
    target = tmp_path / "secret.txt"
    target.write_text("top secret", encoding="utf-8")

    permission = PermissionService()
    from opencode_schema.permission import PermissionRule

    permission.add_rule(PermissionRule(action="read", resource=str(target), mode="deny"))

    import json

    client = ScriptedClient(
        [
            [ToolCallDelta(id="call_1", name="read", arguments=json.dumps({"path": str(target)}))],
            [TextDelta(text="ok")],
        ]
    )

    db = Database(tmp_path / "perm.db")
    store = SessionStore(db)
    editor = ToolEditor()
    register_all(editor)
    loop = AgentLoop(
        store=store,
        tools=ToolService(editor),
        bus=Bus(),
        permission=permission,
        client_factory=lambda provider: client,
    )
    sid = store.create(title="t")

    await loop.run(sid, "read the secret")

    tool_msg = [m for m in store.messages(sid) if m.type == "tool"][0]
    assert tool_msg.is_error is True
    assert "Permission denied" in tool_msg.output
    db.close()
