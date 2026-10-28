"""MCP client tests: JSON-RPC reconciliation + tool registration.

These tests exercise the transport-agnostic core (McpClient) with an
in-memory fake transport, avoiding any subprocess or network.
"""

import asyncio
from typing import Any

import pytest

from opencode_core.mcp.client import McpClient, McpError
from opencode_core.tool import ToolEditor
from opencode_schema.tool import ToolContext


class _InMemoryTransport(McpClient):
    """A fake transport that answers requests from a canned script."""

    def __init__(self, name: str, script: dict[str, list]) -> None:
        super().__init__(name)
        # method -> list of results to return in order.
        self.script = dict(script)
        self.sent: list[dict] = []
        self.started = False
        self.stopped = False

    async def _send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)
        method = message.get("method")
        if message.get("id") is None:
            return  # notification, no reply
        bucket = self.script.get(method, [])
        result = bucket.pop(0) if bucket else {}
        self._handle_message(
            {"jsonrpc": "2.0", "id": message["id"], "result": result}
        )

    async def _receive_loop(self) -> None:
        return

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


async def test_initialize_handshake():
    client = _InMemoryTransport(
        "srv",
        {"initialize": [{"name": "demo", "version": "1.0"}]},
    )
    info = await client.initialize()
    assert info["name"] == "demo"
    # The initialized notification should have been sent with no id.
    assert client.sent[-1]["method"] == "notifications/initialized"
    assert "id" not in client.sent[-1]


async def test_request_matches_correct_id():
    client = _InMemoryTransport("srv", {"tools/list": [{"tools": []}]})
    # Two concurrent requests must each resolve with their own result.
    async def call(method):
        return await client.request(method)

    results = await asyncio.gather(call("tools/list"), call("ping"))
    assert results == [{"tools": []}, {}]


async def test_jsonrpc_error_raises_mcp_error():
    client = _InMemoryTransport("srv", {})
    # Override _send to return a JSON-RPC error for a specific id.
    async def fail_send(message):
        if message.get("id") is not None:
            client._handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {"code": -32601, "message": "method not found"},
                }
            )

    client._send = fail_send
    with pytest.raises(McpError) as excinfo:
        await client.request("bogus/method")
    assert "-32601" in str(excinfo.value)


async def test_register_tools_namespaced():
    client = _InMemoryTransport(
        "math",
        {
            "tools/list": [
                {
                    "tools": [
                        {
                            "name": "add",
                            "description": "add two numbers",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"a": {"type": "number"}},
                                "required": ["a"],
                            },
                        }
                    ]
                }
            ],
            "tools/call": [{"content": [{"type": "text", "text": "3"}]}],
        },
    )
    await client.list_tools()
    editor = ToolEditor()
    client.register_tools(editor)

    tool = editor.get("math_add")
    assert tool is not None
    assert tool.definition.description == "add two numbers"

    result = await tool.execute(
        {"a": 1}, ToolContext(session_id="s", agent="a", message_id="m", call_id="c")
    )
    assert result.render() == "3"


async def test_call_tool_error_surfaces_in_result():
    client = _InMemoryTransport(
        "math",
        {
            "tools/list": [
                {"tools": [{"name": "add", "description": "", "inputSchema": {}}]}
            ],
        },
    )
    await client.list_tools()
    editor = ToolEditor()
    client.register_tools(editor)

    # Make the next call fail.
    async def fail_call(name, arguments):
        raise McpError("connection lost")

    client.call_tool = fail_call
    tool = editor.get("math_add")
    result = await tool.execute(
        {}, ToolContext(session_id="s", agent="a", message_id="m", call_id="c")
    )
    assert result.metadata.get("error") is True


def test_render_tool_result_text_content():
    result = {"content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]}
    assert McpClient._render_tool_result(result) == "hello\nworld"


def test_sse_parse_collects_data_events():
    body = (
        'event: message\n'
        'data: {"jsonrpc": "2.0", "id": 1, "result": {}}\n'
        '\n'
        'data: {"jsonrpc": "2.0", "id": 2, "result": {}}\n'
    )
    from opencode_core.mcp.http import HttpMcpClient

    msgs = HttpMcpClient._parse_sse(body)
    assert len(msgs) == 2
    assert msgs[0]["id"] == 1
    assert msgs[1]["id"] == 2
