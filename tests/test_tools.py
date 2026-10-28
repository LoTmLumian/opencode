"""Tool registry + built-in tools tests."""

import pytest
from pydantic import BaseModel, Field

from opencode_core.builtins import register_all
from opencode_core.tool import ToolEditor, ToolRegistrationError, ToolService
from opencode_schema.tool import ToolCallError, ToolContext, ToolResult


def _ctx(session_id="ses_1") -> ToolContext:
    return ToolContext(session_id=session_id, agent="build", message_id="msg_1", call_id="call_1")


def test_register_and_list():
    editor = ToolEditor()
    register_all(editor)
    names = [t.definition.name for t in editor.list()]
    for expected in ("read", "write", "edit", "glob", "grep", "shell", "todo", "webfetch", "subagent"):
        assert expected in names


def test_duplicate_registration_raises():
    editor = ToolEditor()
    editor.add("dup", "one", None, execute=None)
    with pytest.raises(ToolRegistrationError):
        editor.add("dup", "two", None)


def test_input_schema_from_pydantic_model():
    class Input(BaseModel):
        path: str = Field(..., description="target path")
        offset: int | None = None

    editor = ToolEditor()
    tool = editor.add("t", "desc", Input)
    schema = tool.definition.input_schema
    assert schema.type == "object"
    assert "path" in schema.properties
    assert "path" in schema.required


def test_to_llm_definitions():
    editor = ToolEditor()
    editor.add("t", "desc", None)
    defs = editor.to_llm_definitions()
    assert defs[0].name == "t"
    assert defs[0].inputSchema["type"] == "object"
    assert defs[0].inputSchema["properties"] == {}
    assert defs[0].inputSchema["required"] == []


async def test_service_unknown_tool_is_error_result():
    svc = ToolService()
    result = await svc.execute("nope", {}, _ctx())
    assert result.metadata.get("error") is True
    assert "Unknown tool" in result.render()


async def test_service_captures_tool_exception():
    async def boom(input, context):
        raise ToolCallError("bad file")

    editor = ToolEditor()
    editor.add("boom", "desc", None, execute=boom)
    svc = ToolService(editor)
    result = await svc.execute("boom", {}, _ctx())
    assert result.metadata.get("error") is True
    assert "bad file" in result.render()


async def test_read_tool(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("line1\nline2\nline3\n", encoding="utf-8")
    editor = ToolEditor()
    register_all(editor)
    svc = ToolService(editor)
    result = await svc.execute("read", {"path": str(f)}, _ctx())
    assert not result.metadata.get("error")
    assert "line2" in result.render()


async def test_read_tool_missing_file():
    editor = ToolEditor()
    register_all(editor)
    svc = ToolService(editor)
    result = await svc.execute("read", {"path": "/no/such/file.txt"}, _ctx())
    assert result.metadata.get("error") is True


async def test_write_tool_roundtrip(tmp_path):
    target = tmp_path / "out.py"
    editor = ToolEditor()
    register_all(editor)
    svc = ToolService(editor)
    result = await svc.execute("write", {"path": str(target), "content": "print(1)\n"}, _ctx())
    assert not result.metadata.get("error")
    assert target.read_text(encoding="utf-8") == "print(1)\n"


async def test_glob_tool(tmp_path):
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "b.txt").write_text("", encoding="utf-8")
    editor = ToolEditor()
    register_all(editor)
    svc = ToolService(editor)
    result = await svc.execute("glob", {"pattern": "*.py", "path": str(tmp_path)}, _ctx())
    assert "a.py" in result.render()
    assert "b.txt" not in result.render()
