"""Schema-layer unit tests: model refs, messages, tool results, session ids."""

import pytest
from pydantic import ValidationError

from opencode_schema.model import ModelRef, ToolResult
from opencode_schema.session import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
    new_session_id,
)


def test_model_ref_parse_roundtrip():
    ref = ModelRef.parse("openai/gpt-4o#default")
    assert ref.providerID == "openai"
    assert ref.id == "gpt-4o"
    assert ref.variant == "default"
    assert str(ref) == "openai/gpt-4o#default"


def test_model_ref_parse_no_variant():
    ref = ModelRef.parse("anthropic/claude-sonnet-4")
    assert ref.variant is None
    assert str(ref) == "anthropic/claude-sonnet-4"


@pytest.mark.parametrize("bad", ["nope", "/missing-provider", "provider/", "a/b/c#x#y"])
def test_model_ref_parse_rejects_invalid(bad):
    with pytest.raises(ValueError):
        ModelRef.parse(bad)


def test_model_ref_coerces_from_string():
    # ModelRef(providerID=..., id=...) is the canonical form; a string must
    # go through ModelRef.parse(). The model_validator handles ``parse``.
    ref = ModelRef(providerID="openai", id="gpt-4o")
    assert ref.id == "gpt-4o"
    assert str(ref) == "openai/gpt-4o"


def test_tool_result_render():
    tr = ToolResult.text("hello", path="a.txt")
    assert tr.render() == "hello"
    assert tr.metadata["path"] == "a.txt"
    # content wins over output when both are present.
    assert ToolResult(content="shown", output=42).render() == "shown"
    # output alone renders as a string too.
    assert ToolResult(output=42).render() == "42"


def test_session_message_discriminated_union():
    """Each concrete message round-trips through its `type` discriminator."""
    for msg in [
        UserMessage(text="hi"),
        AssistantMessage(agent="build", model=ModelRef.parse("openai/gpt-4o")),
        SystemMessage(text="boot"),
        ToolMessage(tool_call_id="c1", tool_name="read", output="ok"),
    ]:
        dumped = msg.model_dump()
        assert dumped["type"] == msg.type


def test_new_ids_are_unique_and_prefixed():
    ids = {new_session_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(i.startswith("ses_") for i in ids)
