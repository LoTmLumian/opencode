"""Permission engine tests: rule matching, findLast precedence, deny priority."""

import pytest

from opencode_core.permission import (
    PermissionBlockedError,
    PermissionDeniedError,
    PermissionService,
    evaluate,
)
from opencode_schema.permission import PermissionRequest, PermissionRule


@pytest.mark.parametrize(
    "action,resource,expected",
    [
        ("read", "a.txt", "allow"),
        ("write", "a.txt", "ask"),
        ("bash", "ls -la", "deny"),
        ("read", "other.md", "ask"),
    ],
)
def test_evaluate_basic(action, resource, expected):
    rules = [
        PermissionRule(action="read", resource="*.txt", mode="allow"),
        PermissionRule(action="bash", resource="*", mode="deny"),
    ]
    assert evaluate(action, resource, rules).decision == expected


def test_evaluate_find_last_wins():
    """Later-defined rules override earlier ones for the same match."""
    rules = [
        PermissionRule(action="read", resource="*", mode="deny"),
        PermissionRule(action="read", resource="*.md", mode="allow"),
    ]
    assert evaluate("read", "notes.md", rules).decision == "allow"
    assert evaluate("read", "main.py", rules).decision == "deny"


def test_evaluate_glob_star():
    """`*` matches everything; `**` crosses path separators."""
    assert PermissionRule(action="a", resource="*").matches("a", "anything/at/all")
    assert PermissionRule(action="a", resource="**/*.py").matches("a", "deep/nested/x.py")
    assert not PermissionRule(action="a", resource="*.py").matches("a", "deep/x.py")


async def test_assert_not_denied_allows_unknown():
    """assert_not_denied never asks and allows anything not explicitly denied."""
    svc = PermissionService()
    result = await svc.assert_not_denied(PermissionRequest(action="bash", resource="rm -rf /"))
    assert result.decision == "allow"


async def test_assert_deny_raises_blocked():
    svc = PermissionService(rules=[PermissionRule(action="bash", resource="*", mode="deny")])
    with pytest.raises(PermissionBlockedError):
        await svc.assert_(PermissionRequest(action="bash", resource="echo hi"))


async def test_assert_ask_without_handler_denied():
    """An ask with no user handler raises PermissionDeniedError."""
    svc = PermissionService()
    with pytest.raises(PermissionDeniedError):
        await svc.assert_(PermissionRequest(action="webfetch", resource="http://x"))


async def test_assert_ask_with_handler_and_save():
    """A granted ask is remembered so the next identical check skips the prompt."""
    asked = []

    async def handler(req):
        asked.append(req)
        return "allow"

    svc = PermissionService(ask_handler=handler)
    req = PermissionRequest(action="bash", resource="npm test")
    assert (await svc.assert_(req)).decision == "allow"
    assert (await svc.assert_(req)).decision == "allow"
    assert len(asked) == 1  # second time short-circuited by the saved rule


async def test_multi_resource_deny_blocks_all():
    """A denial on any resource blocks the whole request."""
    svc = PermissionService(rules=[PermissionRule(action="write", resource="secrets/*", mode="deny")])
    req = PermissionRequest(action="write", resources=["a.py", "secrets/key.txt"])
    with pytest.raises(PermissionBlockedError):
        await svc.assert_not_denied(req)
