from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = ["PermissionRule", "PermissionMode", "PermissionRequest", "PermissionDecision", "PermissionResult"]


PermissionMode = Literal["plan", "build"]

# A permission outcome: allow (auto-run), deny (blocked), or ask (prompt the user).
PermissionDecision = Literal["allow", "deny", "ask"]


class PermissionRule(BaseModel):
    """A single permission rule.

    Mirrors opencode's permission config: each rule matches a tool action
    (e.g. ``bash``, ``webfetch``) and optionally a glob of the target
    resource (e.g. ``*``, ``**/*.md``).
    """

    action: str
    # Glob pattern for the resource this rule applies to ("*" matches everything).
    resource: str = "*"
    mode: PermissionDecision = "ask"

    def matches(self, action: str, resource: str) -> bool:
        """Return True if this rule applies to the given action+resource."""
        if self.action != "*" and self.action != action:
            return False
        return _glob_match(self.resource, resource)


class PermissionRequest(BaseModel):
    """A permission check that needs to be evaluated or escalated to the user."""

    action: str
    resource: str = "*"
    # Multiple resources mirror the original project's multi-resource semantics.
    # If both `resource` and `resources` are provided, `resources` wins. This
    # field is optional to preserve back-compat with the single-resource form.
    resources: list[str] | None = None
    agent: str | None = None
    session_id: str | None = None
    message_id: str | None = None

    def resource_list(self) -> list[str]:
        """Return the effective list of resources to check.

        `resources` (if provided) takes priority; otherwise fall back to the
        single `resource` field.
        """
        return self.resources if self.resources is not None else [self.resource]


class PermissionResult(BaseModel):
    """The outcome of a permission evaluation."""

    decision: PermissionDecision
    # For "ask", which rule triggered the prompt (if any).
    rule: PermissionRule | None = None
    # Human-readable reason, e.g. the user's typed answer.
    reason: str | None = None


def _glob_match(pattern: str, value: str) -> bool:
    """Minimal glob matcher supporting ``*`` (any run of chars) and ``**``.

    ``**`` also crosses path separators. This is intentionally simple;
    the original uses `picomatch`.
    """
    import re

    if pattern == "*":
        return True
    if pattern == value:
        return True
    # Translate glob to regex.
    i, n = 0, len(pattern)
    out: list[str] = ["^"]
    while i < n:
        c = pattern[i]
        if c == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                out.append(".*")
                i += 2
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    out.append("$")
    return re.match("".join(out), value) is not None
