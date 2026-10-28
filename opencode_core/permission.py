from __future__ import annotations

"""Permission engine (mirrors `packages/core/src/permission.ts`).

This module implements the runtime side of the permission system: an
`evaluate()` function that resolves a (action, resource) pair against an
ordered rule list, plus a `PermissionService` that combines rules from
several sources and applies the original project's decision semantics.

The static rule model (`PermissionRule`, `PermissionRequest`, ...) lives in
`opencode_schema.permission`; this module is the pure-Python runtime that
acts on it.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from opencode_schema.event import Event
from opencode_schema.permission import (
    PermissionDecision,
    PermissionMode,
    PermissionRequest,
    PermissionResult,
    PermissionRule,
)

__all__ = [
    "PermissionDeniedError",
    "PermissionBlockedError",
    "PermissionService",
    "evaluate",
]


class PermissionDeniedError(RuntimeError):
    """The user (or the absence of a handler) refused a permission request."""


class PermissionBlockedError(RuntimeError):
    """A permission rule denied the request outright — no room to ask the user.

    Attributes:
        rules: The full rule set that was in effect when the denial happened,
            useful for diagnostics.
    """

    def __init__(self, message: str, rules: list[PermissionRule] | None = None) -> None:
        super().__init__(message)
        self.rules = rules or []


def evaluate(action: str, resource: str, rules: list[PermissionRule]) -> PermissionResult:
    """Evaluate a single (action, resource) pair against an ordered rule list.

    Mirrors the original `Permission.evaluate` semantics:

    * Only rules whose `matches(action, resource)` returns true are considered.
    * Among the matches, the **last-defined** rule wins (findLast semantics:
      later rules override earlier ones). This lets a specific rule written
      after a broad one take precedence.
    * If nothing matches, the outcome defaults to ``ask`` (the original
      project's default effect).

    Returns a `PermissionResult` carrying the decision and the winning rule
    (if any).
    """
    matched = [r for r in rules if r.matches(action, resource)]
    if matched:
        rule = matched[-1]
        return PermissionResult(decision=rule.mode, rule=rule)
    return PermissionResult(decision="ask")


class PermissionService:
    """Permission service (mirrors `Permission.Service`).

    Rule sources are merged from lowest to highest precedence:

    1. Agent-provided permissions (``AgentDefinition.permissions``), passed
       per call via ``agent_rules``.
    2. Session-level rules added at runtime via :meth:`add_rule`.
    3. Rules the user has explicitly saved (allowed) via :meth:`add_saved`.

    Decision semantics (matching the original's ``denied()`` priority plus
    the ``effects.includes("ask")`` logic):

    * Any resource that is **deny**  -> the whole request is denied outright
      (raises `PermissionBlockedError`), no user prompt.
    * Otherwise any resource that is **ask** (and nothing is denied)
      -> prompt the user.
    * Otherwise -> **allow**.

    The request may carry multiple resources; each is checked independently
    and the most restrictive outcome governs the whole request.
    """

    def __init__(
        self,
        mode: PermissionMode = "build",
        *,
        rules: list[PermissionRule] | None = None,
        ask_handler: Callable[[PermissionRequest], Awaitable[PermissionDecision]] | None = None,
        on_event: Callable[[Event], None] | None = None,
    ) -> None:
        self.mode = mode  # plan / build
        self.rules = list(rules or [])
        # Injected callback that actually asks the user (implemented by the CLI
        # layer). If None, `_ask` refuses the request instead.
        self.ask_handler = ask_handler
        # Event callback for the bus/SSE layer. If provided, `_ask` emits
        # permission.ask / permission.resolve events.
        self.on_event = on_event
        # Rules the user has explicitly allowed, persisted across checks.
        self._saved: list[PermissionRule] = []

    def add_rule(self, rule: PermissionRule) -> None:
        """Register a session-level permission rule."""
        self.rules.append(rule)

    def add_saved(self, rule: PermissionRule) -> None:
        """Record a rule the user explicitly allowed (mirrors `PermissionSaved`)."""
        self._saved.append(rule)

    def _all_rules(self, agent_rules: list[PermissionRule] | None = None) -> list[PermissionRule]:
        """Merge all rule sources into one ordered list.

        Order matters because `evaluate` uses findLast: later entries override
        earlier ones, so saved rules are appended last (highest precedence).
        """
        return [*self.rules, *(agent_rules or []), *self._saved]

    async def assert_(
        self, request: PermissionRequest, agent_rules: list[PermissionRule] | None = None
    ) -> PermissionResult:
        """Assert permission for a request, escalating to the user when needed.

        Mirrors the original `Permission.assert` (which may ask):

        * deny -> raise `PermissionBlockedError`.
        * ask  -> go through the ask flow (:meth:`_ask`), returning its result.
        * allow -> return an allow result.

        Args:
            request: The permission check to perform.
            agent_rules: Optional agent-level rules merged under session rules.

        Returns:
            The resolved `PermissionResult`.

        Raises:
            PermissionBlockedError: If any resource is denied by a rule.
        """
        all_rules = self._all_rules(agent_rules)
        # Evaluate every resource the request targets.
        results = [evaluate(request.action, r, all_rules) for r in request.resource_list()]

        # Deny has priority: a single denial blocks the whole request.
        for res in results:
            if res.decision == "deny":
                raise PermissionBlockedError(
                    f"Permission denied: {request.action} {request.resource}", all_rules
                )

        # Nothing denied: if anything needs asking, ask once for the request.
        for res in results:
            if res.decision == "ask":
                decision = await self._ask(request)
                return PermissionResult(decision=decision, rule=res.rule)

        return PermissionResult(decision="allow")

    async def assert_not_denied(
        self, request: PermissionRequest, agent_rules: list[PermissionRule] | None = None
    ) -> PermissionResult:
        """Check only that nothing is denied — never ask the user.

        Mirrors the original `Permission.assertNotDenied`: a non-denied outcome
        (allow or ask) is collapsed to ``allow`` without prompting.

        Args:
            request: The permission check to perform.
            agent_rules: Optional agent-level rules.

        Returns:
            ``allow`` if no resource is denied.

        Raises:
            PermissionBlockedError: If any resource is denied by a rule.
        """
        all_rules = self._all_rules(agent_rules)
        for r in request.resource_list():
            res = evaluate(request.action, r, all_rules)
            if res.decision == "deny":
                raise PermissionBlockedError(
                    f"Permission denied: {request.action} {r}", all_rules
                )
        return PermissionResult(decision="allow")

    async def _ask(self, request: PermissionRequest) -> PermissionDecision:
        """Drive the ask flow for a request.

        Emits a ``permission.ask`` event; if an ``ask_handler`` is injected it
        is awaited to obtain the user's decision, and on ``allow`` the outcome
        is remembered as a saved rule so future checks skip the prompt. A
        ``permission.resolve`` event carries the final decision back.

        If no handler is configured, the request is refused by raising
        `PermissionDeniedError`.

        Raises:
            PermissionDeniedError: If no handler is available to ask the user.
        """
        if self.on_event:
            self.on_event(
                Event(
                    type="permission.ask",
                    session_id=request.session_id,
                    payload=request.model_dump(),
                )
            )

        if self.ask_handler is not None:
            decision = await self.ask_handler(request)
            if self.on_event:
                self.on_event(
                    Event(
                        type="permission.resolve",
                        session_id=request.session_id,
                        payload={"decision": decision},
                    )
                )
            if decision == "allow":
                # Remember this grant so it won't re-prompt (saved rules).
                self.add_saved(
                    PermissionRule(action=request.action, resource=request.resource, mode="allow")
                )
            return decision

        raise PermissionDeniedError(f"No permission handler for {request.action} {request.resource}")
