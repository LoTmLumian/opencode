from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Awaitable, Callable

from opencode_schema.event import Event

__all__ = ["Bus", "Subscriber"]


Subscriber = Callable[[Event], Awaitable[None] | None]


class Bus:
    """In-process event bus (mirrors `packages/core/src/bus.ts`).

    Subscribers register per event type; `publish` fans out to all matching
    subscribers. This is the local, in-memory channel — the server layer
    bridges it to SSE for remote clients.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._all: list[Subscriber] = []

    def subscribe(self, event_type: str | None, fn: Subscriber) -> None:
        """Register a subscriber. ``event_type=None`` subscribes to everything."""
        if event_type is None:
            self._all.append(fn)
        else:
            self._subscribers[event_type].append(fn)

    def unsubscribe(self, event_type: str | None, fn: Subscriber) -> None:
        if event_type is None:
            if fn in self._all:
                self._all.remove(fn)
        elif fn in self._subscribers.get(event_type, []):
            self._subscribers[event_type].remove(fn)

    async def publish(self, event: Event) -> None:
        """Fan an event out to all matching subscribers."""
        targets = [*self._all, *self._subscribers.get(event.type, [])]
        for fn in targets:
            result = fn(event)
            if result is not None:
                await result

    def publish_sync(self, event: Event) -> None:
        """Fire-and-forget publish for subscribers that don't need awaiting."""
        targets = [*self._all, *self._subscribers.get(event.type, [])]
        for fn in targets:
            result = fn(event)
            if result is not None:
                asyncio.ensure_future(result)
