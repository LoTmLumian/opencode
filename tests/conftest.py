"""Shared test fixtures."""

from typing import AsyncIterator

import pytest

from opencode_ai.client import (
    LLMClient,
    LLMEvent,
    LLMRequest,
    MessageStop,
    StepFinish,
    StepStart,
    TextDelta,
)


class ScriptedClient(LLMClient):
    """A fake LLMClient that plays back a script of tool-call/text turns.

    After the script is exhausted the client stops calling tools and answers
    with a plain ``Done.`` text turn, so multi-step loops terminate cleanly.
    """

    def __init__(self, turns: list[list[LLMEvent]]) -> None:
        self.turns = list(turns)
        self.step = 0
        self.request_models: list[str] = []

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        self.request_models.append(request.model)
        yield StepStart()
        if self.step < len(self.turns):
            events = self.turns[self.step]
            self.step += 1
        else:
            events = [TextDelta(text="Done.")]
        for event in events:
            yield event
        yield StepFinish()
        yield MessageStop()
