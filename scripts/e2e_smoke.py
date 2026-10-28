"""End-to-end smoke test: drive AgentLoop with a fake LLM client.

Verifies the full multi-step tool loop (user -> assistant(tool call) ->
tool -> assistant(text)) and SQLite persistence, offline.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opencode_ai.client import MessageStop, StepFinish, StepStart, TextDelta, ToolCallDelta
from opencode_server.server import App


class FakeClient:
    def __init__(self) -> None:
        self.step = 0

    async def stream(self, request):
        self.step += 1
        yield StepStart()
        if self.step == 1:
            yield TextDelta(text="Listing directory. ")
            yield ToolCallDelta(
                id="call_1",
                name="glob",
                arguments='{"pattern": "*.py", "path": "."}',
            )
        else:
            yield TextDelta(text="Found the python files.")
        yield StepFinish()
        yield MessageStop()


async def main() -> None:
    tmp = tempfile.mkdtemp()
    app = App(db_path=Path(tmp) / "e2e.db")
    app.loop = app.loop.__class__(
        store=app.store,
        tools=app.tool_service,
        bus=app.bus,
        permission=app.permission,
        client_factory=lambda provider: FakeClient(),
    )
    sid = app.store.create(title="e2e", model="openai/gpt-4o")
    await app.loop.run(sid, "list python files")

    msgs = app.store.messages(sid)
    print("transcript:", [m.type for m in msgs])
    for m in msgs:
        if m.type == "assistant":
            for part in m.content:
                print("  assistant part:", part.type, getattr(part, "text", "")[:40])
        if m.type == "tool":
            print("  tool:", m.tool_name, "->", m.output[:80])

    assert [m.type for m in msgs] == ["user", "assistant", "tool", "assistant"], "unexpected transcript"
    print("E2E OK")
    app.db.close()


if __name__ == "__main__":
    asyncio.run(main())
