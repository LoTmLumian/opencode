"""HTTP server + SDK integration tests (FastAPI TestClient, fake LLM).

The server's agent loop runs with a scripted fake client, so the full
HTTP RPC path (create session -> prompt -> message) is verified offline.
"""

from fastapi.testclient import TestClient

from opencode_ai.client import TextDelta
from opencode_server.server import App

from .conftest import ScriptedClient


def _make_app(tmp_path, client):
    app = App(db_path=tmp_path / "server.db")
    app.loop = app.loop.__class__(
        store=app.store,
        tools=app.tool_service,
        bus=app.bus,
        permission=app.permission,
        client_factory=lambda provider: client,
    )
    return app


def test_session_lifecycle_http(tmp_path):
    client = ScriptedClient([[TextDelta(text="Hello from server")]])
    app = _make_app(tmp_path, client)
    tc = TestClient(app.app)

    # Create a session.
    resp = tc.post("/api/session", json={"title": "e2e"})
    assert resp.status_code == 200
    sid = resp.json()["session_id"]

    # Prompt it (background task).
    resp = tc.post(f"/api/session/{sid}/prompt", json={"text": "say hi"})
    assert resp.status_code == 200

    # Wait for the background agent task to finish.
    import time

    for _ in range(100):
        messages = tc.get(f"/api/session/{sid}/message").json()
        if any(m["type"] == "assistant" for m in messages):
            break
        time.sleep(0.02)

    messages = tc.get(f"/api/session/{sid}/message").json()
    types = [m["type"] for m in messages]
    assert types == ["user", "assistant"]
    assistant = messages[-1]
    assert assistant["content"][0]["text"] == "Hello from server"

    # Session listing + detail work.
    listing = tc.get("/api/session").json()
    assert any(s["id"] == sid for s in listing)

    detail = tc.get(f"/api/session/{sid}").json()
    assert detail["metadata"]["id"] == sid
    assert len(detail["messages"]) == 2

    app.db.close()


def test_tool_list_and_model_endpoints(tmp_path):
    client = ScriptedClient([])
    app = _make_app(tmp_path, client)
    tc = TestClient(app.app)

    tools = tc.get("/api/tool").json()
    names = {t["name"] for t in tools}
    assert "read" in names and "write" in names

    model = tc.get("/api/model").json()
    assert model["id"] == "openai/gpt-4o"

    app.db.close()


def test_404_for_missing_session(tmp_path):
    client = ScriptedClient([])
    app = _make_app(tmp_path, client)
    tc = TestClient(app.app)

    resp = tc.get("/api/session/ses_nope")
    assert resp.status_code == 404

    resp = tc.post("/api/session/ses_nope/prompt", json={"text": "hi"})
    assert resp.status_code == 404

    app.db.close()


def test_sdk_local_runtime_and_remote_client(tmp_path):
    """LocalRuntime + OpenCodeClient both drive the same session flow."""
    import asyncio

    import httpx

    from opencode_sdk.client import OpenCodeClient
    from opencode_sdk.opencode import OpenCode

    # Local SDK path: create a session and prompt it with a fake client.
    async def local_flow():
        oc = OpenCode(db_path=tmp_path / "sdk.db")
        oc.local.loop = oc.local.loop.__class__(
            store=oc.local.store,
            tools=oc.local.tool_service,
            bus=oc.local.bus,
            permission=oc.local.permission,
            client_factory=lambda provider: ScriptedClient([[TextDelta(text="local reply")]]),
        )
        session_id = await oc.local.create_session("sdk session")
        await oc.local.prompt(session_id, "hi")
        messages = await oc.local.get_messages(session_id)
        assert [m["type"] for m in messages] == ["user", "assistant"]
        await oc.close()

    asyncio.run(local_flow())

    # Remote SDK path against the ASGI app via httpx's ASGITransport (no socket).
    client = ScriptedClient([[TextDelta(text="remote reply")]])
    app = _make_app(tmp_path, client)
    transport = httpx.ASGITransport(app=app.app)

    async def remote_flow():
        remote = OpenCodeClient("http://testserver")
        await remote._http.aclose()
        remote._http = httpx.AsyncClient(
            base_url="http://testserver",
            transport=transport,
            timeout=httpx.Timeout(30.0),
        )
        sid = await remote.create_session("remote session")
        await remote.prompt(sid, "hello")

        for _ in range(100):
            messages = await remote.get_messages(sid)
            if any(m["type"] == "assistant" for m in messages):
                break
            await asyncio.sleep(0.02)
        assert [m["type"] for m in messages] == ["user", "assistant"]
        await remote.close()

    asyncio.run(remote_flow())
    app.db.close()
