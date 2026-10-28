from __future__ import annotations

"""FastAPI 服务端（对应原 packages/server，HTTP RPC + SSE 事件流）。

API 设计（简化版原项目 /api 路由）：
- POST /api/session            -> 创建会话 {session_id}
- GET  /api/session            -> 会话列表 [SessionSummary]
- GET  /api/session/{id}       -> 会话详情（元数据 + 消息）
- POST /api/session/{id}/prompt -> {text} -> 触发 agent 循环（后台任务），返回 {ok}
- POST /api/session/{id}/messages -> 追加用户消息（等价 prompt）
- GET  /api/session/{id}/message -> 读消息列表
- GET  /api/event              -> SSE 事件流（Bus 事件推给客户端）
- GET  /api/tool               -> 工具列表
- GET  /api/model              -> 模型/配置信息
"""

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from opencode_core.agent import AgentLoop
from opencode_core.bus import Bus
from opencode_core.database import Database
from opencode_core.mcp import McpRegistry
from opencode_core.permission import PermissionService
from opencode_core.session import SessionStore
from opencode_core.tool import ToolEditor, ToolService
from opencode_core.builtins import register_all
from opencode_schema.event import Event, EventMessage, EventType

__all__ = ["App"]


# ---------------------------------------------------------------------------
# Request / response bodies (kept tiny for a learning rewrite)
# ---------------------------------------------------------------------------


class CreateSessionBody(BaseModel):
    title: str = ""


class PromptBody(BaseModel):
    text: str


class AddMessageBody(BaseModel):
    text: str


class _ThreadSafeDatabase(Database):
    """SQLite 连接允许跨线程使用。

    FastAPI 的 TestClient 在 worker 线程里运行 ASGI 应用，而 App 在创建时于
    主线程建立连接；``check_same_thread=False`` 让连接可被任意线程复用，
    避免 ``SQLite objects created in a thread can only be used in that same thread``。
    生产 uvicorn（单事件循环线程）同样受益。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()


# ---------------------------------------------------------------------------
# Application root
# ---------------------------------------------------------------------------


class App:
    """组装核心服务 + FastAPI 应用。

    App 是组合根：创建 Database、SessionStore、ToolEditor(+内置工具)、
    ToolService、Bus、PermissionService、AgentLoop，然后构建 FastAPI app。
    """

    def __init__(self, db_path: str | Path = ".data/opencode.db", *, config=None) -> None:
        self.db = _ThreadSafeDatabase(db_path)
        self.store = SessionStore(self.db)
        self.bus = Bus()
        self.tools = ToolEditor()
        register_all(self.tools)
        self.tool_service = ToolService(self.tools)
        self.permission = PermissionService()
        self.loop = AgentLoop(
            store=self.store,
            tools=self.tool_service,
            bus=self.bus,
            permission=self.permission,
        )
        # mcp registry（可选，config 里有 mcp servers 时连接）。
        self.mcp = McpRegistry(self.tools)
        self.config = config or {}
        # 后台 agent 任务引用，避免被 GC（asyncio 只保留弱引用）。
        self._bg_tasks: set[asyncio.Task] = set()
        self._subscribers: list[asyncio.Queue[Event]] = []

        self.app = FastAPI(title="opencode-python", version="0.1.0")
        self._register_routes()

    # -- SSE --------------------------------------------------------------

    def _sse_feed(self, request: Request) -> StreamingResponse:
        """构造 SSE 响应：为每个客户端建一个队列，bus 事件广播进去。

        格式为 ``event: <type>\\ndata: <json>\\n\\n``。客户端断开时移除队列。
        """
        return StreamingResponse(
            self._sse_gen(request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _sse_gen(self, request: Request):
        """SSE 异步生成器：从队列里取事件并逐条 yield。"""
        queue: asyncio.Queue[Event] = asyncio.Queue()

        async def _handler(event: Event) -> None:
            await queue.put(event)

        self.bus.subscribe(None, _handler)
        self._subscribers.append(queue)
        try:
            while True:
                # 探测断开：若客户端已断开会话，停止发送。
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # 心跳，保持连接存活。
                    yield ": keep-alive\n\n"
                    continue
                msg = EventMessage.from_event(event)
                yield f"event: {msg.event}\ndata: {json.dumps(msg.data, ensure_ascii=False)}\n\n"
        finally:
            self.bus.unsubscribe(None, _handler)
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    # -- helpers ----------------------------------------------------------

    def _publish_sync(self, event: Event) -> None:
        """把事件同步广播给所有活跃 SSE 客户端队列。"""
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except Exception:  # noqa: BLE001 - a full/closed queue must not break others
                pass

    def _start_bg(self, coro: Any) -> None:
        """在后台跑一个协程（不阻塞 HTTP 响应），并追踪其生命周期。"""
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _run_agent(self, session_id: str, text: str) -> None:
        """后台任务：跑 agent 循环，结束时广播 session.idle / session.error。"""
        try:
            await self.loop.run(session_id, text)
        except Exception as exc:  # noqa: BLE001 - report to SSE, don't crash the task
            self._publish_sync(
                Event(
                    type="session.error",
                    session_id=session_id,
                    payload={"message": str(exc)},
                )
            )

    # -- routes -----------------------------------------------------------

    def _register_routes(self) -> None:
        app = self.app

        @app.post("/api/session")
        async def create_session(body: CreateSessionBody):
            sid = self.store.create(title=body.title)
            return {"session_id": sid}

        @app.get("/api/session")
        async def list_sessions():
            return [s.model_dump() for s in self.store.list()]

        @app.get("/api/session/{session_id}")
        async def get_session(session_id: str):
            record = self.store.get(session_id)
            if record is None:
                raise HTTPException(status_code=404, detail="session not found")
            return {
                "metadata": {
                    "id": record.id,
                    "title": record.title,
                    "agent": record.agent,
                    "model": record.model_str,
                    "created": record.created,
                    "updated": record.updated,
                },
                "messages": [m.model_dump() for m in self.store.messages(session_id)],
            }

        @app.post("/api/session/{session_id}/prompt")
        async def prompt(session_id: str, body: PromptBody):
            if self.store.get(session_id) is None:
                raise HTTPException(status_code=404, detail="session not found")
            self._start_bg(self._run_agent(session_id, body.text))
            return {"ok": True}

        @app.post("/api/session/{session_id}/messages")
        async def add_message(session_id: str, body: AddMessageBody):
            if self.store.get(session_id) is None:
                raise HTTPException(status_code=404, detail="session not found")
            self._start_bg(self._run_agent(session_id, body.text))
            return {"ok": True}

        @app.get("/api/session/{session_id}/message")
        async def get_messages(session_id: str):
            if self.store.get(session_id) is None:
                raise HTTPException(status_code=404, detail="session not found")
            return [m.model_dump() for m in self.store.messages(session_id)]

        @app.get("/api/event")
        async def event_stream(request: Request):
            return self._sse_feed(request)

        @app.get("/api/tool")
        async def list_tools():
            return [
                {
                    "name": t.definition.name,
                    "description": t.definition.description,
                    "inputSchema": t.definition.input_schema.model_dump(exclude_none=True),
                }
                for t in self.tools.list()
            ]

        @app.get("/api/model")
        async def get_model():
            return {
                "id": self.config.get("model", "openai/gpt-4o"),
                "provider": self.config.get("provider", "openai"),
                "name": "opencode-python",
            }

    # -- lifecycle --------------------------------------------------------

    def start(self, host: str = "127.0.0.1", port: int = 3456) -> None:
        """用 uvicorn 启动服务。"""
        import uvicorn

        uvicorn.run(self.app, host=host, port=port)
