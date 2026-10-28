from __future__ import annotations

"""高层 SDK：本地模式（直接实例化 core 服务）和远程模式（连 server）。"""

from pathlib import Path
from typing import Any

from opencode_core.agent import AgentLoop
from opencode_core.bus import Bus
from opencode_core.database import Database
from opencode_core.permission import PermissionService
from opencode_core.session import SessionStore
from opencode_core.tool import ToolEditor, ToolService
from opencode_core.builtins import register_all

from .client import OpenCodeClient

__all__ = ["LocalRuntime", "OpenCode"]


class LocalRuntime:
    """本地运行时：直接组合 core 各组件，不启动 HTTP server。

    复用 opencode_server.App 的组合方式，但只暴露编程接口（无 FastAPI 路由）。
    """

    def __init__(self, db_path: str | Path = ".data/opencode.db") -> None:
        self.db = Database(db_path)
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

    async def create_session(self, title: str = "") -> str:
        return self.store.create(title=title)

    async def get_session(self, session_id: str) -> dict:
        record = self.store.get(session_id)
        if record is None:
            raise KeyError(f"session not found: {session_id}")
        return {
            "id": record.id,
            "title": record.title,
            "agent": record.agent,
            "model": record.model_str,
            "created": record.created,
            "updated": record.updated,
        }

    async def list_sessions(self) -> list[dict]:
        return [s.model_dump() for s in self.store.list()]

    async def get_messages(self, session_id: str) -> list[dict]:
        return [m.model_dump() for m in self.store.messages(session_id)]

    async def prompt(self, session_id: str, text: str) -> None:
        await self.loop.run(session_id, text)

    def subscribe(self):
        """本地事件流：返回一个 async iterator，yield 本地 bus 事件。"""
        import asyncio

        from opencode_schema.event import EventMessage

        queue: asyncio.Queue = asyncio.Queue()

        async def _handler(event: Any) -> None:
            await queue.put(EventMessage.from_event(event))

        self.bus.subscribe(None, _handler)

        async def _gen():
            try:
                while True:
                    yield await queue.get()
            finally:
                self.bus.unsubscribe(None, _handler)

        return _gen()

    def close(self) -> None:
        self.db.close()


class OpenCode:
    """组合入口：本地直接跑 AgentLoop（不启动 HTTP server），或连远程 server。

    Args:
        db_path: 本地模式的数据文件路径。
        base_url: 若提供，则连接远程 opencode-server，否则走本地模式。
    """

    def __init__(self, *, db_path: str | Path = ".data/opencode.db", base_url: str | None = None) -> None:
        if base_url:
            self.client = OpenCodeClient(base_url)
            self.local = None
        else:
            self.local = LocalRuntime(db_path)
            self.client = None

    async def session(self, title: str = ""):
        """创建（或返回）一个会话句柄。"""
        if self.client is not None:
            session_id = await self.client.create_session(title)
            return RemoteSession(self.client, session_id)
        session_id = await self.local.create_session(title)
        return LocalSession(self.local, session_id)

    async def prompt(self, session_id: str, text: str):
        if self.client is not None:
            return await self.client.prompt(session_id, text)
        await self.local.prompt(session_id, text)

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
        elif self.local is not None:
            self.local.close()


class _BaseSession:
    """会话句柄基类：封装常见操作。"""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id


class RemoteSession(_BaseSession):
    """远程会话句柄。"""

    def __init__(self, client: OpenCodeClient, session_id: str) -> None:
        super().__init__(session_id)
        self.client = client

    async def prompt(self, text: str) -> dict:
        return await self.client.prompt(self.session_id, text)

    async def messages(self) -> list[dict]:
        return await self.client.get_messages(self.session_id)

    async def info(self) -> dict:
        return await self.client.get_session(self.session_id)


class LocalSession(_BaseSession):
    """本地会话句柄。"""

    def __init__(self, runtime: LocalRuntime, session_id: str) -> None:
        super().__init__(session_id)
        self.runtime = runtime

    async def prompt(self, text: str) -> None:
        await self.runtime.prompt(self.session_id, text)

    async def messages(self) -> list[dict]:
        return await self.runtime.get_messages(self.session_id)

    async def info(self) -> dict:
        return await self.runtime.get_session(self.session_id)
