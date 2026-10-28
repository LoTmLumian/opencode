from __future__ import annotations

"""SDK 客户端（对应原 packages/client，HTTP RPC + SSE 事件订阅）。"""

import json
from typing import AsyncIterator

import httpx

from opencode_schema.event import EventMessage

__all__ = ["OpenCodeClient"]


class OpenCodeClient:
    """远程连接到 opencode-server 的客户端。

    - session 管理：create / list / get / prompt
    - 事件订阅：SSE 流（async iterator）
    """

    def __init__(self, base_url: str = "http://127.0.0.1:3456") -> None:
        self.base_url = base_url.rstrip("/")
        # ``trust_env=False`` 让 httpx 忽略系统/环境代理设置，避免公司透明代理
        # 拦截 localhost 请求（表现为 504 Gateway Time-out）。SDK 面向本地 server。
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(120.0),
            trust_env=False,
        )

    # -- sessions ---------------------------------------------------------

    async def create_session(self, title: str = "") -> str:
        resp = await self._http.post("/api/session", json={"title": title})
        resp.raise_for_status()
        return resp.json()["session_id"]

    async def list_sessions(self) -> list[dict]:
        resp = await self._http.get("/api/session")
        resp.raise_for_status()
        return resp.json()

    async def get_session(self, session_id: str) -> dict:
        resp = await self._http.get(f"/api/session/{session_id}")
        resp.raise_for_status()
        return resp.json()

    async def prompt(self, session_id: str, text: str) -> dict:
        resp = await self._http.post(
            f"/api/session/{session_id}/prompt", json={"text": text}
        )
        resp.raise_for_status()
        return resp.json()

    async def add_message(self, session_id: str, text: str) -> dict:
        resp = await self._http.post(
            f"/api/session/{session_id}/messages", json={"text": text}
        )
        resp.raise_for_status()
        return resp.json()

    async def get_messages(self, session_id: str) -> list[dict]:
        resp = await self._http.get(f"/api/session/{session_id}/message")
        resp.raise_for_status()
        return resp.json()

    # -- tools / model ----------------------------------------------------

    async def list_tools(self) -> list[dict]:
        resp = await self._http.get("/api/tool")
        resp.raise_for_status()
        return resp.json()

    async def get_model(self) -> dict:
        resp = await self._http.get("/api/model")
        resp.raise_for_status()
        return resp.json()

    # -- SSE --------------------------------------------------------------

    async def subscribe(self) -> AsyncIterator[EventMessage]:
        """SSE 事件流：返回 async iterator，yield `EventMessage`。"""
        async with self._http.stream("GET", "/api/event") as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_name = line[len("event:") :].strip()
                    continue
                if line.startswith("data:"):
                    data = line[len("data:") :].strip()
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    yield EventMessage(event=event_name, data=payload)

    # -- lifecycle --------------------------------------------------------

    async def close(self) -> None:
        await self._http.aclose()
