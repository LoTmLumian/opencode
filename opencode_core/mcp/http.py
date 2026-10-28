"""streamable HTTP MCP 传输（简化版，对应原 MCP streamable HTTP）。

远程传输：客户端通过 HTTP 与一个远程 MCP 服务器通信。标准 streamable HTTP
传输是——客户端 ``POST /message`` 发送 JSON-RPC 请求，服务器用 SSE
（Server-Sent Events）返回响应；此外客户端可开一条 ``GET /message`` 长连接
接收服务器的主动推送。

本实现刻意简化（对应原 SDK 的 ``StreamableHTTPClientTransport`` 的一个学习用
子集）：

* **请求-响应对 1:1**：每次 ``_send`` 直接 ``POST`` 单个 JSON-RPC 消息，并从
  该次响应的 SSE 流里解析出匹配 ``id`` 的响应，就地分发。
* **忽略服务器主动推送**：不维护常驻的 ``GET /message`` 长连接，所以服务器
  主动推送的 notification 不会被处理。

因此 ``_receive_loop`` 在此实现里是空操作——响应都在 ``_send`` 内同步处理。
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from opencode_core.mcp.client import McpClient


class HttpMcpClient(McpClient):
    """通过 streamable HTTP 与一个远程 MCP 服务器通信的客户端。

    ``config`` 是一个 ``RemoteMcpConfig``，提供：
      * ``url``：streamable HTTP 端点（如 ``https://example.com/mcp``）
      * ``headers``：附加到每个请求的 HTTP 头（如认证头）
    """

    def __init__(self, name: str, config) -> None:
        """构造 HTTP 客户端。

        Args:
            name: 服务器名称。
            config: ``RemoteMcpConfig``（含 url / headers）。
        """
        super().__init__(name)
        self.config = config
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        self._closed = False

    # ------------------------------------------------------------------ #
    # 生命周期                                                            #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """准备就绪（httpx 客户端已在构造时创建）。"""
        self._closed = False

    async def stop(self) -> None:
        """关闭底层 HTTP 客户端。"""
        self._closed = True
        await self._http.aclose()

    # ------------------------------------------------------------------ #
    # 传输层实现                                                           #
    # ------------------------------------------------------------------ #

    async def _send(self, message: dict[str, Any]) -> None:
        """POST 一个 JSON-RPC 消息并就地处理对应的 SSE 响应。

        简化实现：每次发送都独立 ``POST`` 一次。带 ``id`` 的消息是请求，需要
        从响应的 SSE 流里解析出匹配该 ``id`` 的 JSON-RPC 响应并分发；不带
        ``id`` 的是 notification，POST 后无需等待响应。
        """
        if self._closed:
            raise RuntimeError(f"{self.name}: http transport not connected")
        url = self.config.url
        if not url:
            raise RuntimeError(f"{self.name}: remote MCP config has no url")

        request_id = message.get("id")
        headers = {
            "Content-Type": "application/json",
            # 声明我们接受 SSE 流式响应。
            "Accept": "application/json, text/event-stream",
            **self.config.headers,
        }
        resp = await self._http.post(
            url,
            content=json.dumps(message, ensure_ascii=False),
            headers=headers,
        )
        resp.raise_for_status()

        # notification：无 id，无需解析响应体。
        if request_id is None:
            return

        # 解析响应体（可能是 application/json，也可能是 text/event-stream）。
        content_type = resp.headers.get("content-type", "")
        body = resp.text
        if "text/event-stream" in content_type:
            messages = self._parse_sse(body)
        else:
            # 非 SSE：整个 body 应是一个 JSON-RPC 响应。
            try:
                messages = [json.loads(body)]
            except json.JSONDecodeError:
                messages = []

        # 逐个分发；只关心与本次请求 id 匹配的响应。
        for parsed in messages:
            if isinstance(parsed, dict):
                self._handle_message(parsed)

    async def _receive_loop(self) -> None:
        """空操作：本实现不维护常驻接收流，响应都在 ``_send`` 内处理。"""

    # ------------------------------------------------------------------ #
    # SSE 解析                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_sse(body: str) -> list[dict[str, Any]]:
        """解析 SSE 文本，取出所有 ``data:`` 载荷中的 JSON 对象。

        SSE 事件格式：以 ``event:``/``data:`` 开头的行，事件之间用空行分隔。
        MCP 的 SSE 数据行是一个 JSON-RPC 消息。这里把每个 ``data:`` 行里
        的 JSON 解析出来收集返回。
        """
        messages: list[dict[str, Any]] = []
        data_lines: list[str] = []
        for raw_line in body.splitlines():
            line = raw_line.rstrip("\r")
            if line == "":
                # 空行结束一个事件：把累积的 data 解析成一个 JSON。
                if data_lines:
                    payload = "\n".join(data_lines)
                    try:
                        obj = json.loads(payload)
                        if isinstance(obj, dict):
                            messages.append(obj)
                    except json.JSONDecodeError:
                        pass
                    data_lines = []
            elif line.startswith("data:"):
                # ``data:`` 后面的内容（去掉一个可选前导空格）。
                data_lines.append(line[len("data:"):].lstrip())
        # 处理末尾没有空行收尾的最后一个事件。
        if data_lines:
            payload = "\n".join(data_lines)
            try:
                obj = json.loads(payload)
                if isinstance(obj, dict):
                    messages.append(obj)
            except json.JSONDecodeError:
                pass
        return messages
