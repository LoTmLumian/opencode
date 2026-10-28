"""MCP 客户端抽象（对应原 mcp/client.ts）。

本模块提供两个核心类：

* ``McpClient`` —— 一个 MCP 服务器连接的抽象基类。它承载所有与具体传输无关的
  逻辑：JSON-RPC 请求/响应对账（``request`` / ``notify``）、MCP 协议方法
  （``initialize`` / ``list_tools`` / ``call_tool``）、生命周期（``start`` / ``stop``）
  以及把服务器暴露的工具注册进 ``ToolEditor``（``register_tools``）。
  具体的消息收发（写请求、读响应）由传输子类实现 ``_send`` 与 ``_receive_loop``。
* ``McpRegistry`` —— 管理多个 MCP 服务器：根据配置创建对应客户端、连接全部、
  把它们的工具全部注册进同一个 ``ToolEditor``、最后统一关闭。

JSON-RPC 对账方式：客户端为每个请求分配自增 ``id``，在 ``_pending`` 里存一个
``asyncio.Future``；发送请求后 await 这个 future。传输子类的 ``_receive_loop``
读入消息后调用 ``_handle_message``，后者按 ``id`` 找到 future 并 set_result /
set_exception，从而唤醒对应的 ``request`` 调用方。
"""
from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any

from opencode_schema.tool import ToolDefinition, ToolInputSchema, ToolContext, ToolResult
from opencode_core.tool import ToolEditor


class McpError(RuntimeError):
    """MCP 层错误基类（如服务器返回 JSON-RPC error，或传输失败）。"""


class McpClient(ABC):
    """一个 MCP 服务器连接：JSON-RPC over 某种传输。

    子类只需实现两件事：

    * ``_send(message)`` —— 把一个 JSON-RPC 消息（dict）实际写到传输层；
    * ``_receive_loop()`` —— 一个长期运行的任务，负责从传输层读消息并调用
      ``_handle_message`` 分发。

    公共方法（``request`` / ``initialize`` / ``list_tools`` / ...）与传输无关。
    """

    #: MCP 协议版本。这里固定一个较新的规范版本；真实环境可能需要按服务器协商。
    PROTOCOL_VERSION = "2024-11-05"
    #: 客户端自我描述，随 ``initialize`` 发给服务器。
    CLIENT_NAME = "opencode-python"
    CLIENT_VERSION = "0.1.0"

    def __init__(self, name: str) -> None:
        """构造一个客户端。

        Args:
            name: 服务器名称（配置里的 key，也是工具注册时的命名空间）。
        """
        self.name = name
        # 自增请求 id，保证同一连接内每个请求唯一。
        self._request_id = 0
        # id -> Future 的映射，用于把响应匹配回对应的 request 调用方。
        self._pending: dict[int, asyncio.Future] = {}
        # 记录 initialize 返回的 serverInfo（标题、版本等）。
        self.server_info: dict[str, Any] | None = None
        # 最近一次 list_tools 拉取到的工具列表缓存，供 register_tools 使用。
        self._tools_cache: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # JSON-RPC 基础                                                       #
    # ------------------------------------------------------------------ #

    def _next_id(self) -> int:
        """分配下一个请求 id（自增，从 1 开始）。"""
        self._request_id += 1
        return self._request_id

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """发一个 JSON-RPC 请求并等待对应响应。

        Args:
            method: 方法名，如 ``"tools/list"``。
            params: 可选的参数对象。

        Returns:
            响应中的 ``result`` 字段。

        Raises:
            McpError: 服务器返回了 JSON-RPC error，或连接/传输出错。
        """
        request_id = self._next_id()
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            message["params"] = params
        try:
            await self._send(message)
        except Exception as exc:  # noqa: BLE001 - 传输失败时清理 pending
            self._pending.pop(request_id, None)
            raise McpError(f"{self.name}: failed to send {method}: {exc}") from exc
        # 等待响应（由 _handle_message 通过 id 匹配并唤醒）。
        return await future

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """发一个 JSON-RPC notification（无 id，服务器不回响应）。"""
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        await self._send(message)

    # ------------------------------------------------------------------ #
    # MCP 协议方法                                                        #
    # ------------------------------------------------------------------ #

    async def initialize(self) -> dict[str, Any]:
        """执行 MCP initialize 握手。

        流程（对应原 SDK 的 connect）：\n
        1. 发 ``initialize`` 请求，携带 protocolVersion / capabilities / clientInfo；\n
        2. 服务器返回 serverInfo；\n
        3. 发 ``notifications/initialized`` notification 告知服务器已就绪；\n
        4. 之后才能调用 ``tools/list`` 等。

        Returns:
            服务器返回的 serverInfo dict（含 name / version / protocolVersion 等）。
        """
        result = await self.request(
            "initialize",
            {
                "protocolVersion": self.PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": self.CLIENT_NAME, "version": self.CLIENT_VERSION},
            },
        )
        self.server_info = result if isinstance(result, dict) else {}
        # 握手完成后通知服务器客户端已就绪。
        await self.notify("notifications/initialized")
        return self.server_info

    async def list_tools(self) -> list[dict[str, Any]]:
        """列出该服务器暴露的工具（``tools/list``）。

        Returns:
            原始工具描述 dict 列表（每个含 name / description / inputSchema 等）。
        """
        result = await self.request("tools/list")
        if isinstance(result, dict):
            self._tools_cache = list(result.get("tools", []))
        else:
            self._tools_cache = []
        return self._tools_cache

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """调用该服务器上的一个工具（``tools/call``）。

        Args:
            name: 工具名。
            arguments: 传给工具的参数对象。

        Returns:
            ``tools/call`` 的原始 result（含 content 数组等）。
        """
        return await self.request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )

    # ------------------------------------------------------------------ #
    # 生命周期                                                            #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """打开传输并启动接收循环（子类实现细节）。"""

    async def stop(self) -> None:
        """关闭传输并结束接收循环（子类实现细节）。"""

    # ------------------------------------------------------------------ #
    # 工具注册                                                            #
    # ------------------------------------------------------------------ #

    def register_tools(self, editor: ToolEditor) -> None:
        """把该服务器暴露的工具注册进 ``ToolEditor``。

        每个 MCP tool 变成一个本地工具：``ToolDefinition`` 的 schema 取自服务器
        声明的 ``inputSchema``，``execute`` 是一个包装——调用时转发给
        ``self.call_tool``，并把服务器返回的 content 转成 ``ToolResult``。

        工具名会加上 ``<server>_`` 前缀做命名空间，避免不同服务器之间、以及与
        本地内置工具重名冲突。

        Args:
            editor: 目标 ``ToolEditor``，工具被注册进去。
        """
        # 从缓存读取该服务器暴露的工具（由 list_tools 填充）。
        for tool in self._tools_cache:
            name = tool.get("name", "")
            if not name:
                continue
            # 以 `<server>_<tool>` 命名，避免跨服务器/与内置工具重名。
            full_name = f"{self.name}_{name}"
            description = tool.get("description", "")
            input_schema = tool.get("inputSchema") or tool.get("input_schema") or {}
            try:
                editor.add(
                    full_name,
                    description=description,
                    input_model=input_schema,
                    execute=self._make_executor(name),
                    permission=full_name,
                )
            except Exception:  # noqa: BLE001 - 单个工具注册失败不影响其余
                continue

    def _make_executor(self, tool_name: str):
        """构造一个工具的执行函数：转发给 ``self.call_tool`` 并归一化为 ToolResult。

        Args:
            tool_name: 服务器侧的工具名（不含命名空间前缀）。

        Returns:
            一个 ``async (input, context) -> ToolResult`` 的可调用对象。
        """

        async def _execute(input: dict[str, Any], context: ToolContext) -> ToolResult:
            try:
                result = await self.call_tool(tool_name, input or {})
            except McpError as exc:
                return ToolResult(content=f"MCP tool {tool_name} failed: {exc}", metadata={"error": True})
            # 把 MCP 返回的 content 数组拼接成一段文本，方便回灌给 LLM。
            text = self._render_tool_result(result)
            return ToolResult(content=text, metadata={"raw": result})

        return _execute

    @staticmethod
    def _render_tool_result(result: Any) -> str:
        """把 ``tools/call`` 的原始 result 渲染成一段纯文本。

        兼容常见的两种形态：\n
        * 带 ``content`` 数组（每项是 ``{type: \"text\", text: ...}``）—— 标准 MCP；\n
        * 直接是字符串 / 普通对象。
        """
        if isinstance(result, dict) and isinstance(result.get("content"), list):
            parts: list[str] = []
            for part in result["content"]:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif isinstance(part, str):
                    parts.append(part)
            if parts:
                return "\n".join(parts)
        # 回退：把整个 result 序列化为 JSON。
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)

    # ------------------------------------------------------------------ #
    # 传输层接口（子类实现）                                               #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def _send(self, message: dict[str, Any]) -> None:
        """把一个 JSON-RPC 消息写到传输层。"""

    @abstractmethod
    async def _receive_loop(self) -> None:
        """从传输层读消息并调用 ``_handle_message`` 分发（长期运行）。"""

    # -- 响应分发 -------------------------------------------------------- #

    def _handle_message(self, message: dict[str, Any]) -> None:
        """处理一条从服务器收到的 JSON-RPC 消息。

        若消息带 ``id``（响应），把它匹配到 ``_pending`` 中对应 future：\n
        * 有 ``result`` → set_result(result)；\n
        * 有 ``error`` → set_exception(McpError(...))；\n
        * 否则按协议错误处理。

        若消息是服务器主动推送的 notification（无 id，或非响应），当前简化实现
        直接忽略——学习用客户端不做订阅/推送处理。
        """
        msg_id = message.get("id")
        if msg_id is None:
            # notification / server push，忽略。
            return
        future = self._pending.pop(msg_id, None)
        if future is None or future.done():
            return
        if "error" in message and message["error"] is not None:
            err = message["error"]
            code = err.get("code") if isinstance(err, dict) else None
            text = err.get("message") if isinstance(err, dict) else str(err)
            future.set_exception(McpError(f"{self.name}: JSON-RPC error {code}: {text}"))
            return
        future.set_result(message.get("result"))

    # -- 便捷：连接并加载工具 --------------------------------------------- #

    async def connect(self) -> dict[str, Any]:
        """启动传输并完成 initialize 握手。

        Returns:
            serverInfo（含标题等信息）。
        """
        await self.start()
        return await self.initialize()

    async def close(self) -> None:
        """关闭客户端：结束接收循环、关闭传输、唤醒所有 pending 请求。"""
        # 让所有还挂着等响应的请求以错误收场，避免调用方永久阻塞。
        for future in self._pending.values():
            if not future.done():
                future.set_exception(McpError(f"{self.name}: connection closed"))
        self._pending.clear()
        await self.stop()


class McpRegistry:
    """管理多个 MCP 服务器：连接、断开、把它们的工具全部注册进 ToolEditor。

    对应原代码中把多个 MCP 连接聚合、统一管理的部分。它持有共享的
    ``ToolEditor``，这样所有 MCP 工具和本地内置工具处于同一个注册表。
    """

    def __init__(self, editor: ToolEditor) -> None:
        """构造注册表。

        Args:
            editor: 所有 MCP 工具要注册进的共享 ``ToolEditor``。
        """
        self.editor = editor
        # 服务器名 -> McpClient。
        self.clients: dict[str, McpClient] = {}

    # ------------------------------------------------------------------ #
    # 创建 / 连接                                                         #
    # ------------------------------------------------------------------ #

    async def add(self, config, name: str | None = None) -> McpClient:
        """根据配置创建一个客户端并注册到注册表（不连接）。

        Args:
            config: ``LocalMcpConfig`` 或 ``RemoteMcpConfig``。
            name: 服务器名称；缺省时用 ``config`` 的 ``name`` 或 ``url``/命令派生。

        Returns:
            新创建的 ``McpClient``（尚未 start）。
        """
        if config.disabled:
            raise McpError(f"{name or 'server'}: server is disabled")
        client_name = name or self._default_name(config)
        # 根据 type 选择对应传输子类。
        if config.type == "local":
            from .stdio import StdioMcpClient

            client = StdioMcpClient(client_name, config)
        elif config.type == "remote":
            from .http import HttpMcpClient

            client = HttpMcpClient(client_name, config)
        else:  # pragma: no cover - schema 限定 type 只能是 local/remote
            raise McpError(f"{client_name}: unknown config type {config.type!r}")
        self.clients[client_name] = client
        return client

    @staticmethod
    def _default_name(config) -> str:
        """从配置派生一个默认服务器名。"""
        if config.url:
            return config.url
        if config.command:
            return config.command[0]
        return "mcp"

    async def connect_all(self, configs: list) -> None:
        """逐个创建并连接所有配置。

        每个配置都会 ``add`` 出客户端并 ``connect``（start + initialize）。
        被禁用的配置会被跳过。

        Args:
            configs: ``McpServerConfig``（或其子类）列表，可带 ``name`` 字段。
        """
        for config in configs:
            if getattr(config, "disabled", False):
                continue
            name = getattr(config, "name", None)
            client = await self.add(config, name=name)
            try:
                await client.connect()
            except Exception as exc:  # noqa: BLE001 - 单台连接失败不阻断其余
                await client.close()

    # ------------------------------------------------------------------ #
    # 工具注册                                                            #
    # ------------------------------------------------------------------ #

    async def register_all_tools(self) -> None:
        """对每个已连接客户端：list_tools 并 register_tools。

        工具以 ``<server>_<tool>`` 命名空间形式注册进共享的 ``ToolEditor``。
        """
        for name, client in self.clients.items():
            try:
                # list_tools 会填充客户端的工具缓存，register_tools 从中读取。
                await client.list_tools()
                client.register_tools(self.editor)
            except Exception as exc:  # noqa: BLE001 - 单台失败不阻断其余
                import logging

                logging.getLogger(__name__).warning("register tools failed for %s: %s", name, exc)

    # ------------------------------------------------------------------ #
    # 关闭                                                                #
    # ------------------------------------------------------------------ #

    async def close_all(self) -> None:
        """关闭所有客户端并清空注册表。"""
        for client in self.clients.values():
            try:
                await client.close()
            except Exception:  # noqa: BLE001 - 单台关闭失败不阻断其余
                pass
        self.clients.clear()
