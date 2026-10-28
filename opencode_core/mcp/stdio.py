"""stdio MCP 传输（对应原 mcp/stdio.ts）。

本地传输：客户端 spawn 一个子进程（如 ``npx some-mcp-server``），通过
stdin/stdout 用 newline-delimited JSON 交换 JSON-RPC 消息——每写入一行就是一个
请求/notification，每读到一行就是一个响应/服务器推送。

本实现对应原 TypeScript 中 ``McpStdio.make`` 的简化版：

* 用 ``asyncio.create_subprocess_exec`` 启动子进程；
* ``_receive_loop`` 是一个后台任务，逐行读 stdout、解析 JSON 并交给
  ``McpClient._handle_message`` 分发到对应的 pending future；
* ``_send`` 把消息序列化后写一行到 stdin（写后 flush，确保立即送达子进程）。
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from opencode_core.mcp.client import McpClient


class StdioMcpClient(McpClient):
    """通过子进程 stdio 与一个本地 MCP 服务器通信的客户端。

    对应原 ``packages/core/src/mcp/stdio.ts`` 的 ``McpStdio`` 传输。

    ``config`` 是一个 ``LocalMcpConfig``，提供：
      * ``command``：启动命令（数组，如 ``["npx", "-y", "some-server"]``）
      * ``cwd``：可选工作目录
      * ``environment``：可选额外环境变量
    """

    def __init__(self, name: str, config) -> None:
        """构造 stdio 客户端。

        Args:
            name: 服务器名称。
            config: ``LocalMcpConfig``（含 command / cwd / environment）。
        """
        super().__init__(name)
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ #
    # 生命周期                                                            #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """spawn 子进程并启动接收循环。"""
        if self._process is not None:
            return
        command = self.config.command or []
        if not command:
            raise RuntimeError(f"{self.name}: local MCP config has no command")
        # 环境变量 = 当前进程环境 + 配置声明的额外变量（对应原 extendEnv: true）。
        env = {**os.environ, **self.config.environment}
        self._process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.config.cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # 后台任务：持续读 stdout 并分发响应。
        self._reader_task = asyncio.create_task(self._receive_loop())
        # 后台任务：把 stderr 转发到日志，避免子进程写满 stderr 管道而阻塞。
        if self._process.stderr is not None:
            asyncio.create_task(self._drain_stderr(self._process.stderr))

    async def stop(self) -> None:
        """终止子进程并结束接收循环。"""
        # 先取消接收循环，避免读到一半的进程退出。
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._reader_task = None
        process = self._process
        self._process = None
        if process is None:
            return
        if process.returncode is None:
            try:
                process.terminate()
                # 给进程一个短暂优雅退出窗口，超时再强杀。
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            except ProcessLookupError:
                pass

    # ------------------------------------------------------------------ #
    # 传输层实现                                                           #
    # ------------------------------------------------------------------ #

    async def _send(self, message: dict[str, Any]) -> None:
        """把一个 JSON-RPC 消息写为一行到子进程 stdin。"""
        if self._process is None or self._process.stdin is None or self._process.returncode is not None:
            raise RuntimeError(f"{self.name}: stdio transport not connected")
        line = json.dumps(message, ensure_ascii=False) + "\n"
        self._process.stdin.write(line.encode("utf-8"))
        await self._process.stdin.drain()

    async def _receive_loop(self) -> None:
        """逐行读 stdout，解析 JSON-RPC 消息并分发。"""
        assert self._process is not None and self._process.stdout is not None
        try:
            async for raw_line in self._process.stdout:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    # 非 JSON 行（如某些服务器打印的日志）—— 忽略。
                    continue
                if isinstance(message, dict):
                    self._handle_message(message)
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001 - stdout 结束或异常都视为连接关闭
            pass

    async def _drain_stderr(self, stderr) -> None:
        """把子进程 stderr 转发到标准日志，避免阻塞。"""
        try:
            async for raw_line in stderr:
                text = raw_line.decode("utf-8", errors="replace").strip()
                if text:
                    print(f"[mcp:{self.name}] {text}", flush=True)
        except Exception:  # noqa: BLE001
            pass
