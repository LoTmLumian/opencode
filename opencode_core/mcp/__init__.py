"""MCP 客户端（对应原 packages/core/src/mcp）。

MCP (Model Context Protocol) 是给 LLM 提供工具 / 资源 / 提示词的标准协议，
客户端通过 JSON-RPC 2.0 与服务器通信。本包支持两种传输：

* ``stdio`` —— 本地传输：spawn 一个子进程（如 ``npx some-mcp-server``），
  通过 stdin/stdout 用 newline-delimited JSON 交换 JSON-RPC 消息。
* ``streamable HTTP`` —— 远程传输：通过 HTTP POST /message 发送 JSON-RPC，
  用 SSE 接收响应。

包结构：

* ``client.py`` —— ``McpClient`` 抽象基类（JSON-RPC 公共逻辑 + MCP 协议方法）
  与 ``McpRegistry`` 注册表（管理多个服务器并把它们的工具注册进 ``ToolEditor``）。
* ``stdio.py`` —— ``StdioMcpClient``：子进程传输。
* ``http.py``   —— ``HttpMcpClient``：streamable HTTP 传输（简化版）。
"""
from .client import McpClient, McpError, McpRegistry
from .stdio import StdioMcpClient
from .http import HttpMcpClient

__all__ = ["McpClient", "McpError", "McpRegistry", "StdioMcpClient", "HttpMcpClient"]
