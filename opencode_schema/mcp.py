"""MCP 配置 schema（对应原 packages/schema/src/mcp.ts）。

这里定义了 MCP 服务器的配置数据契约：
  * ``McpServerConfig`` —— 一个服务器的配置（local=子进程 / remote=HTTP URL）
  * ``LocalMcpConfig`` / ``RemoteMcpConfig`` —— 按 ``type`` 区分开的两种具体配置
  * ``McpServerStatus`` —— 服务器连接状态（对应原 ``Mcp.Status``）
  * ``McpServer`` —— 一个已注册的服务器实例（对应原 ``Mcp.Server``）

本模块与 `opencode_schema` 中其他模块一致：只承载纯数据类型，不含任何运行时
行为（连接、发消息等都在 `opencode_core.mcp` 中实现）。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "McpServerConfig",
    "LocalMcpConfig",
    "RemoteMcpConfig",
    "McpServerStatus",
    "McpServer",
]


class McpServerConfig(BaseModel):
    """一个 MCP 服务器配置（local=子进程 / remote=HTTP URL）。

    对应原 ``Mcp.ServerConfig``（``LocalConfig | RemoteConfig`` 的 tagged union）。
    这里用单个基类承载两种形态的公共字段，再用 ``type`` 判别：

    * ``local``   —— ``command`` 是要 spawn 的启动命令（数组），可选 ``cwd`` / ``environment``
    * ``remote``  —— ``url`` 是 streamable HTTP 端点，可选 ``headers``

    字段大多可选，因为 local 与 remote 各自只用到其中一部分。运行时创建客户端时
    根据 ``type`` 选择 ``StdioMcpClient`` 或 ``HttpMcpClient``，并只读取相关字段。
    """

    #: 服务器形态：``local`` 表示本地子进程，``remote`` 表示远程 HTTP 端点。
    type: Literal["local", "remote"]

    # -- local 专用字段 ----------------------------------------------------
    #: 启动命令（数组形式，如 ``["npx", "-y", "@modelcontextprotocol/server-filesystem"]``）。
    command: list[str] | None = None
    #: 子进程工作目录（可选；相对路径通常相对工作区目录解析）。
    cwd: str | None = None
    #: 传递给子进程的额外环境变量（可选）。
    environment: dict[str, str] = Field(default_factory=dict)

    # -- remote 专用字段 ---------------------------------------------------
    #: streamable HTTP 端点 URL。
    url: str | None = None
    #: 附加到每个请求的 HTTP 头（如认证头）。
    headers: dict[str, str] = Field(default_factory=dict)

    #: 是否禁用该服务器（禁用后不会连接、不会注册工具）。
    disabled: bool = False


class LocalMcpConfig(McpServerConfig):
    """local 形态的 MCP 配置：通过子进程启动（对应原 ``Mcp.LocalConfig``）。

    ``type`` 固定为 ``"local"``，使 ``config.type == "local"`` 在运行时成为
    选择 stdio 传输的判据。
    """

    type: Literal["local"] = "local"


class RemoteMcpConfig(McpServerConfig):
    """remote 形态的 MCP 配置：通过 streamable HTTP 访问（对应原 ``Mcp.RemoteConfig``）。

    ``type`` 固定为 ``"remote"``，使 ``config.type == "remote"`` 在运行时成为
    选择 HTTP 传输的判据。
    """

    type: Literal["remote"] = "remote"


class McpServerStatus(BaseModel):
    """服务器连接状态（对应原 ``Mcp.Status``）。

    ``status`` 取值：
      * ``connected``   —— 已连接并完成 initialize 握手
      * ``pending``     —— 正在连接中
      * ``disabled``    —— 配置里被禁用，未尝试连接
      * ``failed``      —— 连接/初始化失败
      * ``needs_auth``  —— 需要认证（OAuth）才能继续
    """

    #: 当前状态。
    status: Literal["connected", "pending", "disabled", "failed", "needs_auth"]
    #: 失败时的错误信息（仅在 failed / needs_auth 时有意义）。
    error: str | None = None


class McpServer(BaseModel):
    """一个已注册的 MCP 服务器（对应原 ``Mcp.Server``）。

    用于向外部（如 UI / CLI）描述当前已注册了哪些服务器及其连接状态。
    """

    #: 服务器名称（配置里的 key，也是注册工具的命名空间）。
    name: str
    #: 连接状态。
    status: McpServerStatus
    #: 展示标题，来自连接后 ``serverInfo.title``（未连接或未提供时为空）。
    title: str | None = None

    @property
    def is_connected(self) -> bool:
        """便捷判断：是否已连接。"""
        return self.status.status == "connected"
