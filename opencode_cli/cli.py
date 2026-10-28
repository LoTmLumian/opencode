"""opencode CLI（Typer 版，对应原 packages/cli）。

一个实用的命令行入口，支持一次性 run、交互式 run、启动 FastAPI server、
查看本地会话、列出模型/provider 等。教学向：保持简单，不做完整 TUI。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer

from opencode_server.server import App

__all__ = ["app", "main", "default_db_path", "interactive"]

app = typer.Typer(
    help="opencode-python — an AI coding agent",
    no_args_is_help=True,
)
session_app = typer.Typer(help="Manage sessions")
app.add_typer(session_app, name="session")


# ---------------------------------------------------------------------------
# 已知 provider / model 的静态登记（教学用途：opencode 的 models 命令会读取
# 实际注册的 provider 包；这里给出一个简化的内置列表，便于演示）。
# ---------------------------------------------------------------------------
_KNOWN_PROVIDERS: list[tuple[str, str, list[str]]] = [
    ("openai", "OpenAI", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]),
    ("anthropic", "Anthropic", ["claude-sonnet-4", "claude-opus-4", "claude-haiku-3"]),
]


def default_db_path() -> Path:
    """默认数据目录下的 SQLite 文件。"""
    return Path.home() / ".opencode-python" / "opencode.db"


# ---------------------------------------------------------------------------
# run —— 一次性 / 交互模式
# ---------------------------------------------------------------------------


def _build_app(db: str | None, config: dict | None = None) -> App:
    """构造本地 App。db 未指定时使用默认数据目录。"""
    db_path = Path(db) if db else default_db_path()
    return App(db_path=db_path, config=config)


def _prompt_from_stdin() -> str:
    """从 stdin 读取一次性 prompt（管道输入）。"""
    return sys.stdin.read().strip()


def _assistant_texts(messages: list) -> list[str]:
    """从一轮会话消息中提取所有 assistant 文本回复。"""
    texts: list[str] = []
    for msg in messages:
        if getattr(msg, "type", None) != "assistant":
            continue
        for part in getattr(msg, "content", []) or []:
            if getattr(part, "type", None) == "text" and getattr(part, "text", ""):
                texts.append(part.text)
        if getattr(msg, "error", None):
            texts.append(f"[error] {msg.error}")
    return texts


@app.command()
def run(
    prompt: str = typer.Argument(None, help="Prompt to run. If omitted, read from stdin."),
    model: str = typer.Option(None, "--model", "-m", help="Model to use, e.g. openai/gpt-4o"),
    agent: str = typer.Option("build", "--agent", "-a", help="Agent to use"),
    db: str = typer.Option(None, "--db", help="Database path"),
) -> None:
    """Run the agent once with a prompt (or interactively if no prompt)."""
    # 无参数：优先读 stdin 管道；stdin 非 TTY 且非空则作为一次性 prompt，
    # 否则进入交互模式。
    if prompt is None and not sys.stdin.isatty():
        piped = _prompt_from_stdin()
        if piped:
            prompt = piped

    app_obj = _build_app(db)
    try:
        if prompt is None:
            asyncio.run(interactive(app_obj))
            return

        sid = app_obj.store.create(title=prompt[:60], agent=agent, model=model)
        try:
            asyncio.run(app_obj.loop.run(sid, prompt))
        except Exception as exc:  # noqa: BLE001 - 运行失败打印错误，不崩栈
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1)

        for text in _assistant_texts(app_obj.store.messages(sid)):
            typer.echo(text)
    finally:
        app_obj.db.close()


async def interactive(app_obj: App) -> None:
    """简单的交互式 agent 循环。

    ``>>> `` 提示符读用户输入，调用 AgentLoop，打印该轮 assistant 回复。
    教学向：不做完整 TUI。
    """
    sid = app_obj.store.create(title="interactive")
    typer.echo("opencode-python interactive mode. Type 'exit' to quit.")
    try:
        while True:
            try:
                text = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not text.strip():
                continue
            if text.strip().lower() in ("exit", "quit"):
                break
            try:
                await app_obj.loop.run(sid, text)
            except Exception as exc:  # noqa: BLE001 - 打印错误，继续交互
                typer.echo(f"error: {exc}", err=True)
                continue
            # 打印该轮最新 assistant 消息。
            for msg_text in _assistant_texts(app_obj.store.messages(sid)):
                typer.echo(msg_text)
    finally:
        app_obj.db.close()


# ---------------------------------------------------------------------------
# api —— 启动 FastAPI server
# ---------------------------------------------------------------------------


@app.command()
def api(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(3456, "--port", "-p"),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Start the FastAPI server."""
    app_obj = _build_app(db)
    # uvicorn 前台运行（阻塞直到 Ctrl-C）。
    app_obj.start(host=host, port=port)


# ---------------------------------------------------------------------------
# session —— 查看本地会话
# ---------------------------------------------------------------------------


@session_app.command("list")
def session_list(
    db: str = typer.Option(None, "--db"),
    limit: int = typer.Option(50, "--limit", help="Max sessions to show"),
) -> None:
    """List local sessions (newest first)."""
    app_obj = _build_app(db)
    try:
        summaries = app_obj.store.list(limit=limit)
        if not summaries:
            typer.echo("No sessions.")
            return
        for s in summaries:
            typer.echo(f"{s.id}\t{s.title or ''}\t{s.updated:.0f}")
    finally:
        app_obj.db.close()


@session_app.command("show")
def session_show(
    session_id: str,
    db: str = typer.Option(None, "--db"),
) -> None:
    """Show a session's metadata and messages."""
    app_obj = _build_app(db)
    try:
        record = app_obj.store.get(session_id)
        if record is None:
            typer.echo(f"session not found: {session_id}", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"id:      {record.id}")
        typer.echo(f"title:   {record.title or ''}")
        typer.echo(f"agent:   {record.agent or ''}")
        typer.echo(f"model:   {record.model_str or ''}")
        typer.echo(f"created: {record.created:.0f}")
        typer.echo(f"updated: {record.updated:.0f}")
        typer.echo("messages:")
        for msg in app_obj.store.messages(session_id):
            typer.echo(_message_summary(msg))
    finally:
        app_obj.db.close()


def _message_summary(msg) -> str:
    """把一条会话消息渲染成单行摘要。"""
    mtype = getattr(msg, "type", "?")
    if mtype == "user":
        return f"  user      : {getattr(msg, 'text', '')}"
    if mtype == "assistant":
        texts = [
            p.text
            for p in getattr(msg, "content", []) or []
            if getattr(p, "type", None) == "text" and getattr(p, "text", "")
        ]
        body = " ".join(texts) or ("[error] " + getattr(msg, "error", "") if getattr(msg, "error", "") else "[no text]")
        return f"  assistant : {body}"
    if mtype == "tool":
        return f"  tool      : {getattr(msg, 'tool_name', '')} -> {getattr(msg, 'output', '')}"
    if mtype == "system":
        return f"  system    : {getattr(msg, 'text', '')}"
    return f"  {mtype:<10}: (message)"


# ---------------------------------------------------------------------------
# models —— 列出可用 provider / model
# ---------------------------------------------------------------------------


@app.command()
def models() -> None:
    """List known providers/models."""
    for pid, name, model_ids in _KNOWN_PROVIDERS:
        typer.echo(f"{pid} ({name}):")
        for mid in model_ids:
            typer.echo(f"  {pid}/{mid}")


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print version."""
    typer.echo("opencode-python 0.1.0")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
