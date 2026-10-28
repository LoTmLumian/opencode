# opencode-python

A Python reimplementation of [opencode](https://github.com/anomalyco/opencode) — an AI coding agent runtime. This is a **learning/teaching-oriented port**: the structure mirrors the original TypeScript monorepo, but written in idiomatic Python with the mainstream async stack (**Pydantic v2, httpx, FastAPI, SQLite, Typer**).

Each source file carries a comment saying which original file it mirrors, so you can jump between the two codebases side by side.

## Status

Alpha. Core engine implemented and covered by an offline test suite (a fake LLM drives the loop end-to-end, no network needed):

- agent loop (multi-step tool calling)
- tool system (registry + 9 built-in tools)
- multi-provider adapter (OpenAI-compatible + Anthropic)
- session persistence with crash recovery (claim/release)
- permission engine
- MCP client (stdio + streamable HTTP)
- FastAPI server (HTTP RPC + SSE)
- Typer CLI
- SDK (local runtime + remote client)

`pytest` — **56 tests, all passing**.

## Repository layout

```
opencode-python/
├── opencode_schema/     # 数据契约层  (mirrors packages/schema)
├── opencode_ai/         # LLM 适配层   (mirrors packages/ai)
├── opencode_core/       # 核心运行时  (mirrors packages/core)
├── opencode_server/     # HTTP 服务端 (mirrors packages/server)
├── opencode_sdk/        # 客户端 SDK  (mirrors packages/client + packages/sdk)
├── opencode_cli/        # 命令行入口  (mirrors packages/cli)
└── tests/               # pytest 测试
```

## Architecture

The layering matches the original monorepo's dependency direction. Everything flows through the **event bus** — the agent loop publishes, the server bridges to SSE, the SDK subscribes.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         opencode_cli (Typer)                         │
│   run / api / session / models / version   (packages/cli)            │
└───────────────┬─────────────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────────────┐
│                     opencode_server (FastAPI)                        │
│   HTTP RPC: /api/session{/id}{/prompt,/message,/tool,/model}         │
│   SSE:      /api/event          (packages/server: api.ts, routes.ts) │
└───────────────┬─────────────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────────────┐
│                     opencode_core  (agent runtime)                   │
│                                                                     │
│  AgentLoop  ──►  Bus  ◄──  server/SDK subscribe                      │
│    │   ▲            │                                                │
│    │   │            ├── PermissionService  (permission.ts)           │
│    ▼   │            ├── ToolEditor/ToolService (tool.ts)             │
│ SessionStore ◄──────┴── builtins/ (9 tools, tool/plugin/*)           │
│    │   ▲                      │                                      │
│    │   │                      └── mcp/ (stdio + streamable HTTP)     │
│    ▼   │                                                            │
│  Database (SQLite + WAL)                                            │
└───────────────┬─────────────────────────────────────────────────────┘
                │  client_factory(provider) -> LLMClient
┌───────────────▼─────────────────────────────────────────────────────┐
│                     opencode_ai  (provider adapters)                 │
│   LLMClient ──► OpenAIClient  (/chat/completions, SSE)               │
│             └─► AnthropicClient (Messages API)                       │
│   (mirrors packages/ai: llm.ts, client.ts, protocols/)               │
└───────────────┬─────────────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────────────┐
│                  opencode_schema (Pydantic contracts)                │
│   model / session / tool / permission / event / provider / mcp      │
└─────────────────────────────────────────────────────────────────────┘
```

## Package-by-package guide

### `opencode_schema` — data contracts (mirrors `packages/schema`)

Pure Pydantic models; no logic beyond validation. Every other layer depends on it.

| File | Mirrors | Contents |
|---|---|---|
| `model.py` | `schema/src/model.ts` | `ModelRef` (`providerID/modelID#variant`), content blocks, `ToolResult` |
| `session.py` | `schema/src/session.ts` | `SessionMessage` discriminated union (user/assistant/tool/system), ids |
| `tool.py` | `schema/src/tool.ts` | `ToolDefinition`, `ToolInputSchema`, `ToolContext`, `ToolResult` |
| `permission.py` | `schema/src/permission.ts` | `PermissionRule`, `PermissionRequest`, glob matching |
| `event.py` | `schema/src/event.ts` | `Event`, `EventMessage` (SSE wire format), `Snapshot` |
| `provider.py` | `schema/src/provider.ts` | `ProviderConfig`, `ProviderDefinition` |
| `agent.py` | `schema/src/agent.ts` | `AgentDefinition`, `AgentPrompt` |
| `mcp.py` | `schema/src/mcp.ts` | MCP server config models |

### `opencode_ai` — provider adapters (mirrors `packages/ai`)

The model-facing layer. A provider-agnostic request/event/response model plus concrete wire-protocol clients.

| File | Mirrors | Contents |
|---|---|---|
| `messages.py` | `ai/src/messages.ts` | `Message`, content parts, `from_openai`/`from_anthropic` |
| `client.py` | `ai/src/client.ts` | `LLMRequest`, `LLMEvent` stream, `LLMResponse`, `LLMClient` ABC |
| `llm.py` | `ai/src/llm.ts` | `generate`/`stream`/`loop` — the multi-step tool-call loop |
| `providers/base.py` | `ai/src/protocols/` | `build_client` factory, message/tool serializers |
| `providers/openai.py` | OpenAI protocol | `OpenAIClient` (SSE parsing, tool calls) |
| `providers/anthropic.py` | Anthropic protocol | `AnthropicClient` (Messages API) |

### `opencode_core` — agent runtime (mirrors `packages/core`)

| File | Mirrors | Contents |
|---|---|---|
| `agent.py` | `core/src/session.ts` + `session/runner/*` | `AgentLoop` — claim → stream → execute tools → repeat → release |
| `bus.py` | `core/src/bus.ts` | In-process event bus |
| `permission.py` | `core/src/permission.ts` | `evaluate()` (findLast), `PermissionService` (deny > ask > allow) |
| `tool.py` | `core/src/tool.ts` | `ToolEditor` registry, `ToolService` fail-safe executor |
| `builtins/` | `core/src/tool/plugin/*` | read, write, edit, glob, grep, shell, todo, webfetch, subagent |
| `database.py` | `core/src/database/database.ts` | SQLite + WAL, three tables (sessions/messages/events) |
| `session.py` | `core/src/session/store.ts` | `SessionStore` + crash-recoverable claim/release |
| `mcp/` | `core/src/mcp.ts` + `mcp/` | `McpClient` (JSON-RPC), `StdioMcpClient`, `HttpMcpClient` |

### `opencode_server` — HTTP + SSE (mirrors `packages/server`)

`App` is the composition root: it wires Database/SessionStore/ToolEditor/Bus/PermissionService/AgentLoop together and exposes them over FastAPI. `_sse_feed` gives each client an `asyncio.Queue` bridged to the bus.

### `opencode_sdk` — client (mirrors `packages/client` + `packages/sdk`)

- `OpenCodeClient` — httpx client for the server's HTTP RPC + SSE subscribe.
- `OpenCode` / `LocalRuntime` — run the agent loop in-process without a server.

### `opencode_cli` — CLI (mirrors `packages/cli`)

Typer commands: `run` (one-shot or interactive), `api` (start the server), `session list/show`, `models`, `version`.

## Key design notes

### 1. The agent loop

`AgentLoop.run` is the Python equivalent of `SessionRunner.drain` driving `SessionStep.attempt`:

1. **Claim** the session (crash recovery) — if another runner holds a live claim, back off.
2. Append the user prompt as a `UserMessage`.
3. Loop (up to `max_steps`): build an `LLMRequest` from the transcript, stream the model, persist the assistant message.
4. If the response contains tool calls, execute each (permission-gated, fail-safe) and feed results back as `ToolMessage`s, then loop again.
5. Release the claim and emit `session.idle`.

### 2. Session crash recovery

A claim writes a `time_suspended` timestamp and bumps `resume_attempts`. If the runner dies mid-flight the timestamp is never cleared; a new runner seeing a claim older than `stale_claim_ms` (60s) treats the previous owner as dead and reclaims.

### 3. Permission semantics

`evaluate(action, resource, rules)` uses **findLast** (later rules win) and defaults to `ask`. `PermissionService.assert_` escalates to the user when nothing denies; `assert_not_denied` (used by the loop) never asks and only blocks on an explicit `deny`, so tool execution keeps flowing.

### 4. MCP

`McpClient` implements the transport-agnostic JSON-RPC core: request/response reconciliation via `id -> Future`, `initialize` handshake, `tools/list`, `tools/call`. Tools register into the shared `ToolEditor` namespaced as `<server>_<tool>`. Two transports: `stdio` (newline-delimited JSON over a subprocess) and streamable HTTP (POST + SSE).

### 5. Providers

A `ProviderConfig` (id, protocol, base_url, api_key_env) resolves to an `LLMClient` via `build_client`. `protocol="openai"` covers native OpenAI **and** any OpenAI-compatible endpoint (DeepSeek, Qwen, OpenRouter, local vLLM/Ollama) — just set `base_url`. `protocol="anthropic"` uses the Messages API. The `client_factory` hook on `AgentLoop` lets you inject a client (which is how the tests run offline).

## Usage

### Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .                 # adds the `opencode` command
```

### CLI

```bash
opencode models                 # list known providers/models
opencode session list           # local sessions
opencode run "write a hello world" --model openai/gpt-4o
opencode run                    # interactive mode
opencode api --port 3456        # start the FastAPI server
```

### Server + SDK

```bash
opencode api --port 3456 &
```

```python
import asyncio
from opencode_sdk import OpenCode

async def main():
    oc = OpenCode(base_url="http://127.0.0.1:3456")   # remote mode
    s = await oc.session("demo")
    await oc.prompt(s.session_id, "hi")
    print(await s.messages())
    await oc.close()

asyncio.run(main())
```

Or run entirely locally without a server:

```python
oc = OpenCode(db_path=".data/opencode.db")   # local mode
```

### Providers

Set the API key in the environment; the key name comes from `ProviderConfig.api_key_env`:

```bash
export OPENAI_API_KEY=sk-...        # openai / openai-compatible
# or
export ANTHROPIC_API_KEY=sk-ant-... # anthropic
```

## Running the tests

```bash
pip install -e ".[dev]"   # or: pip install pytest pytest-asyncio
pytest
```

The suite is fully offline — a `ScriptedClient` (fake `LLMClient`) drives the agent loop through multi-step tool calling, provider errors, permission denials, HTTP RPC, and the SDK.

## Scope vs. the original

Implemented: core engine, tool system, multi-provider adapter, session persistence + crash recovery, permission engine, MCP client, FastAPI server (HTTP RPC + SSE), Typer CLI, SDK.

Not implemented (out of scope for this learning port): web/desktop app, full TUI, the original's Effect-based concurrency, the large provider package ecosystem, and the plugin system.

## License

MIT.
