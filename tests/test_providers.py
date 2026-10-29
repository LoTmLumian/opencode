#!/usr/bin/env python3
"""Provider registry + AgentLoop provider resolution tests.

Covers: built-in provider defaults (deepseek included), config-file merging
(openai-compatible custom providers like ``local``), and that the agent loop
builds the right client for a session's provider (``deepseek`` -> OpenAI-
compatible protocol + ``DEEPSEEK_API_KEY`` + deepseek base URL).
"""

import json
import os
import tempfile
from pathlib import Path

from opencode_ai.providers.anthropic import AnthropicClient
from opencode_ai.providers.openai import OpenAIClient
from opencode_ai.providers.registry import ProviderRegistry, default_registry
from opencode_core.agent import AgentLoop
from opencode_server.server import App


def test_builtin_defaults_include_deepseek():
    reg = ProviderRegistry()
    cfg = reg.config("deepseek")
    assert cfg is not None
    assert cfg.protocol == "openai"
    assert cfg.base_url == "https://api.deepseek.com/v1"
    assert cfg.api_key_env == "DEEPSEEK_API_KEY"
    ids = reg.ids()
    assert "deepseek" in ids and "openai" in ids and "anthropic" in ids


def test_definitions_include_deepseek():
    reg = ProviderRegistry()
    models = [d.id for d in reg.definitions()]
    assert "deepseek" in models
    deepseek = next(d for d in reg.definitions() if d.id == "deepseek")
    assert "deepseek-chat" in deepseek.models


def test_from_mapping_merges_over_builtin():
    reg = ProviderRegistry.from_mapping(
        {"providers": {"deepseek": {"base_url": "https://custom.example/v1"}}}
    )
    cfg = reg.config("deepseek")
    assert cfg.base_url == "https://custom.example/v1"
    # Protocol / api key env survive the partial merge.
    assert cfg.protocol == "openai"
    assert cfg.api_key_env == "DEEPSEEK_API_KEY"


def test_from_mapping_adds_unknown_provider():
    reg = ProviderRegistry.from_mapping(
        {"providers": {"local": {"base_url": "http://localhost:11434/v1", "api_key_env": None}}}
    )
    cfg = reg.config("local")
    assert cfg is not None
    assert cfg.base_url == "http://localhost:11434/v1"
    assert cfg.api_key_env is None
    # Unknown providers get a definition so ``opencode models`` shows them.
    assert "local" in [d.id for d in reg.definitions()]


def test_config_example_file_loads():
    example = Path(__file__).resolve().parents[1] / "examples" / "config.example.json"
    data = json.loads(example.read_text(encoding="utf-8"))
    reg = ProviderRegistry.from_mapping(data)
    deepseek = reg.config("deepseek")
    assert deepseek is not None
    assert deepseek.base_url == "https://api.deepseek.com/v1"
    assert reg.config("local") is not None
    assert reg.config("anthropic") is not None


def test_build_client_selects_protocol():
    reg = ProviderRegistry()
    assert isinstance(reg.build_client("deepseek"), OpenAIClient)
    assert isinstance(reg.build_client("openai"), OpenAIClient)
    assert isinstance(reg.build_client("anthropic"), AnthropicClient)


def _config_with_deepseek_api_key(tmp_path: Path, key: str = "sk-ds-test") -> dict:
    """A config with deepseek + a DEEPSEEK_API_KEY env var set."""
    os.environ["DEEPSEEK_API_KEY"] = key
    return {"providers": {"deepseek": {"base_url": "https://api.deepseek.com/v1"}}}


def test_agent_loop_builds_deepseek_client(tmp_path):
    """AgentLoop._client_for('deepseek') yields an OpenAI-compatible client
    pointed at the deepseek base URL, using DEEPSEEK_API_KEY."""
    cfg = _config_with_deepseek_api_key(tmp_path)
    default_registry().merge(cfg)
    app = App(db_path=tmp_path / "t.db", config=cfg)

    from opencode_ai.client import LLMRequest

    client = app.loop._client_for(LLMRequest(model="deepseek-chat", provider="deepseek"))
    assert isinstance(client, OpenAIClient)
    assert "api.deepseek.com" in client._base
    assert client.config.api_key_env == "DEEPSEEK_API_KEY"
    assert os.environ.get("DEEPSEEK_API_KEY") == "sk-ds-test"
    app.db.close()


def test_agent_loop_request_provider_from_session_model(tmp_path):
    """The request built for a deepseek session carries provider='deepseek'."""
    cfg = _config_with_deepseek_api_key(tmp_path)
    app = App(db_path=tmp_path / "t.db", config=cfg)
    sid = app.store.create(title="t", model="deepseek/deepseek-chat")

    request = app.loop._build_request(sid, app.store.messages(sid))
    assert request.provider == "deepseek"
    assert request.model == "deepseek/deepseek-chat"
    app.db.close()


def test_agent_loop_respects_client_factory(tmp_path):
    """A caller-supplied client_factory still wins over the registry."""
    from opencode_ai.client import LLMClient

    class SpyClient(LLMClient):
        async def stream(self, request):
            return  # pragma: no cover
            yield  # pragma: no cover  # keep this an async generator

    factory = lambda provider: SpyClient()  # noqa: E731
    app = App(db_path=tmp_path / "t.db")
    app.loop = AgentLoop(
        store=app.store,
        tools=app.tool_service,
        bus=app.bus,
        permission=app.permission,
        client_factory=factory,
    )
    from opencode_ai.client import LLMRequest

    client = app.loop._client_for(LLMRequest(model="deepseek-chat", provider="deepseek"))
    assert isinstance(client, SpyClient)
    app.db.close()

