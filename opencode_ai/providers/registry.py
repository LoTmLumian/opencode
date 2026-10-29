from __future__ import annotations

"""Provider 注册表（对应原项目按 provider 包注册的方式，简化为一张配置表）。

统一管理：
- 内置默认 provider（openai / anthropic / deepseek 等）
- 从配置文件（JSON，见 ``examples/config.example.json``）覆盖/新增 provider
- 按 provider id 解析出 `ProviderConfig` 供 `build_client` 使用
"""

import json
from pathlib import Path
from typing import Any, Iterable

from opencode_schema.provider import ProviderConfig, ProviderDefinition

__all__ = ["ProviderRegistry", "default_registry"]

# 内置登记信息（用于 ``opencode models`` 展示）。
_DEFAULT_DEFINITIONS: list[ProviderDefinition] = [
    ProviderDefinition(
        id="openai",
        name="OpenAI",
        protocol="openai",
        default_base_url="https://api.openai.com/v1",
        models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    ),
    ProviderDefinition(
        id="anthropic",
        name="Anthropic",
        protocol="anthropic",
        default_base_url="https://api.anthropic.com/v1",
        models=["claude-sonnet-4", "claude-opus-4", "claude-haiku-3"],
    ),
    ProviderDefinition(
        id="deepseek",
        name="DeepSeek",
        protocol="openai",
        default_base_url="https://api.deepseek.com/v1",
        models=["deepseek-chat", "deepseek-reasoner"],
    ),
]

# 内置默认 ProviderConfig（无配置文件时也能用）。
_DEFAULT_CONFIGS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        id="openai",
        protocol="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),
    "anthropic": ProviderConfig(
        id="anthropic",
        protocol="anthropic",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "deepseek": ProviderConfig(
        id="deepseek",
        protocol="openai",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
    ),
}


class ProviderRegistry:
    """A mutable table of provider definitions and configs.

    Lookups fall back to the built-in defaults, so a registry created with no
    config still resolves ``openai`` / ``anthropic`` / ``deepseek``.
    """

    def __init__(self) -> None:
        self._configs: dict[str, ProviderConfig] = dict(_DEFAULT_CONFIGS)
        self._definitions: dict[str, ProviderDefinition] = {
            d.id: d for d in _DEFAULT_DEFINITIONS
        }

    # -- config file loading ---------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "ProviderRegistry":
        """Build a registry from a JSON config file (``examples/config.example.json``)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ProviderRegistry":
        """Build a fresh registry from a parsed config dict."""
        return cls().merge(data)

    def merge(self, data: dict[str, Any]) -> "ProviderRegistry":
        """Merge a parsed config dict into this registry (in place, returns self).

        Recognised keys: ``providers`` (id -> config dict). Every provider entry
        is merged over its built-in default, so a partial entry like
        ``{"deepseek": {"api_key_env": "DEEPSEEK_KEY"}}`` keeps the default
        base URL/protocol. Unknown providers (e.g. ``local``/``ollama``) are
        added as new entries, which makes them resolvable by the client factory
        and visible to ``opencode models``.
        """
        providers = data.get("providers") or {}
        if isinstance(providers, dict):
            for pid, raw in providers.items():
                if not isinstance(raw, dict):
                    continue
                cfg = self.upsert(pid, raw)
                # Register a matching definition so the provider shows up in
                # ``opencode models`` even when it has no built-in entry.
                if pid not in self._definitions:
                    self._definitions[pid] = ProviderDefinition(
                        id=pid,
                        name=raw.get("name") or pid,
                        protocol=cfg.protocol,
                        default_base_url=cfg.base_url or "",
                        models=list(raw.get("models") or []),
                    )
        return self

    # -- mutation ---------------------------------------------------------

    def upsert(self, pid: str, patch: dict[str, Any] | ProviderConfig) -> ProviderConfig:
        """Register (or merge over) a provider config."""
        if isinstance(patch, ProviderConfig):
            config = patch
        else:
            base = self._configs.get(pid)
            merged = dict(base.model_dump() if base is not None else {"id": pid})
            merged.update({k: v for k, v in patch.items() if v is not None})
            config = ProviderConfig(**merged)
        self._configs[config.id] = config
        return config

    def add_definition(self, definition: ProviderDefinition) -> None:
        """Register a static provider definition (used by ``opencode models``)."""
        self._definitions[definition.id] = definition

    # -- lookups ----------------------------------------------------------

    def config(self, pid: str) -> ProviderConfig | None:
        """Resolve a provider config by id, or ``None`` if unknown."""
        return self._configs.get(pid)

    def config_or_default(self, pid: str) -> ProviderConfig:
        """Resolve a provider config, falling back to an OpenAI-compatible default."""
        config = self._configs.get(pid)
        if config is not None:
            return config
        return ProviderConfig(
            id=pid or "openai",
            protocol="openai",
            api_key_env=None,
        )

    def definitions(self) -> list[ProviderDefinition]:
        """All known provider definitions (built-ins + any added), sorted by id."""
        return sorted(self._definitions.values(), key=lambda d: d.id)

    def ids(self) -> list[str]:
        """All configured provider ids, sorted."""
        return sorted(self._configs)

    # -- client construction ---------------------------------------------

    def build_client(self, pid: str, http: Any = None):
        """Build an `LLMClient` for a provider id (via ``build_client``)."""
        from .base import build_client

        return build_client(self.config_or_default(pid), http=http)


_DEFAULT_REGISTRY = ProviderRegistry()


def default_registry() -> ProviderRegistry:
    """The shared process-wide registry (built-ins merged with config)."""
    return _DEFAULT_REGISTRY

