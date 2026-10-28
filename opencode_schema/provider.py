from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = ["ProviderConfig", "ProviderDefinition", "AuthType"]


AuthType = Literal["api_key", "oauth", "none"]


class ProviderConfig(BaseModel):
    """A configured provider (mirrors `packages/schema/src/provider.ts`).

    The original ships many provider packages; here a provider is a small
    config object: an id, a base URL, an auth mechanism, and the protocol
    family it speaks (``openai``-compatible or ``anthropic``).
    """

    id: str
    # e.g. "openai", "anthropic", "openai-compatible" (for any base URL).
    protocol: Literal["openai", "anthropic"] = "openai"
    base_url: str | None = None
    api_key_env: str | None = None
    auth: AuthType = "api_key"
    # Arbitrary provider-specific options (models, headers, etc).
    options: dict[str, Any] = Field(default_factory=dict)

    def api_key(self) -> str | None:
        """Resolve the API key from the environment."""
        if not self.api_key_env:
            return None
        import os

        return os.environ.get(self.api_key_env)


class ProviderDefinition(BaseModel):
    """Static registry entry describing what a provider supports."""

    id: str
    name: str
    protocol: Literal["openai", "anthropic"]
    default_base_url: str
    # Known model IDs this provider offers (used by the models tool).
    models: list[str] = Field(default_factory=list)
