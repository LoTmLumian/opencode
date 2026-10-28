from __future__ import annotations

"""Provider protocol implementations.

Each module implements the OpenAI-compatible or Anthropic wire protocol on
top of `httpx`. The `build_client` factory selects the right implementation
for a given `ProviderConfig`.
"""

from .base import ProtocolError, build_client, make_client

__all__ = ["ProtocolError", "build_client", "make_client"]
