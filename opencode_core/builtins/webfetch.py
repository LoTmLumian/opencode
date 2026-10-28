from __future__ import annotations

"""Fetch a web page (mirrors `tool/plugin/webfetch.ts`)."""

from typing import Any

import httpx
from pydantic import BaseModel, Field

from opencode_core.tool import ToolEditor
from opencode_schema.tool import ToolCallError, ToolContext, ToolResult


class WebFetchInput(BaseModel):
    """Parameters for the `webfetch` tool."""

    url: str = Field(..., description="URL to fetch.")
    method: str = Field("GET", description="HTTP method to use (e.g. GET, POST).")


async def execute_webfetch(input: dict[str, Any], context: ToolContext) -> ToolResult:
    """Fetch `input["url"]` and return the response text.

    Network access may be unavailable in some environments; in that case the
    error is surfaced as a `ToolCallError` rather than crashing the loop.
    """
    req = WebFetchInput.model_validate(input)
    method = req.method.upper()

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.request(method, req.url)
    except httpx.HTTPError as exc:
        raise ToolCallError(f"Request failed for {req.url}: {exc}") from exc

    try:
        text = response.text
    except Exception as exc:  # noqa: BLE001 - response may not be decodable
        text = f"(unable to decode response body: {exc})"

    return ToolResult.text(
        text,
        url=str(response.url),
        status_code=response.status_code,
    )


def register(editor: ToolEditor) -> None:
    """Register the `webfetch` tool with the editor."""
    editor.add(
        "webfetch",
        "Fetch a web page and return its text content.",
        WebFetchInput,
        execute=execute_webfetch,
    )
