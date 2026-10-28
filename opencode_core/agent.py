from __future__ import annotations

"""Agent loop (mirrors ``packages/core/src/session.ts`` + ``runner/*``).

This is the runtime that turns a user prompt into a completed session. It is
the Python equivalent of ``SessionRunner.drain`` driving ``SessionStep.attempt``:

1. **Claim** the session so only one runner processes it at a time.
2. Optionally append the user's prompt as a `UserMessage`.
3. Loop (up to ``max_steps``): build an `LLMRequest` from the transcript,
   **stream** the model, and persist the assistant message as it forms.
4. If the response contains tool calls, execute each one (feeding results back
   as `ToolMessage` rows) and loop again; otherwise finish.
5. **Release** the session claim and emit ``session.idle``.

Every persisted mutation (new/updated message, tool result) is also published
as an event on the `Bus` so the server/CLI layers can stream live updates.
"""

import json
from typing import Any, AsyncIterator, Callable

from opencode_ai.client import (
    LLMClient,
    LLMEvent,
    LLMRequest,
    LLMResponse,
    MessageStop,
    ProviderError,
    TextDelta,
    ToolCallDelta,
    is_provider_error,
)
from opencode_ai.messages import (
    Message,
    TextPart as AITextPart,
    ToolCallPart as AIToolCallPart,
    ToolResultPart,
)
from opencode_schema.event import Event, MessageUpdated, SessionUpdated
from opencode_schema.model import ModelRef
from opencode_schema.permission import PermissionRequest
from opencode_schema.session import (
    AssistantMessage,
    SessionMessage,
    SystemMessage,
    TextPart,
    ToolCallPart,
    ToolMessage,
    UserMessage,
    new_message_id,
    new_tool_call_id,
)
from opencode_schema.tool import ToolContext

from .bus import Bus
from .permission import PermissionService
from .session import SessionRecord, SessionStore, system_prompt
from .tool import ToolService

__all__ = ["AgentLoop"]


class AgentLoop:
    """A session-driving agent loop.

    Args:
        store: Persistence for sessions and messages.
        tools: Executes tool calls (a `ToolService` over a `ToolEditor`).
        bus: Event channel for live updates.
        permission: Permission engine (used to gate tool execution).
        client_factory: ``provider -> LLMClient``. When omitted, the provider
            is looked up via `opencode_ai.providers.build_client`.
        max_steps: Upper bound on tool-call iterations per run.
        agent: Name of the agent (recorded on assistant messages).
    """

    def __init__(
        self,
        *,
        store: SessionStore,
        tools: ToolService,
        bus: Bus,
        permission: PermissionService,
        client_factory: Callable[[str], LLMClient] | None = None,
        max_steps: int = 64,
        agent: str = "build",
    ) -> None:
        self.store = store
        self.tools = tools
        self.bus = bus
        self.permission = permission
        self.client_factory = client_factory
        self.max_steps = max_steps
        self.agent = agent
        # Cache one client per provider so HTTP transports are reused across
        # loop steps (and so a stateful client keeps its step counters).
        self._clients: dict[str, LLMClient] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, session_id: str, prompt: str | None = None) -> None:
        """Process one round of input for a session.

        Claims the session (returning immediately if another runner holds a
        live claim), appends `prompt` as a `UserMessage` if given, then runs
        the tool-call loop until the model stops calling tools or we hit
        ``max_steps``. The claim is always released in ``finally``.

        Args:
            session_id: The session to process.
            prompt: Optional user text to append before running.
        """
        if not self.store.claim(session_id):
            # Another runner is actively processing this session; back off.
            return

        try:
            # Optionally inject the user's prompt as a transcript message.
            if prompt is not None:
                user_msg = UserMessage(text=prompt)
                self.store.add_message(session_id, user_msg)
                await self._publish(MessageUpdated(session_id=session_id, payload={"message": self._msg_payload(user_msg)}))

            # Run the model until it stops calling tools or we hit the cap.
            for _step in range(self.max_steps):
                response = await self._run_step(session_id)
                if response is None:
                    # Stream ended without producing a usable response.
                    break

                tool_messages = await self._execute_tools(session_id, response)
                if not tool_messages:
                    # No tool calls were made -> the model is done.
                    break
                # Tool results are already persisted and added to the loop via
                # the next _build_request reading them from the store.

            # The run finished cleanly.
            await self._publish(
                SessionUpdated(
                    session_id=session_id,
                    payload={"id": session_id, "status": "idle"},
                )
            )
        finally:
            self.store.release(session_id)

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _build_request(self, session_id: str, messages: list[SessionMessage]) -> LLMRequest:
        """Assemble an `LLMRequest` from the session transcript.

        The request carries the default system prompt, the history converted
        into AI-layer `Message`s, and the currently registered tool
        definitions. The model/provider are resolved from the session record
        when available.
        """
        record = self.store.get(session_id)
        model = record.model if record is not None else None

        # Convert persisted session messages into AI-layer messages.
        ai_messages: list[Message] = [self._to_ai_message(m) for m in messages]

        return LLMRequest(
            model=str(model) if model is not None else "",
            provider=model.providerID if model is not None else "",
            system=[system_prompt(record)],
            messages=ai_messages,
            tools=self.tools.editor.to_llm_definitions(),
        )

    async def _run_step(self, session_id: str) -> LLMResponse | None:
        """Stream one model turn and persist the resulting assistant message.

        Returns the assembled `LLMResponse`, or ``None`` if the stream produced
        no content (e.g. a provider error or an empty response).
        """
        messages = self.store.messages(session_id)
        request = self._build_request(session_id, messages)

        client = self._client_for(request)
        events: list[LLMEvent] = []
        async for event in client.stream(request):
            events.append(event)

        if is_provider_error(events):
            # Surface the error and stop this run.
            err = next((e.message for e in events if isinstance(e, ProviderError)), "provider error")
            self._persist_assistant_error(session_id, err)
            return None

        response = LLMResponse.from_events(events)
        if response is None:
            return None

        self._persist_assistant(session_id, response)
        return response

    def _client_for(self, request: LLMRequest) -> LLMClient:
        """Return a client for the request's provider.

        A caller-supplied ``client_factory`` wins; otherwise we fall back to
        the provider factory from `opencode_ai.providers`.
        """
        if self.client_factory is not None:
            provider = request.provider or ""
            if provider not in self._clients:
                self._clients[provider] = self.client_factory(provider)
            return self._clients[provider]

        from opencode_ai.providers.base import build_client
        from opencode_schema.provider import ProviderConfig

        config = ProviderConfig(
            id=request.provider or "openai",
            protocol="openai",
            api_key_env="OPENAI_API_KEY",
        )
        return build_client(config)

    async def _execute_tools(self, session_id: str, response: LLMResponse) -> list[ToolMessage]:
        """Execute every tool call in `response` and persist the results.

        Each tool call becomes a `ToolMessage` in the transcript. Execution is
        fail-safe: any error is captured in the message rather than crashing
        the loop. Returns the list of persisted `ToolMessage`s (empty when the
        response made no tool calls).
        """
        calls = self._extract_tool_calls(response)
        results: list[ToolMessage] = []

        assistant_msg = self._last_assistant(session_id)
        message_id = assistant_msg.id if assistant_msg is not None else new_message_id()

        for call in calls:
            call_id = call.get("id") or new_tool_call_id()
            name = call.get("name") or "unknown"
            args = self._parse_args(call.get("arguments"))

            context = ToolContext(
                session_id=session_id,
                agent=self.agent,
                message_id=message_id,
                call_id=call_id,
            )

            # Permission gate before running the tool.
            allowed = await self._check_permission(session_id, name, args, message_id, call_id)
            if not allowed:
                result = ToolMessage(
                    tool_call_id=call_id,
                    tool_name=name,
                    output="Permission denied.",
                    is_error=True,
                )
            else:
                tool_result = await self.tools.execute(name, args, context)
                result = ToolMessage(
                    tool_call_id=call_id,
                    tool_name=name,
                    output=tool_result.render(),
                    is_error=bool(tool_result.metadata.get("error")),
                )

            self.store.add_message(session_id, result)
            await self._publish(
                MessageUpdated(
                    session_id=session_id,
                    payload={"message": self._msg_payload(result)},
                )
            )
            results.append(result)

        return results

    async def _check_permission(
        self,
        session_id: str,
        name: str,
        args: dict[str, Any],
        message_id: str,
        call_id: str,
    ) -> bool:
        """Ask the permission service whether a tool call may proceed.

        Never raises; a denied/blocked request simply denies the tool call so
        the loop continues.
        """
        resource = str(args.get("path") or args.get("command") or args.get("url") or "*")
        request = PermissionRequest(
            action=name,
            resource=resource,
            agent=self.agent,
            session_id=session_id,
            message_id=message_id,
        )
        try:
            # ``assert_not_denied`` allows by default and only blocks when a
            # rule explicitly denies the tool; it never prompts, so the loop
            # keeps flowing and the tool (or its placeholder) decides output.
            result = await self.permission.assert_not_denied(request)
            return result.decision != "deny"
        except Exception:  # noqa: BLE001 - permission must not crash the loop
            return False

    # ------------------------------------------------------------------
    # Assistant message persistence
    # ------------------------------------------------------------------

    def _session_model(self, session_id: str) -> ModelRef:
        """Return the session's model ref, or an empty placeholder when unset.

        An assistant message always carries a `ModelRef`; sessions created
        without one fall back to an empty ref so persistence never fails.
        """
        record = self.store.get(session_id)
        if record is not None and record.model is not None:
            return record.model
        return ModelRef(providerID="", id="")

    def _persist_assistant(self, session_id: str, response: LLMResponse) -> None:
        """Persist a completed assistant message from an `LLMResponse`."""
        model = self._session_model(session_id)

        content: list[TextPart | ToolCallPart] = []
        for part in response.message.content:
            if isinstance(part, AIToolCallPart):
                content.append(
                    ToolCallPart(
                        id=part.id,
                        name=part.name,
                        arguments=self._parse_args(part.arguments),
                    )
                )
            elif isinstance(part, AITextPart):
                content.append(TextPart(text=part.text))

        msg = AssistantMessage(
            agent=self.agent,
            model=model,
            content=content,
            finish_reason=response.finishReason,
            usage=dict(response.usage or {}),
        )
        self.store.add_message(session_id, msg)
        self.bus.publish_sync(
            MessageUpdated(session_id=session_id, payload={"message": self._msg_payload(msg)})
        )

    def _persist_assistant_error(self, session_id: str, error: str) -> None:
        """Persist an assistant message that records a provider error."""
        model = self._session_model(session_id)
        msg = AssistantMessage(
            agent=self.agent,
            model=model,
            content=[],
            error=error,
            finish_reason="error",
        )
        self.store.add_message(session_id, msg)
        self.bus.publish_sync(
            MessageUpdated(session_id=session_id, payload={"message": self._msg_payload(msg)})
        )

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _to_ai_message(self, message: SessionMessage) -> Message:
        """Convert a persisted `SessionMessage` into an AI-layer `Message`."""
        if isinstance(message, UserMessage):
            return Message.user(message.text)
        if isinstance(message, SystemMessage):
            return Message.system(message.text)
        if isinstance(message, AssistantMessage):
            content: list[Any] = []
            for part in message.content:
                # Persisted assistant messages use the schema-layer part types.
                if isinstance(part, ToolCallPart):
                    content.append(
                        AIToolCallPart(
                            id=part.id,
                            name=part.name,
                            arguments=json.dumps(part.arguments, ensure_ascii=False),
                        )
                    )
                elif isinstance(part, TextPart):
                    content.append(AITextPart(text=part.text))
            return Message.assistant(content)
        if isinstance(message, ToolMessage):
            return Message.tool(
                ToolResultPart(
                    id=message.tool_call_id,
                    name=message.tool_name,
                    result=message.output,
                    is_error=message.is_error,
                )
            )
        # Fallback: treat unknown messages as user text.
        return Message.user("")

    @staticmethod
    def _extract_tool_calls(response: LLMResponse) -> list[dict[str, Any]]:
        """Collect tool calls from a response (toolCalls or message content)."""
        calls: list[dict[str, Any]] = list(response.toolCalls or [])
        if not calls:
            for part in response.message.content:
                if isinstance(part, AIToolCallPart):
                    calls.append(
                        {
                            "id": part.id,
                            "name": part.name,
                            "arguments": part.arguments,
                        }
                    )
        return calls

    @staticmethod
    def _parse_args(arguments: Any) -> dict[str, Any]:
        """Parse a tool-call arguments value (JSON string or dict) into a dict."""
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                return json.loads(arguments or "{}")
            except json.JSONDecodeError:
                return {}
        return {}

    def _last_assistant(self, session_id: str) -> AssistantMessage | None:
        """Return the most recent assistant message in the transcript, if any."""
        for message in reversed(self.store.messages(session_id)):
            if isinstance(message, AssistantMessage):
                return message
        return None

    @staticmethod
    def _msg_payload(message: SessionMessage) -> dict[str, Any]:
        """Render a session message into an event payload dict."""
        return message.model_dump()

    async def _publish(self, event: Event) -> None:
        """Publish an event and await its subscribers."""
        await self.bus.publish(event)
