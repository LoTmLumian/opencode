from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .model import ModelRef

__all__ = ["AgentDefinition", "AgentMode", "AgentPrompt"]


AgentMode = Literal["primary", "subagent"]


class AgentDefinition(BaseModel):
    """An agent configuration (mirrors `packages/schema/src/agent.ts`).

    In the original, agents are user-configurable files that define a
    system prompt, an optional model, permission overrides, and whether they
    can be spawned as subagents.
    """

    id: str
    name: str
    mode: AgentMode = "subagent"
    description: str | None = None
    # System prompt template. `{file_path}`-style placeholders are substituted.
    prompt: str = ""
    model: ModelRef | None = None
    # Permission overrides layered on top of the global config.
    permissions: dict[str, str] = Field(default_factory=dict)
    # Whether this agent is hidden from the subagent picker.
    hidden: bool = False

    def render_prompt(self, variables: dict[str, Any] | None = None) -> str:
        """Substitute `{key}` placeholders in the prompt template."""
        text = self.prompt
        for key, value in (variables or {}).items():
            text = text.replace("{" + key + "}", str(value))
        return text


class AgentPrompt(BaseModel):
    """The runtime prompt assembled for an agent (used to seed sessions)."""

    text: str
    files: list[str] = Field(default_factory=list)
    agents: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
