"""Provider protocol: resolves model aliases to pinned versions and generates responses.

Concrete providers live in sibling modules (anthropic_provider, openai_provider,
mock_provider). Anything that calls a model implements `Provider`; the runner and
scorers only ever depend on this protocol, never on a concrete SDK.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ResolvedModel(BaseModel):
    """A model alias resolved to its exact provider + version string.

    Traces stamp this, never the alias, so a run stays reproducible even after
    `models.yaml` repoints an alias at a newer pinned version.
    """

    provider: str
    model_id: str
    alias: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ProviderResponse(BaseModel):
    content: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    input_tokens: int
    output_tokens: int
    latency_ms: float


@runtime_checkable
class Provider(Protocol):
    """A model provider: resolves aliases to pinned versions and generates responses."""

    def resolve_model(self, alias: str) -> ResolvedModel: ...

    def generate(
        self,
        resolved: ResolvedModel,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ProviderResponse: ...


__all__ = ["Message", "Provider", "ProviderResponse", "ResolvedModel"]
