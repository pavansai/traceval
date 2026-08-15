"""OpenAI provider stub.

`resolve_model` works against the pinned registry today, proving the
`Provider` protocol has more than one implementor. `generate` is intentionally
unimplemented until OpenAI support is actually needed, since there's no point
taking on the SDK dependency and a second code path to test before there's a
real caller.
"""

from __future__ import annotations

from typing import Any

from traceval.providers import Message, ProviderResponse, ResolvedModel
from traceval.providers.registry import ModelRegistry


class OpenAIProvider:
    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def resolve_model(self, alias: str) -> ResolvedModel:
        return self._registry.resolve("openai", alias)

    def generate(
        self,
        resolved: ResolvedModel,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ProviderResponse:
        raise NotImplementedError(
            "OpenAIProvider.generate is not implemented yet. resolve_model() already "
            "works against the pinned registry, so wiring up generation later won't "
            "require touching the runner, trace, or scoring layers."
        )


__all__ = ["OpenAIProvider"]
