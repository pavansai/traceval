"""Real Anthropic provider.

`resolve_model` is a pure registry lookup and is safe to unit test without a
key. The API client is constructed lazily on first `generate()` call so that
constructing an `AnthropicProvider` (e.g. to resolve a model alias) never
requires `ANTHROPIC_API_KEY` to be set; only actually calling the model does.
"""

from __future__ import annotations

import os
import time
from typing import Any, cast

import anthropic

from traceval.providers import Message, ProviderResponse, ResolvedModel
from traceval.providers.registry import ModelRegistry


class AnthropicProvider:
    def __init__(self, registry: ModelRegistry, api_key: str | None = None) -> None:
        self._registry = registry
        self._api_key = api_key
        self._client: anthropic.Anthropic | None = None

    def resolve_model(self, alias: str) -> ResolvedModel:
        return self._registry.resolve("anthropic", alias)

    def generate(
        self,
        resolved: ResolvedModel,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ProviderResponse:
        client = self._client_or_create()
        start = time.monotonic()
        # `messages`/`tools` cross from our provider-agnostic domain types into the
        # Anthropic SDK's own TypedDict unions here: a deliberate, single-boundary
        # cast rather than importing SDK request types into the Provider protocol.
        response = client.messages.create(
            model=resolved.model_id,
            max_tokens=4096,
            messages=cast(Any, [{"role": m.role, "content": m.content} for m in messages]),
            tools=cast(Any, tools or []),
        )
        latency_ms = (time.monotonic() - start) * 1000

        text_blocks = [block.text for block in response.content if block.type == "text"]
        tool_calls = [
            {"id": block.id, "name": block.name, "input": block.input}
            for block in response.content
            if block.type == "tool_use"
        ]
        return ProviderResponse(
            content="\n".join(text_blocks),
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
        )

    def _client_or_create(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=self._api_key or os.environ.get("ANTHROPIC_API_KEY")
            )
        return self._client


__all__ = ["AnthropicProvider"]
