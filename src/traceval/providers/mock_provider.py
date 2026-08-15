"""Deterministic, no-network provider.

Used directly by unit tests, and as the judge provider in integration tests
that exercise ModelGradedScorer without ever touching a real API key.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from traceval.providers import Message, ProviderResponse, ResolvedModel

_MOCK_VERSIONS = {
    "mock-1": "mock-1-20260101",
    "mock-judge": "mock-judge-1-20260101",
}


class MockProvider:
    """Deterministic response derived from a hash of the input messages.

    `canned_response`, if set, is returned verbatim instead (useful for
    pinning a specific judge verdict in a scoring test).
    """

    def __init__(self, canned_response: str | None = None) -> None:
        self._canned_response = canned_response

    def resolve_model(self, alias: str) -> ResolvedModel:
        model_id = _MOCK_VERSIONS.get(alias, f"{alias}-mock")
        return ResolvedModel(provider="mock", model_id=model_id, alias=alias)

    def generate(
        self,
        resolved: ResolvedModel,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ProviderResponse:
        start = time.monotonic()
        content = self._canned_response or self._deterministic_reply(messages)
        latency_ms = (time.monotonic() - start) * 1000
        return ProviderResponse(
            content=content,
            input_tokens=sum(len(m.content.split()) for m in messages),
            output_tokens=len(content.split()),
            latency_ms=latency_ms,
        )

    @staticmethod
    def _deterministic_reply(messages: list[Message]) -> str:
        digest = hashlib.sha256(
            "\n".join(f"{m.role}:{m.content}" for m in messages).encode()
        ).hexdigest()[:8]
        return f"mock-response-{digest}"


__all__ = ["MockProvider"]
