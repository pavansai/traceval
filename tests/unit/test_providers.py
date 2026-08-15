from pathlib import Path

import pytest

from traceval.providers import Message
from traceval.providers.anthropic_provider import AnthropicProvider
from traceval.providers.mock_provider import MockProvider
from traceval.providers.openai_provider import OpenAIProvider
from traceval.providers.registry import (
    ModelRegistry,
    UnknownModelAliasError,
    find_default_registry_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_find_default_registry_path_locates_models_yaml() -> None:
    path = find_default_registry_path(start=REPO_ROOT / "src" / "traceval")
    assert path == REPO_ROOT / "models.yaml"


def test_registry_resolves_pinned_alias() -> None:
    registry = ModelRegistry.load(REPO_ROOT / "models.yaml")
    resolved = registry.resolve("anthropic", "claude-sonnet-5")
    assert resolved.provider == "anthropic"
    assert resolved.alias == "claude-sonnet-5"
    assert resolved.model_id == "claude-sonnet-5"


def test_registry_resolves_alias_to_distinct_dated_model_id() -> None:
    # Haiku 4.5 publishes a distinct dated model id, unlike Sonnet 5 / Opus 5.
    # This is the case the registry exists to capture.
    registry = ModelRegistry.load(REPO_ROOT / "models.yaml")
    resolved = registry.resolve("anthropic", "claude-haiku-4-5")
    assert resolved.model_id == "claude-haiku-4-5-20251001"
    assert resolved.model_id != resolved.alias


def test_registry_unknown_alias_raises() -> None:
    registry = ModelRegistry.load(REPO_ROOT / "models.yaml")
    with pytest.raises(UnknownModelAliasError):
        registry.resolve("anthropic", "does-not-exist")


def test_mock_provider_resolve_model_no_key_needed() -> None:
    provider = MockProvider()
    resolved = provider.resolve_model("mock-1")
    assert resolved.provider == "mock"
    assert resolved.model_id == "mock-1-20260101"


def test_mock_provider_generate_is_deterministic() -> None:
    provider = MockProvider()
    resolved = provider.resolve_model("mock-1")
    messages = [Message(role="user", content="what is 2+2")]
    r1 = provider.generate(resolved, messages)
    r2 = provider.generate(resolved, messages)
    assert r1.content == r2.content
    assert r1.input_tokens == r2.input_tokens == 3


def test_mock_provider_canned_response() -> None:
    provider = MockProvider(canned_response="pinned answer")
    resolved = provider.resolve_model("mock-judge")
    response = provider.generate(resolved, [Message(role="user", content="grade this")])
    assert response.content == "pinned answer"


def test_anthropic_provider_resolve_model_requires_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    registry = ModelRegistry.load(REPO_ROOT / "models.yaml")
    provider = AnthropicProvider(registry)
    resolved = provider.resolve_model("claude-sonnet-5")
    assert resolved.model_id == "claude-sonnet-5"


def test_openai_provider_resolve_model_works_generate_not_implemented() -> None:
    registry = ModelRegistry.load(REPO_ROOT / "models.yaml")
    provider = OpenAIProvider(registry)
    resolved = provider.resolve_model("gpt-5")
    assert resolved.model_id == "gpt-5"
    with pytest.raises(NotImplementedError):
        provider.generate(resolved, [Message(role="user", content="hi")])
