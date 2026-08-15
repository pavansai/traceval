"""Pinned model-version registry, loaded from models.yaml.

Bumping a pinned model version is a one-line, reviewable diff to models.yaml.
`Provider.resolve_model()` implementations look up here rather than ever
resolving a provider's own "latest" alias at runtime, so a trace's stamped
`ResolvedModel.model_id` is always the exact version that ran.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from traceval.paths import find_repo_file
from traceval.providers import ResolvedModel


class UnknownModelAliasError(KeyError):
    """Raised when an alias has no pinned entry for the given provider in models.yaml."""


def find_default_registry_path(start: Path | None = None) -> Path:
    """Walk up from `start` (default: CWD) looking for models.yaml."""
    return find_repo_file("models.yaml", start=start)


class ModelRegistry:
    def __init__(self, mapping: dict[str, dict[str, str]]) -> None:
        self._mapping = mapping

    @classmethod
    def load(cls, path: Path | None = None) -> ModelRegistry:
        registry_path = path or find_default_registry_path()
        data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        return cls(data)

    def resolve(self, provider: str, alias: str) -> ResolvedModel:
        try:
            model_id = self._mapping[provider][alias]
        except KeyError as exc:
            raise UnknownModelAliasError(
                f"no pinned version for provider={provider!r} alias={alias!r} in models.yaml"
            ) from exc
        return ResolvedModel(provider=provider, model_id=model_id, alias=alias)


__all__ = ["ModelRegistry", "UnknownModelAliasError", "find_default_registry_path"]
