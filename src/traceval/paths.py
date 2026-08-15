"""Locate a checked-in config file relative to the current working directory,
walking up parent directories until found. Shared by the model registry
(`models.yaml`) and the pricing table (`pricing.yaml`).
"""

from __future__ import annotations

from pathlib import Path


def find_repo_file(filename: str, start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        candidate_path = candidate / filename
        if candidate_path.exists():
            return candidate_path
    raise FileNotFoundError(f"{filename} not found in the current directory or any parent")


__all__ = ["find_repo_file"]
