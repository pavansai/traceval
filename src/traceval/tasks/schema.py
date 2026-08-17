"""Task schema: a task.yaml file plus the fixture files it references.

`TASK_FORMAT_VERSION` is bumped on breaking schema changes and stamped into
every trace header, so an old trace stays interpretable even after the task
format evolves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

TASK_FORMAT_VERSION = 1


class EnvironmentConfig(BaseModel):
    kind: str  # e.g. "browser"; a future "desktop" env plugs in the same way
    config: dict[str, Any] = Field(default_factory=dict)


class ScorerConfig(BaseModel):
    kind: str  # "exact_match" | "rubric" | "model_graded"
    config: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    id: str
    format_version: int = TASK_FORMAT_VERSION
    seed: int
    environment: EnvironmentConfig
    scorers: list[ScorerConfig] = Field(default_factory=list)
    max_steps: int = 20
    fixture_files: list[str] = Field(default_factory=list)
    expected: Any = None
    # A canned mock judge response can only ever rubber-stamp a verdict, not
    # actually judge anything, so a task whose model_graded scorer needs a
    # real judge to mean anything sets this rather than faking a pass. The
    # CLI skips such tasks (not errors them) when only a mock judge is
    # configured.
    requires_live_judge: bool = False
    task_dir: Path


class TaskFixture(BaseModel):
    """Resolved fixture file paths for a task, relative to the task directory."""

    task_dir: Path
    files: dict[str, Path] = Field(default_factory=dict)

    def path(self, name: str) -> Path:
        return self.files[name]


def load_task(task_dir: Path) -> Task:
    task_yaml_path = task_dir / "task.yaml"
    data = yaml.safe_load(task_yaml_path.read_text(encoding="utf-8"))
    return Task(
        id=data["id"],
        format_version=data.get("format_version", TASK_FORMAT_VERSION),
        seed=data["seed"],
        environment=EnvironmentConfig(**data["environment"]),
        scorers=[ScorerConfig(**s) for s in data.get("scorers", [])],
        max_steps=data.get("max_steps", 20),
        fixture_files=data.get("fixture_files", []),
        expected=data.get("expected"),
        requires_live_judge=data.get("requires_live_judge", False),
        task_dir=task_dir,
    )


def build_fixture(task: Task) -> TaskFixture:
    files = {name: task.task_dir / name for name in task.fixture_files}
    return TaskFixture(task_dir=task.task_dir, files=files)


__all__ = [
    "TASK_FORMAT_VERSION",
    "EnvironmentConfig",
    "ScorerConfig",
    "Task",
    "TaskFixture",
    "build_fixture",
    "load_task",
]
