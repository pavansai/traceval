"""Task schema: a task.yaml file plus the fixture files it references.

`TASK_FORMAT_VERSION` is bumped on breaking schema changes and stamped into
every trace header, so an old trace stays interpretable even after the task
format evolves. Version 2 clarifies what `environment.config.observe_selectors`
(browser environment) means: every element the agent needs to see to decide
what to do, i.e. the same information a `live` agent's observation is built
from, not just the elements a scorer happens to check afterward. `load_task`
enforces the load-bearing half of that contract: an `exact_match` scorer's
`target` must be a selector the agent was actually shown, or the task is
structurally impossible to complete and fails to load rather than silently
scoring `None` against `expected` forever.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

TASK_FORMAT_VERSION = 2


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


class TaskValidationError(ValueError):
    """Raised when a loaded task is structurally impossible to complete."""


class TaskFixture(BaseModel):
    """Resolved fixture file paths for a task, relative to the task directory."""

    task_dir: Path
    files: dict[str, Path] = Field(default_factory=dict)

    def path(self, name: str) -> Path:
        return self.files[name]


def load_task(task_dir: Path) -> Task:
    task_yaml_path = task_dir / "task.yaml"
    data = yaml.safe_load(task_yaml_path.read_text(encoding="utf-8"))
    task = Task(
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
    _validate_scorer_selectors_are_observed(task)
    return task


def _validate_scorer_selectors_are_observed(task: Task) -> None:
    """An `exact_match` target the agent was never shown can never match.

    Only the browser environment has an `observe_selectors` concept; other
    environment kinds are left alone. Only `exact_match` reads a selector
    out of the observation at scoring time -- `rubric`'s `target` is an
    *action* target (what the agent clicked/typed into), checked against the
    trace's action history, not a value read from an observation, so it
    isn't part of this contract.
    """
    if task.environment.kind != "browser":
        return
    observe_selectors = set(task.environment.config.get("observe_selectors", []))
    for scorer in task.scorers:
        if scorer.kind != "exact_match":
            continue
        target = scorer.config.get("target")
        if target is not None and target not in observe_selectors:
            raise TaskValidationError(
                f"task {task.id!r}: exact_match scorer targets {target!r}, which is "
                f"not in observe_selectors {sorted(observe_selectors)}. The agent was "
                "never shown this element, so scoring can never succeed. Add it to "
                "environment.config.observe_selectors."
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
    "TaskValidationError",
    "build_fixture",
    "load_task",
]
