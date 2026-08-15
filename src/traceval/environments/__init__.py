"""Environment protocol: a seeded, steppable, fingerprintable world an agent acts in.

`browser.py` is the first implementation (Playwright-driven). A future desktop
environment implements the same protocol and plugs into the runner without any
runner changes, since the runner only ever depends on `Environment`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from traceval.tasks import TaskFixture


class Action(BaseModel):
    kind: str  # "click" | "type" | "navigate" | "wait"
    target: str | None = None
    value: str | None = None
    url: str | None = None
    ms: int | None = None


class Observation(BaseModel):
    url: str
    title: str
    elements: dict[str, str] = Field(default_factory=dict)


class StepResult(BaseModel):
    observation: Observation
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Environment(Protocol):
    def reset(self, seed: int, fixture: TaskFixture) -> Observation: ...

    def step(self, action: Action) -> StepResult: ...

    def fingerprint(self) -> str: ...

    def close(self) -> None: ...


__all__ = ["Action", "Environment", "Observation", "StepResult"]
