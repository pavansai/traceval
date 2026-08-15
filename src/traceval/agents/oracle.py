"""OracleAgent: replays a task's scripted_trajectory.jsonl.

Proves the environment/runner/scoring pipeline works without any model: the
deterministic backbone that keeps the whole test suite key-free. Each script
line may carry an `expect` block describing the observation state expected
before that action runs; a mismatch raises loudly, so a scripted trajectory
stays an honest fixture rather than a blind replay.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traceval.agents import AgentStep
from traceval.environments import Action, Observation
from traceval.trace import TraceStep


class ScriptedTrajectoryDivergedError(RuntimeError):
    """Raised when the live observation doesn't match a script line's `expect` block."""


class OracleAgent:
    def __init__(self, script_path: Path) -> None:
        self._entries: list[dict[str, Any]] = [
            json.loads(line)
            for line in script_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._index = 0

    def act(self, observation: Observation, history: list[TraceStep]) -> AgentStep:
        if self._index >= len(self._entries):
            return AgentStep(action=None)
        entry = self._entries[self._index]
        self._index += 1
        expect = entry.get("expect")
        if expect is not None:
            self._check_expectation(observation, expect, step_index=self._index - 1)
        return AgentStep(action=Action.model_validate(entry["action"]))

    @staticmethod
    def _check_expectation(
        observation: Observation, expect: dict[str, Any], *, step_index: int
    ) -> None:
        for selector, expected_value in expect.get("elements", {}).items():
            actual = observation.elements.get(selector)
            if actual != expected_value:
                raise ScriptedTrajectoryDivergedError(
                    f"script step {step_index}: expected element {selector!r} == "
                    f"{expected_value!r}, got {actual!r}"
                )


__all__ = ["OracleAgent", "ScriptedTrajectoryDivergedError"]
