"""ReplayAgent: replays a previously recorded trace's actions verbatim.

Backs `traceval replay <trace>`, re-running a failed run's exact action
sequence with no model involved, so the resulting trace can be diffed against
the original (or against a fresh oracle/live run) via `trace/diff.py`.
"""

from __future__ import annotations

from pathlib import Path

from traceval.agents import AgentStep
from traceval.environments import Action, Observation
from traceval.trace import TraceStep, read_trace


class ReplayAgent:
    def __init__(self, trace_path: Path) -> None:
        trace = read_trace(trace_path)
        self._actions = [Action.model_validate(step.action) for step in trace.steps]
        self.source_task_id = trace.header.task_id
        self.source_seed = trace.header.seed
        self._index = 0

    def act(self, observation: Observation, history: list[TraceStep]) -> AgentStep:
        if self._index >= len(self._actions):
            return AgentStep(action=None)
        action = self._actions[self._index]
        self._index += 1
        return AgentStep(action=action)


__all__ = ["ReplayAgent"]
