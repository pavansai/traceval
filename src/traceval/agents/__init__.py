"""Agent protocol: given an observation and step history, produce the next step.

`action=None` on the returned `AgentStep` signals the episode is complete (no
more actions to take). `usage` and `agent_latency_ms` travel alongside the
action so the runner can stamp real token counts and model-call latency onto
the trace, rather than the environment-only latency it used to record.
`oracle.py`, `replay.py`, and `live.py` are the three implementations; only
`live.py` requires an API key and reports nonzero usage.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from traceval.environments import Action, Observation
from traceval.trace import TraceStep


class AgentUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AgentStep(BaseModel):
    action: Action | None
    usage: AgentUsage = Field(default_factory=AgentUsage)
    agent_latency_ms: float = 0.0


@runtime_checkable
class Agent(Protocol):
    def act(self, observation: Observation, history: list[TraceStep]) -> AgentStep: ...


__all__ = ["Agent", "AgentStep", "AgentUsage"]
