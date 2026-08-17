"""Trace schema: the append-only JSONL record of a single task run.

A trace file is one header line, N step lines, one footer line. Every field
needed to reproduce or audit a run lives here: task id/hash, resolved model
versions (never aliases), environment fingerprint, seed, the pricing rates
and traceval version that produced it, and scorer output.
`TRACE_SCHEMA_VERSION` is bumped on any breaking field change; readers
refuse to load a trace with a version they don't understand.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from traceval.providers import ResolvedModel

TRACE_SCHEMA_VERSION = 1


class AgentKind(StrEnum):
    ORACLE = "oracle"
    REPLAY = "replay"
    LIVE = "live"


class Outcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"


class ScoreResult(BaseModel):
    scorer: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class PricingSnapshot(BaseModel):
    """Per-token rates resolved from `pricing.yaml` at run time.

    Stamped into the header so a report built from a replayed trace keeps
    reporting the cost that actually applied when the run happened, even
    after `pricing.yaml` is later edited.
    """

    input_per_million_usd: float
    output_per_million_usd: float


class TraceHeader(BaseModel):
    type: Literal["header"] = "header"
    schema_version: int = TRACE_SCHEMA_VERSION
    run_id: str
    task_id: str
    task_hash: str
    task_format_version: int
    seed: int
    model_under_test: ResolvedModel
    judge_model: ResolvedModel | None = None
    agent_kind: AgentKind
    environment_fingerprint: str
    traceval_version: str
    pricing_snapshot: PricingSnapshot | None = None
    started_at: datetime


class TraceStep(BaseModel):
    type: Literal["step"] = "step"
    index: int
    observation_hash: str
    observation: dict[str, Any] = Field(default_factory=dict)
    action: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    agent_latency_ms: float = 0.0
    env_latency_ms: float = 0.0
    timestamp: datetime


class TraceTotals(BaseModel):
    input_tokens: int
    output_tokens: int
    agent_latency_ms: float
    env_latency_ms: float


class TraceError(BaseModel):
    """`scorer` is set only when this error came from a specific
    `Scorer.score()` call (its class name); `None` for a
    reset()/step()/agent.act() failure, which aborts the run before scoring
    is even attempted.
    """

    error_type: str
    error_message: str
    scorer: str | None = None


class TraceFooter(BaseModel):
    """`error` is set when reset()/step()/agent.act() raised, aborting the
    run before any scorer ran. `scorer_errors` is set when the run itself
    completed but one or more scorers raised; scorers that succeeded despite
    a sibling scorer's failure still land in `scores`. The two are mutually
    exclusive: if the run never completed, scoring was never attempted.
    """

    type: Literal["footer"] = "footer"
    outcome: Outcome
    total_steps: int
    scores: list[ScoreResult] = Field(default_factory=list)
    totals: TraceTotals
    error: TraceError | None = None
    scorer_errors: list[TraceError] = Field(default_factory=list)
    ended_at: datetime


TraceLine = TraceHeader | TraceStep | TraceFooter


class Trace(BaseModel):
    """In-memory view of a full trace: header, ordered steps, and footer."""

    header: TraceHeader
    steps: list[TraceStep] = Field(default_factory=list)
    footer: TraceFooter | None = None


__all__ = [
    "TRACE_SCHEMA_VERSION",
    "AgentKind",
    "Outcome",
    "PricingSnapshot",
    "ScoreResult",
    "Trace",
    "TraceError",
    "TraceFooter",
    "TraceHeader",
    "TraceLine",
    "TraceStep",
    "TraceTotals",
]
