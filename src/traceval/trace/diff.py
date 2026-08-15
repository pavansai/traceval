"""Structured diff between two traces.

Backs `traceval diff`: replay a failed run, then diff it against the last
passing trace to see exactly where the two trajectories first diverge, plus
the resulting score/latency/cost deltas.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from traceval.providers import ResolvedModel
from traceval.trace.schema import Trace


class HeaderDelta(BaseModel):
    """Whether the two traces ran against the same model/environment/task.

    A step-0 divergence caused by a changed model version or environment
    fingerprint looks identical, at the step level, to an actual agent
    regression, so this is computed and surfaced *before* `first_divergence`
    so that explanation isn't missed.
    """

    model_under_test_a: ResolvedModel
    model_under_test_b: ResolvedModel
    model_under_test_changed: bool
    judge_model_a: ResolvedModel | None
    judge_model_b: ResolvedModel | None
    judge_model_changed: bool
    environment_fingerprint_a: str
    environment_fingerprint_b: str
    environment_fingerprint_changed: bool
    task_hash_a: str
    task_hash_b: str
    task_hash_changed: bool

    @property
    def has_changes(self) -> bool:
        return (
            self.model_under_test_changed
            or self.judge_model_changed
            or self.environment_fingerprint_changed
            or self.task_hash_changed
        )


class StepDivergence(BaseModel):
    index: int
    action_a: dict[str, Any] | None
    action_b: dict[str, Any] | None
    observation_hash_a: str | None
    observation_hash_b: str | None
    observation_a: dict[str, Any] | None
    observation_b: dict[str, Any] | None


class ScoreDelta(BaseModel):
    scorer: str
    score_a: float | None
    score_b: float | None
    passed_a: bool | None
    passed_b: bool | None


class TraceDiff(BaseModel):
    trace_a_path: str
    trace_b_path: str
    same_task: bool
    same_seed: bool
    header_delta: HeaderDelta
    first_divergence: StepDivergence | None
    step_count_a: int
    step_count_b: int
    score_deltas: list[ScoreDelta] = Field(default_factory=list)
    agent_latency_delta_ms: float
    env_latency_delta_ms: float
    input_tokens_delta: int
    output_tokens_delta: int

    @property
    def diverged(self) -> bool:
        return self.first_divergence is not None


def diff_traces(a: Trace, b: Trace, a_path: str = "", b_path: str = "") -> TraceDiff:
    first_divergence = _first_divergence(a, b)
    totals_a = a.footer.totals if a.footer else None
    totals_b = b.footer.totals if b.footer else None
    return TraceDiff(
        trace_a_path=a_path,
        trace_b_path=b_path,
        same_task=(
            a.header.task_id == b.header.task_id and a.header.task_hash == b.header.task_hash
        ),
        same_seed=a.header.seed == b.header.seed,
        header_delta=_header_delta(a, b),
        first_divergence=first_divergence,
        step_count_a=len(a.steps),
        step_count_b=len(b.steps),
        score_deltas=_score_deltas(a, b),
        agent_latency_delta_ms=(totals_b.agent_latency_ms if totals_b else 0.0)
        - (totals_a.agent_latency_ms if totals_a else 0.0),
        env_latency_delta_ms=(totals_b.env_latency_ms if totals_b else 0.0)
        - (totals_a.env_latency_ms if totals_a else 0.0),
        input_tokens_delta=(totals_b.input_tokens if totals_b else 0)
        - (totals_a.input_tokens if totals_a else 0),
        output_tokens_delta=(totals_b.output_tokens if totals_b else 0)
        - (totals_a.output_tokens if totals_a else 0),
    )


def _header_delta(a: Trace, b: Trace) -> HeaderDelta:
    model_a, model_b = a.header.model_under_test, b.header.model_under_test
    judge_a, judge_b = a.header.judge_model, b.header.judge_model
    fingerprint_a = a.header.environment_fingerprint
    fingerprint_b = b.header.environment_fingerprint
    hash_a, hash_b = a.header.task_hash, b.header.task_hash
    return HeaderDelta(
        model_under_test_a=model_a,
        model_under_test_b=model_b,
        model_under_test_changed=model_a != model_b,
        judge_model_a=judge_a,
        judge_model_b=judge_b,
        judge_model_changed=judge_a != judge_b,
        environment_fingerprint_a=fingerprint_a,
        environment_fingerprint_b=fingerprint_b,
        environment_fingerprint_changed=fingerprint_a != fingerprint_b,
        task_hash_a=hash_a,
        task_hash_b=hash_b,
        task_hash_changed=hash_a != hash_b,
    )


def _first_divergence(a: Trace, b: Trace) -> StepDivergence | None:
    for i in range(max(len(a.steps), len(b.steps))):
        step_a = a.steps[i] if i < len(a.steps) else None
        step_b = b.steps[i] if i < len(b.steps) else None
        same = (
            step_a is not None
            and step_b is not None
            and step_a.action == step_b.action
            and step_a.observation_hash == step_b.observation_hash
        )
        if not same:
            return StepDivergence(
                index=i,
                action_a=step_a.action if step_a else None,
                action_b=step_b.action if step_b else None,
                observation_hash_a=step_a.observation_hash if step_a else None,
                observation_hash_b=step_b.observation_hash if step_b else None,
                observation_a=step_a.observation if step_a else None,
                observation_b=step_b.observation if step_b else None,
            )
    return None


def _score_deltas(a: Trace, b: Trace) -> list[ScoreDelta]:
    scores_a = {s.scorer: s for s in (a.footer.scores if a.footer else [])}
    scores_b = {s.scorer: s for s in (b.footer.scores if b.footer else [])}
    deltas: list[ScoreDelta] = []
    for scorer in sorted(set(scores_a) | set(scores_b)):
        sa, sb = scores_a.get(scorer), scores_b.get(scorer)
        deltas.append(
            ScoreDelta(
                scorer=scorer,
                score_a=sa.score if sa else None,
                score_b=sb.score if sb else None,
                passed_a=sa.passed if sa else None,
                passed_b=sb.passed if sb else None,
            )
        )
    return deltas


__all__ = ["HeaderDelta", "ScoreDelta", "StepDivergence", "TraceDiff", "diff_traces"]
