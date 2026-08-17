"""Runner: orchestrates a single task run end-to-end.

env.reset(seed) -> bounded agent loop -> trace write -> scorers -> footer.
This is the one place that wires `Environment`, `Agent`, `TraceWriter`, and
`Scorer` together; adding a new environment kind or agent kind never touches
this file, only the registries/factories each side plugs into.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path

from traceval.agents import Agent
from traceval.environments import Environment, Observation
from traceval.environments.browser import BrowserEnvironment
from traceval.providers import ResolvedModel
from traceval.scoring import Scorer
from traceval.scoring.report import PricingTable
from traceval.tasks import Task, build_fixture, compute_task_hash
from traceval.trace import (
    AgentKind,
    Outcome,
    PricingSnapshot,
    ScoreResult,
    Trace,
    TraceError,
    TraceFooter,
    TraceHeader,
    TraceStep,
    TraceTotals,
    TraceWriter,
)

_ENVIRONMENT_FACTORIES = {
    "browser": BrowserEnvironment,
}


def _traceval_version() -> str:
    try:
        return _package_version("traceval")
    except PackageNotFoundError:
        return "unknown"


TRACEVAL_VERSION = _traceval_version()


def build_environment(task: Task) -> Environment:
    try:
        factory = _ENVIRONMENT_FACTORIES[task.environment.kind]
    except KeyError as exc:
        raise ValueError(f"unknown environment kind: {task.environment.kind!r}") from exc
    return factory(task.environment.config)


def run_task(
    task: Task,
    agent: Agent,
    agent_kind: AgentKind,
    model_under_test: ResolvedModel,
    scorers: list[Scorer],
    trace_dir: Path,
    judge_model: ResolvedModel | None = None,
    run_id: str | None = None,
) -> Trace:
    run_id = run_id or uuid.uuid4().hex

    environment = build_environment(task)
    fixture = build_fixture(task)
    task_hash = compute_task_hash(task)
    trace_path = trace_dir / f"{run_id}.jsonl"

    pricing_rate = PricingTable.load().rates(model_under_test.provider, model_under_test.model_id)
    pricing_snapshot = (
        PricingSnapshot(
            input_per_million_usd=pricing_rate.input_per_million_usd,
            output_per_million_usd=pricing_rate.output_per_million_usd,
        )
        if pricing_rate is not None
        else None
    )

    def _build_header(environment_fingerprint: str) -> TraceHeader:
        return TraceHeader(
            run_id=run_id,
            task_id=task.id,
            task_hash=task_hash,
            task_format_version=task.format_version,
            seed=task.seed,
            model_under_test=model_under_test,
            judge_model=judge_model,
            agent_kind=agent_kind,
            environment_fingerprint=environment_fingerprint,
            traceval_version=TRACEVAL_VERSION,
            pricing_snapshot=pricing_snapshot,
            started_at=datetime.now(UTC),
        )

    steps: list[TraceStep] = []
    header: TraceHeader | None = None
    error: TraceError | None = None
    caught: Exception | None = None

    with TraceWriter(trace_path) as writer:
        try:
            observation = environment.reset(task.seed, fixture)
            header = _build_header(environment.fingerprint())
            writer.write_header(header)

            done = False
            while len(steps) < task.max_steps and not done:
                agent_step = agent.act(observation, steps)
                if agent_step.action is None:
                    break
                env_start = time.monotonic()
                step_result = environment.step(agent_step.action)
                env_latency_ms = (time.monotonic() - env_start) * 1000

                step = TraceStep(
                    index=len(steps),
                    observation_hash=_hash_observation(step_result.observation),
                    observation=step_result.observation.model_dump(),
                    action=agent_step.action.model_dump(),
                    input_tokens=agent_step.usage.input_tokens,
                    output_tokens=agent_step.usage.output_tokens,
                    agent_latency_ms=agent_step.agent_latency_ms,
                    env_latency_ms=env_latency_ms,
                    timestamp=datetime.now(UTC),
                )
                writer.write_step(step)
                steps.append(step)
                observation = step_result.observation
                done = step_result.done
        except Exception as exc:
            caught = exc
            error = TraceError(error_type=type(exc).__name__, error_message=str(exc))
            if header is None:
                # reset() itself raised, so no environment fingerprint was ever
                # obtained. Still write a header so the trace is readable.
                header = _build_header("unavailable")
                writer.write_header(header)
        finally:
            environment.close()

        assert header is not None  # set on every path above, error or not

        scores: list[ScoreResult] = []
        scorer_errors: list[TraceError] = []
        if error is not None:
            outcome = Outcome.ERROR
        else:
            trace_so_far = Trace(header=header, steps=steps, footer=None)
            # Each scorer runs in its own try/except: one scorer failing
            # (e.g. an unparseable judge response) must not discard scores
            # that other scorers already produced, and must not stop later
            # scorers from still running.
            for scorer in scorers:
                try:
                    scores.append(scorer.score(task, trace_so_far))
                except Exception as exc:
                    scorer_errors.append(
                        TraceError(
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            scorer=type(scorer).__name__,
                        )
                    )
                    if caught is None:
                        caught = exc
            if scorer_errors:
                # Can't claim SUCCESS or FAILURE without complete scoring,
                # even if every other scorer passed.
                outcome = Outcome.ERROR
            else:
                outcome = Outcome.SUCCESS if all(s.passed for s in scores) else Outcome.FAILURE

        totals = TraceTotals(
            input_tokens=sum(s.input_tokens for s in steps),
            output_tokens=sum(s.output_tokens for s in steps),
            agent_latency_ms=sum(s.agent_latency_ms for s in steps),
            env_latency_ms=sum(s.env_latency_ms for s in steps),
        )
        footer = TraceFooter(
            outcome=outcome,
            total_steps=len(steps),
            scores=scores,
            totals=totals,
            error=error,
            scorer_errors=scorer_errors,
            ended_at=datetime.now(UTC),
        )
        writer.write_footer(footer)

    if caught is not None:
        raise caught

    return Trace(header=header, steps=steps, footer=footer)


def _hash_observation(observation: Observation) -> str:
    digest = hashlib.sha256(observation.model_dump_json().encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


__all__ = ["build_environment", "run_task"]
