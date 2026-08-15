"""Runner unit tests against a fake Environment/Agent (no Playwright involved).

Covers behavior that's awkward to exercise through the real BrowserEnvironment:
token/latency propagation from AgentStep into TraceStep, and the
reset/step failure paths that must still produce a readable, footer-complete
trace with `Outcome.ERROR`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from traceval.agents import AgentStep, AgentUsage
from traceval.environments import Action, Observation, StepResult
from traceval.providers import ResolvedModel
from traceval.runner import runner as runner_module
from traceval.runner.runner import run_task
from traceval.tasks import Task, TaskFixture, load_task
from traceval.trace import AgentKind, Outcome, TraceStep, read_trace


class FakeEnvironment:
    def __init__(self, config: dict[str, Any]) -> None:
        self.fail_reset: bool = config.get("fail_reset", False)
        self.fail_step_at: int | None = config.get("fail_step_at")
        self._step_index = 0
        self.closed = False

    def reset(self, seed: int, fixture: TaskFixture) -> Observation:
        if self.fail_reset:
            raise RuntimeError("boom-reset")
        return Observation(url="fake://start", title="start", elements={})

    def step(self, action: Action) -> StepResult:
        if self.fail_step_at is not None and self._step_index == self.fail_step_at:
            raise RuntimeError("boom-step")
        self._step_index += 1
        return StepResult(observation=Observation(url="fake://step", title="step"), done=False)

    def fingerprint(self) -> str:
        return "sha256:fakefingerprint"

    def close(self) -> None:
        self.closed = True


class FakeAgent:
    def __init__(self, steps: list[AgentStep]) -> None:
        self._steps = steps
        self._index = 0

    def act(self, observation: Observation, history: list[TraceStep]) -> AgentStep:
        if self._index >= len(self._steps):
            return AgentStep(action=None)
        step = self._steps[self._index]
        self._index += 1
        return step


@pytest.fixture(autouse=True)
def _register_fake_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(runner_module._ENVIRONMENT_FACTORIES, "fake", FakeEnvironment)


def _model() -> ResolvedModel:
    return ResolvedModel(provider="mock", model_id="mock-1-20260101", alias="mock-1")


def _make_task(task_dir: Path, config: dict[str, Any] | None = None, max_steps: int = 5) -> Task:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.yaml").write_text(
        "id: fake_task\n"
        "seed: 1\n"
        f"max_steps: {max_steps}\n"
        "environment:\n"
        "  kind: fake\n"
        f"  config: {json.dumps(config or {})}\n"
    )
    return load_task(task_dir)


def test_failing_reset_produces_readable_error_trace(tmp_path: Path) -> None:
    task = _make_task(tmp_path / "task", config={"fail_reset": True})
    trace_dir = tmp_path / "runs"

    with pytest.raises(RuntimeError, match="boom-reset"):
        run_task(
            task=task,
            agent=FakeAgent([]),
            agent_kind=AgentKind.ORACLE,
            model_under_test=_model(),
            scorers=[],
            trace_dir=trace_dir,
        )

    trace_files = list(trace_dir.glob("*.jsonl"))
    assert len(trace_files) == 1
    trace = read_trace(trace_files[0])
    assert trace.header.environment_fingerprint == "unavailable"
    assert trace.header.traceval_version  # still stamped even on a reset failure
    assert trace.header.pricing_snapshot is not None
    assert trace.footer is not None
    assert trace.footer.outcome == Outcome.ERROR
    assert trace.footer.total_steps == 0
    assert trace.footer.error is not None
    assert trace.footer.error.error_type == "RuntimeError"
    assert "boom-reset" in trace.footer.error.error_message


def test_failing_step_produces_readable_error_trace(tmp_path: Path) -> None:
    task = _make_task(tmp_path / "task", config={"fail_step_at": 0})
    trace_dir = tmp_path / "runs"
    agent = FakeAgent([AgentStep(action=Action(kind="click", target="#x"))])

    with pytest.raises(RuntimeError, match="boom-step"):
        run_task(
            task=task,
            agent=agent,
            agent_kind=AgentKind.ORACLE,
            model_under_test=_model(),
            scorers=[],
            trace_dir=trace_dir,
        )

    trace_files = list(trace_dir.glob("*.jsonl"))
    assert len(trace_files) == 1
    trace = read_trace(trace_files[0])
    assert trace.header.environment_fingerprint == "sha256:fakefingerprint"
    assert trace.footer is not None
    assert trace.footer.outcome == Outcome.ERROR
    assert trace.footer.total_steps == 0
    assert trace.footer.error is not None
    assert trace.footer.error.error_type == "RuntimeError"
    assert "boom-step" in trace.footer.error.error_message


def test_agent_usage_and_latency_split_recorded_separately(tmp_path: Path) -> None:
    task = _make_task(tmp_path / "task")
    trace_dir = tmp_path / "runs"
    agent = FakeAgent(
        [
            AgentStep(
                action=Action(kind="click", target="#x"),
                usage=AgentUsage(input_tokens=7, output_tokens=3),
                agent_latency_ms=42.0,
            )
        ]
    )

    trace = run_task(
        task=task,
        agent=agent,
        agent_kind=AgentKind.ORACLE,
        model_under_test=_model(),
        scorers=[],
        trace_dir=trace_dir,
    )

    assert len(trace.steps) == 1
    step = trace.steps[0]
    assert step.input_tokens == 7
    assert step.output_tokens == 3
    assert step.agent_latency_ms == 42.0
    assert step.env_latency_ms >= 0.0

    assert trace.footer is not None
    assert trace.footer.outcome == Outcome.SUCCESS
    assert trace.footer.totals.input_tokens == 7
    assert trace.footer.totals.output_tokens == 3
    assert trace.footer.totals.agent_latency_ms == 42.0
    assert trace.footer.totals.env_latency_ms >= 0.0


def test_header_stamps_traceval_version_and_pricing_snapshot(tmp_path: Path) -> None:
    task = _make_task(tmp_path / "task")
    trace_dir = tmp_path / "runs"

    trace = run_task(
        task=task,
        agent=FakeAgent([]),
        agent_kind=AgentKind.ORACLE,
        model_under_test=_model(),
        scorers=[],
        trace_dir=trace_dir,
    )

    assert trace.header.traceval_version
    # mock-1-20260101 is priced at 0.0/0.0 in pricing.yaml.
    assert trace.header.pricing_snapshot is not None
    assert trace.header.pricing_snapshot.input_per_million_usd == 0.0
    assert trace.header.pricing_snapshot.output_per_million_usd == 0.0
