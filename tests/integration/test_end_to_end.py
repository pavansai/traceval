"""End-to-end integration tests: real Playwright browser env, oracle/replay
agents, real scorers. No API key is used anywhere in this file; the
model-graded scorer is tested against MockProvider as the judge.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from traceval.agents.live import LiveAgent
from traceval.agents.oracle import OracleAgent
from traceval.agents.replay import ReplayAgent
from traceval.cli.main import app
from traceval.environments import Action, Observation, StepResult
from traceval.providers import ResolvedModel
from traceval.providers.mock_provider import MockProvider
from traceval.runner import run_task
from traceval.runner import runner as runner_module
from traceval.scoring import ExactMatchScorer, ModelGradedScorer, RubricScorer, build_report
from traceval.tasks import ScorerConfig, load_task
from traceval.trace import (
    AgentKind,
    Outcome,
    Trace,
    TraceFooter,
    TraceHeader,
    TraceTotals,
    TraceWriter,
    diff_traces,
    read_trace,
)

EXAMPLE_TASK_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "tasks" / "example_search_task"
)
RANDOM_TASK_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "tasks" / "random_dom_task"
REPO_TASKS_DIR = Path(__file__).resolve().parents[2] / "tasks"


def _model() -> ResolvedModel:
    return ResolvedModel(provider="mock", model_id="mock-1-20260101", alias="mock-1")


def _write_minimal_trace(
    path: Path, *, task_id: str = "task-a", task_hash: str = "sha256:aaa", seed: int = 1
) -> None:
    header = TraceHeader(
        run_id="run-x",
        task_id=task_id,
        task_hash=task_hash,
        task_format_version=1,
        seed=seed,
        model_under_test=_model(),
        agent_kind=AgentKind.ORACLE,
        environment_fingerprint="sha256:env",
        traceval_version="0.1.0-test",
        started_at=datetime.now(UTC),
    )
    footer = TraceFooter(
        outcome=Outcome.SUCCESS,
        total_steps=0,
        totals=TraceTotals(
            input_tokens=0, output_tokens=0, agent_latency_ms=0.0, env_latency_ms=0.0
        ),
        ended_at=datetime.now(UTC),
    )
    with TraceWriter(path) as writer:
        writer.write_header(header)
        writer.write_footer(footer)


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def normalize_cli_output(output: str) -> str:
    """Strip ANSI escape codes and collapse whitespace/line-wrapping.

    typer/rich can color-code and line-wrap CLI output depending on
    terminal detection, splitting a single word or flag across separate
    ANSI spans or line breaks. Every assertion against `CliRunner` output
    should go through this rather than checking `result.output` raw, so the
    assertion doesn't depend on how rich decided to render that particular
    run (see tests/conftest.py's NO_COLOR/TERM/COLUMNS fixture for the other
    half of this: making that rendering deterministic in the first place).
    """
    plain = _ANSI_ESCAPE_RE.sub("", output)
    return " ".join(plain.split())


def test_oracle_run_produces_valid_trace(tmp_path: Path) -> None:
    task = load_task(EXAMPLE_TASK_DIR)
    trace = run_task(
        task=task,
        agent=OracleAgent(EXAMPLE_TASK_DIR / "scripted_trajectory.jsonl"),
        agent_kind=AgentKind.ORACLE,
        model_under_test=_model(),
        scorers=[ExactMatchScorer(), RubricScorer()],
        trace_dir=tmp_path,
    )

    assert trace.header.task_hash.startswith("sha256:")
    assert trace.header.environment_fingerprint.startswith("sha256:")
    assert trace.header.task_format_version == task.format_version
    assert len(trace.steps) == 2
    assert trace.footer is not None
    assert trace.footer.outcome == Outcome.SUCCESS
    assert all(score.passed for score in trace.footer.scores)

    # Round-trips through the JSONL file on disk.
    trace_path = tmp_path / f"{trace.header.run_id}.jsonl"
    reloaded = read_trace(trace_path)
    assert reloaded.header.task_hash == trace.header.task_hash
    assert len(reloaded.steps) == len(trace.steps)
    assert reloaded.footer is not None
    assert reloaded.footer.outcome == Outcome.SUCCESS


def test_unrecoverable_account_task_fails_without_erroring(tmp_path: Path) -> None:
    """`tasks/unrecoverable_account` is intentionally unwinnable: a
    regression signal for the harness itself, since a benchmark where every
    task is passable can't distinguish a working scorer from a lenient one.
    Its actions must all complete cleanly (Outcome.ERROR would mean the
    harness broke) and its scorer must correctly report the goal as unmet
    (Outcome.SUCCESS would mean the scorer or environment silently broke).
    """
    task_dir = REPO_TASKS_DIR / "unrecoverable_account"
    task = load_task(task_dir)

    trace = run_task(
        task=task,
        agent=OracleAgent(task_dir / "scripted_trajectory.jsonl"),
        agent_kind=AgentKind.ORACLE,
        model_under_test=_model(),
        scorers=[ExactMatchScorer()],
        trace_dir=tmp_path,
    )

    assert trace.footer is not None
    assert trace.footer.error is None
    assert trace.footer.scorer_errors == []
    assert trace.footer.outcome == Outcome.FAILURE
    assert all(not score.passed for score in trace.footer.scores)


def test_live_agent_path_produces_nonzero_totals_and_cost(tmp_path: Path) -> None:
    """LiveAgent + MockProvider (no API key) proves usage flows: act() ->
    AgentStep.usage -> TraceStep -> TraceTotals -> report cost, which was
    silently zero end-to-end before AgentStep existed.
    """
    task = load_task(EXAMPLE_TASK_DIR)
    task.max_steps = 1
    # A model priced in pricing.yaml, resolved by hand so MockProvider (which
    # ignores `resolved`) can stand in without ever touching a real API key.
    resolved_model = ResolvedModel(
        provider="anthropic", model_id="claude-sonnet-5", alias="claude-sonnet-5"
    )
    provider = MockProvider(canned_response='{"kind": "wait", "ms": 0}')

    trace = run_task(
        task=task,
        agent=LiveAgent(provider, resolved_model, goal=task.goal),
        agent_kind=AgentKind.LIVE,
        model_under_test=resolved_model,
        scorers=[],
        trace_dir=tmp_path,
    )

    assert trace.footer is not None
    assert trace.footer.totals.input_tokens > 0
    assert trace.footer.totals.output_tokens > 0

    report = build_report([trace])
    assert report.total_cost_usd is not None
    assert report.total_cost_usd > 0


def test_same_seed_and_agent_produce_identical_actions_and_observations(tmp_path: Path) -> None:
    task = load_task(EXAMPLE_TASK_DIR)

    def _run() -> list[tuple[dict[str, object], dict[str, object]]]:
        trace = run_task(
            task=task,
            agent=OracleAgent(EXAMPLE_TASK_DIR / "scripted_trajectory.jsonl"),
            agent_kind=AgentKind.ORACLE,
            model_under_test=_model(),
            scorers=[ExactMatchScorer()],
            trace_dir=tmp_path,
        )
        return [(step.action, step.observation) for step in trace.steps]

    first = _run()
    second = _run()
    assert first == second


def test_oracle_determinism_across_repeated_runs(tmp_path: Path) -> None:
    """Same task, same seed, oracle agent, run N times: traces must be equal
    modulo run_id/timestamps and the four latency fields (real wall-clock
    measurements that legitimately vary between runs, not a determinism bug).
    """
    task = load_task(EXAMPLE_TASK_DIR)

    def _run(run_id: str) -> Trace:
        return run_task(
            task=task,
            agent=OracleAgent(EXAMPLE_TASK_DIR / "scripted_trajectory.jsonl"),
            agent_kind=AgentKind.ORACLE,
            model_under_test=_model(),
            scorers=[ExactMatchScorer(), RubricScorer()],
            trace_dir=tmp_path,
            run_id=run_id,
        )

    def _normalize(trace: Trace) -> tuple[object, object, object]:
        assert trace.footer is not None
        header = trace.header.model_dump(exclude={"run_id", "started_at"})
        steps = [
            step.model_dump(exclude={"timestamp", "agent_latency_ms", "env_latency_ms"})
            for step in trace.steps
        ]
        footer = trace.footer.model_dump(exclude={"ended_at"})
        excluded_totals_keys = {"agent_latency_ms", "env_latency_ms"}
        footer["totals"] = {
            k: v for k, v in footer["totals"].items() if k not in excluded_totals_keys
        }
        return (header, steps, footer)

    traces = [_run(f"determinism-{i}") for i in range(3)]
    normalized = [_normalize(t) for t in traces]

    assert normalized[0] == normalized[1] == normalized[2]


def _run_random_dom_task(seed: int, run_id: str, trace_dir: Path) -> str:
    """Run random_dom_task and return the #rand div's rendered text.

    That div is set by the fixture page's own `Math.random()` call, so its
    value is only reproducible/seed-sensitive if browser.py's seeded-PRNG
    init script is actually substituting `Math.random` before the page
    loads. Unlike example_search_task (a static page with no randomness at
    all), this task fails loudly if that substitution ever regresses.
    """
    task = load_task(RANDOM_TASK_DIR)
    task.seed = seed
    trace = run_task(
        task=task,
        agent=OracleAgent(RANDOM_TASK_DIR / "scripted_trajectory.jsonl"),
        agent_kind=AgentKind.ORACLE,
        model_under_test=_model(),
        scorers=[],
        trace_dir=trace_dir,
        run_id=run_id,
    )
    assert len(trace.steps) == 1
    value = trace.steps[0].observation["elements"]["#rand"]
    assert value  # sanity: the script actually rendered something
    return value


def test_seeded_randomness_reproducible_across_runs(tmp_path: Path) -> None:
    """Positive case: same seed, N runs, identical Math.random() output.

    Fails if the seeded-PRNG substitution in browser.py breaks (e.g. the
    init script stops running, or Math.random reverts to the real one).
    """
    values = [
        _run_random_dom_task(seed=42, run_id=f"rand-same-{i}", trace_dir=tmp_path) for i in range(3)
    ]
    assert values[0] == values[1] == values[2]


def test_different_seeds_produce_different_randomness(tmp_path: Path) -> None:
    """Negative control for the test above: different seeds must produce
    different Math.random() output. Without this, the positive test alone
    can't distinguish "seeding actually works" from "seeding is a no-op and
    Math.random always returns the same thing anyway" (it doesn't, but nothing
    else in this suite would catch it if it started to).
    """
    value_a = _run_random_dom_task(seed=1, run_id="rand-a", trace_dir=tmp_path)
    value_b = _run_random_dom_task(seed=2, run_id="rand-b", trace_dir=tmp_path)
    assert value_a != value_b


def test_oracle_run_ends_at_trajectory_end_not_max_steps(tmp_path: Path) -> None:
    task = load_task(EXAMPLE_TASK_DIR)
    task.max_steps = 50  # far more than the 2 scripted actions
    trace = run_task(
        task=task,
        agent=OracleAgent(EXAMPLE_TASK_DIR / "scripted_trajectory.jsonl"),
        agent_kind=AgentKind.ORACLE,
        model_under_test=_model(),
        scorers=[ExactMatchScorer()],
        trace_dir=tmp_path,
    )
    assert len(trace.steps) == 2
    assert trace.footer is not None
    assert trace.footer.total_steps == 2


def test_replay_run_ends_at_trajectory_end_not_max_steps(tmp_path: Path) -> None:
    task = load_task(EXAMPLE_TASK_DIR)
    original = run_task(
        task=task,
        agent=OracleAgent(EXAMPLE_TASK_DIR / "scripted_trajectory.jsonl"),
        agent_kind=AgentKind.ORACLE,
        model_under_test=_model(),
        scorers=[ExactMatchScorer()],
        trace_dir=tmp_path,
    )
    original_path = tmp_path / f"{original.header.run_id}.jsonl"

    task.max_steps = 50  # far more than the 2 recorded actions
    replayed = run_task(
        task=task,
        agent=ReplayAgent(original_path),
        agent_kind=AgentKind.REPLAY,
        model_under_test=original.header.model_under_test,
        scorers=[ExactMatchScorer()],
        trace_dir=tmp_path,
    )
    assert len(replayed.steps) == 2
    assert replayed.footer is not None
    assert replayed.footer.total_steps == 2


def test_replay_reproduces_actions_and_diff_shows_no_divergence(tmp_path: Path) -> None:
    task = load_task(EXAMPLE_TASK_DIR)
    original = run_task(
        task=task,
        agent=OracleAgent(EXAMPLE_TASK_DIR / "scripted_trajectory.jsonl"),
        agent_kind=AgentKind.ORACLE,
        model_under_test=_model(),
        scorers=[ExactMatchScorer(), RubricScorer()],
        trace_dir=tmp_path,
    )
    original_path = tmp_path / f"{original.header.run_id}.jsonl"

    replayed = run_task(
        task=task,
        agent=ReplayAgent(original_path),
        agent_kind=AgentKind.REPLAY,
        model_under_test=original.header.model_under_test,
        scorers=[ExactMatchScorer(), RubricScorer()],
        trace_dir=tmp_path,
    )
    replayed_path = tmp_path / f"{replayed.header.run_id}.jsonl"

    diff = diff_traces(
        read_trace(original_path), read_trace(replayed_path), str(original_path), str(replayed_path)
    )
    assert diff.diverged is False
    assert diff.same_task is True
    assert all(delta.passed_a == delta.passed_b for delta in diff.score_deltas)


def test_model_graded_scorer_runs_against_mock_judge_no_api_key(
    tmp_path: Path, monkeypatch: object
) -> None:
    task = load_task(EXAMPLE_TASK_DIR)
    task.scorers.append(
        ScorerConfig(
            kind="model_graded",
            config={"rubric_prompt": "Did the agent search for 'playwright' and see a result?"},
        )
    )
    judge = MockProvider(canned_response="PASS\nThe agent completed the search correctly.")
    judge_model = ResolvedModel(
        provider="mock", model_id="mock-judge-1-20260101", alias="mock-judge"
    )

    trace = run_task(
        task=task,
        agent=OracleAgent(EXAMPLE_TASK_DIR / "scripted_trajectory.jsonl"),
        agent_kind=AgentKind.ORACLE,
        model_under_test=_model(),
        scorers=[
            ExactMatchScorer(),
            ModelGradedScorer(judge, judge_model),
        ],
        trace_dir=tmp_path,
        judge_model=judge_model,
    )

    assert trace.footer is not None
    scores_by_name = {s.scorer: s for s in trace.footer.scores}
    assert scores_by_name["model_graded"].passed is True
    assert trace.header.judge_model is not None
    assert trace.header.judge_model.model_id == "mock-judge-1-20260101"


def test_cli_run_end_to_end_with_oracle_agent(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            str(EXAMPLE_TASK_DIR),
            "--agent",
            "oracle",
            "--trace-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    output = normalize_cli_output(result.output)
    assert "success" in output
    assert "accuracy=100%" in output

    trace_files = list(tmp_path.glob("*.jsonl"))
    assert len(trace_files) == 1


def test_cli_diff_refuses_different_tasks_by_default(tmp_path: Path) -> None:
    trace_a = tmp_path / "a.jsonl"
    trace_b = tmp_path / "b.jsonl"
    _write_minimal_trace(trace_a, task_id="task-a", task_hash="sha256:aaa")
    _write_minimal_trace(trace_b, task_id="task-b", task_hash="sha256:bbb")

    runner = CliRunner()
    result = runner.invoke(app, ["diff", str(trace_a), str(trace_b)])
    assert result.exit_code != 0
    assert "--allow-different-task" in normalize_cli_output(result.output)


def test_cli_diff_allows_different_tasks_with_override(tmp_path: Path) -> None:
    trace_a = tmp_path / "a.jsonl"
    trace_b = tmp_path / "b.jsonl"
    _write_minimal_trace(trace_a, task_id="task-a", task_hash="sha256:aaa")
    _write_minimal_trace(trace_b, task_id="task-b", task_hash="sha256:bbb")

    runner = CliRunner()
    result = runner.invoke(app, ["diff", str(trace_a), str(trace_b), "--allow-different-task"])
    assert result.exit_code == 0, result.output

    output = normalize_cli_output(result.output)
    data = json.loads(output[output.index("{") :])
    assert data["same_task"] is False
    assert data["header_delta"]["task_hash_changed"] is True


def test_cli_diff_warns_on_different_seed(tmp_path: Path) -> None:
    trace_a = tmp_path / "a.jsonl"
    trace_b = tmp_path / "b.jsonl"
    _write_minimal_trace(trace_a, seed=1)
    _write_minimal_trace(trace_b, seed=2)

    runner = CliRunner()
    result = runner.invoke(app, ["diff", str(trace_a), str(trace_b)])
    assert result.exit_code == 0, result.output
    assert "WARNING" in normalize_cli_output(result.output)


class _BatchTestEnvironment:
    """Registered as env kind 'cli_batch_fake': deliberately fails in
    reset() when configured to, for testing that `traceval run` continues a
    multi-task batch past one task's failure rather than aborting it.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._should_fail = bool(config.get("should_fail", False))

    def reset(self, seed: int, fixture: object) -> Observation:
        if self._should_fail:
            raise RuntimeError("deliberate failure for batch-continuation test")
        return Observation(url="fake://start", title="start", elements={})

    def step(self, action: Action) -> StepResult:
        raise NotImplementedError

    def fingerprint(self) -> str:
        return "sha256:batchtest"

    def close(self) -> None:
        pass


def _write_batch_task_set(task_set: Path, entries: list[tuple[str, bool]]) -> None:
    for name, should_fail in entries:
        task_dir = task_set / name
        task_dir.mkdir(parents=True)
        (task_dir / "task.yaml").write_text(
            f"id: {name}\n"
            "seed: 1\n"
            "max_steps: 5\n"
            "goal: irrelevant for this fake, non-live environment\n"
            "environment:\n"
            "  kind: cli_batch_fake\n"
            f"  config: {{should_fail: {str(should_fail).lower()}}}\n"
        )
        (task_dir / "scripted_trajectory.jsonl").write_text("")


def test_cli_run_continues_batch_when_middle_task_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        runner_module._ENVIRONMENT_FACTORIES, "cli_batch_fake", _BatchTestEnvironment
    )

    task_set = tmp_path / "task_set"
    _write_batch_task_set(task_set, [("task_a", False), ("task_b", True), ("task_c", False)])

    trace_dir = tmp_path / "runs"
    runner = CliRunner()
    result = runner.invoke(
        app, ["run", str(task_set), "--agent", "oracle", "--trace-dir", str(trace_dir)]
    )

    # A harness that exits 0 with an unscored task would silently green a CI
    # pipeline, so a batch containing any error must exit nonzero by default.
    assert result.exit_code != 0

    output = normalize_cli_output(result.output)
    assert "task_a" in output
    assert "task_b" in output
    assert "task_c" in output
    assert "success=2" in output
    assert "error=1" in output

    trace_files = sorted(trace_dir.glob("*.jsonl"))
    assert len(trace_files) == 3
    traces_by_task = {t.header.task_id: t for t in (read_trace(p) for p in trace_files)}

    assert traces_by_task["task_a"].footer is not None
    assert traces_by_task["task_a"].footer.outcome == Outcome.SUCCESS
    assert traces_by_task["task_b"].footer is not None
    assert traces_by_task["task_b"].footer.outcome == Outcome.ERROR
    assert traces_by_task["task_b"].footer.error is not None
    assert traces_by_task["task_b"].footer.error.error_type == "RuntimeError"
    assert traces_by_task["task_c"].footer is not None
    assert traces_by_task["task_c"].footer.outcome == Outcome.SUCCESS


def test_cli_run_exit_zero_on_error_flag_preserves_old_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        runner_module._ENVIRONMENT_FACTORIES, "cli_batch_fake", _BatchTestEnvironment
    )

    task_set = tmp_path / "task_set"
    _write_batch_task_set(task_set, [("task_a", True)])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            str(task_set),
            "--agent",
            "oracle",
            "--trace-dir",
            str(tmp_path / "runs"),
            "--exit-zero-on-error",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "error=1" in normalize_cli_output(result.output)


def test_cli_run_skips_live_only_task_when_judge_is_mock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        runner_module._ENVIRONMENT_FACTORIES, "cli_batch_fake", _BatchTestEnvironment
    )

    task_set = tmp_path / "task_set"
    task_dir = task_set / "needs_real_judge"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        "id: needs_real_judge\n"
        "seed: 1\n"
        "max_steps: 5\n"
        "goal: irrelevant, never reached\n"
        "requires_live_judge: true\n"
        "environment:\n"
        "  kind: cli_batch_fake\n"
        "  config: {should_fail: false}\n"
        "scorers:\n"
        "  - kind: model_graded\n"
        "    config:\n"
        "      rubric_prompt: irrelevant, never reached\n"
    )
    (task_dir / "scripted_trajectory.jsonl").write_text("")

    trace_dir = tmp_path / "runs"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            str(task_set),
            "--agent",
            "oracle",
            "--judge-provider",
            "mock",
            "--judge-model",
            "mock-judge",
            "--trace-dir",
            str(trace_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    output = normalize_cli_output(result.output)
    assert "needs_real_judge: skipped" in output
    assert "runs: 0" in output
    assert not list(trace_dir.glob("*.jsonl"))
