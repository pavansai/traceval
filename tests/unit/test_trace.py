from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from traceval.providers import ResolvedModel
from traceval.trace import (
    AgentKind,
    NoPassingTraceFoundError,
    Outcome,
    ScoreResult,
    Trace,
    TraceFooter,
    TraceHeader,
    TraceSchemaError,
    TraceStep,
    TraceTotals,
    TraceWriter,
    diff_traces,
    find_last_passing_trace,
    read_trace,
)


def _model() -> ResolvedModel:
    return ResolvedModel(provider="mock", model_id="mock-1-20260101", alias="mock-1")


def _header(**overrides: object) -> TraceHeader:
    defaults: dict[str, object] = dict(
        run_id="run-1",
        task_id="example_search_task",
        task_hash="sha256:deadbeef",
        task_format_version=1,
        seed=42,
        model_under_test=_model(),
        agent_kind=AgentKind.ORACLE,
        environment_fingerprint="sha256:cafef00d",
        traceval_version="0.1.0-test",
        started_at=datetime.now(UTC),
    )
    defaults.update(overrides)
    return TraceHeader(**defaults)  # type: ignore[arg-type]


def _step(index: int, action: dict[str, object] | None = None) -> TraceStep:
    return TraceStep(
        index=index,
        observation_hash=f"sha256:obs{index}",
        action=action or {"kind": "click", "target": "#go"},
        input_tokens=10,
        output_tokens=5,
        agent_latency_ms=10.0,
        env_latency_ms=2.5,
        timestamp=datetime.now(UTC),
    )


def _footer(passed: bool = True) -> TraceFooter:
    return TraceFooter(
        outcome=Outcome.SUCCESS if passed else Outcome.FAILURE,
        total_steps=1,
        scores=[ScoreResult(scorer="exact_match", score=1.0 if passed else 0.0, passed=passed)],
        totals=TraceTotals(
            input_tokens=10, output_tokens=5, agent_latency_ms=10.0, env_latency_ms=2.5
        ),
        ended_at=datetime.now(UTC),
    )


def test_writer_reader_round_trip(tmp_path: Path) -> None:
    trace_path = tmp_path / "run.jsonl"
    header = _header()
    step = _step(0)
    footer = _footer()

    with TraceWriter(trace_path) as writer:
        writer.write_header(header)
        writer.write_step(step)
        writer.write_footer(footer)

    trace = read_trace(trace_path)
    assert trace.header.run_id == header.run_id
    assert trace.header.model_under_test.model_id == "mock-1-20260101"
    assert len(trace.steps) == 1
    assert trace.steps[0].action == step.action
    assert trace.footer is not None
    assert trace.footer.outcome == Outcome.SUCCESS


def test_writer_is_append_only(tmp_path: Path) -> None:
    trace_path = tmp_path / "run.jsonl"
    with TraceWriter(trace_path) as writer:
        writer.write_header(_header())
    with TraceWriter(trace_path) as writer:
        writer.write_step(_step(0))
        writer.write_footer(_footer())

    trace = read_trace(trace_path)
    assert len(trace.steps) == 1
    assert trace.footer is not None


def test_reader_rejects_unknown_schema_version(tmp_path: Path) -> None:
    trace_path = tmp_path / "bad.jsonl"
    header = _header()
    bad_line = header.model_dump_json(exclude={"schema_version"})
    import json

    payload = json.loads(bad_line)
    payload["schema_version"] = 999
    trace_path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(TraceSchemaError):
        read_trace(trace_path)


def test_reader_rejects_missing_header(tmp_path: Path) -> None:
    trace_path = tmp_path / "no_header.jsonl"
    with TraceWriter(trace_path) as writer:
        writer.write_step(_step(0))

    with pytest.raises(TraceSchemaError):
        read_trace(trace_path)


def test_diff_identical_traces_has_no_divergence(tmp_path: Path) -> None:
    header = _header()
    trace = Trace(header=header, steps=[_step(0), _step(1)], footer=_footer())
    diff = diff_traces(trace, trace)
    assert diff.diverged is False
    assert diff.same_task is True
    assert diff.same_seed is True
    assert diff.agent_latency_delta_ms == 0.0
    assert diff.env_latency_delta_ms == 0.0
    assert diff.header_delta.has_changes is False


def test_diff_header_delta_reports_model_environment_and_task_changes() -> None:
    header_a = _header()
    header_b = _header(
        model_under_test=ResolvedModel(provider="mock", model_id="mock-2-x", alias="mock-2"),
        environment_fingerprint="sha256:changed",
        task_hash="sha256:changedtask",
    )
    trace_a = Trace(header=header_a, steps=[], footer=_footer())
    trace_b = Trace(header=header_b, steps=[], footer=_footer())

    diff = diff_traces(trace_a, trace_b)

    assert diff.header_delta.model_under_test_changed is True
    assert diff.header_delta.environment_fingerprint_changed is True
    assert diff.header_delta.task_hash_changed is True
    assert diff.header_delta.judge_model_changed is False
    assert diff.header_delta.has_changes is True
    # task_hash differs even though task_id is the same, so this is caught
    # by same_task as well as header_delta.
    assert diff.same_task is False


def test_diff_detects_first_divergent_step(tmp_path: Path) -> None:
    header = _header()
    trace_a = Trace(
        header=header,
        steps=[_step(0), _step(1, action={"kind": "click", "target": "#a"})],
        footer=_footer(passed=True),
    )
    trace_b = Trace(
        header=header,
        steps=[_step(0), _step(1, action={"kind": "click", "target": "#b"})],
        footer=_footer(passed=False),
    )
    diff = diff_traces(trace_a, trace_b)
    assert diff.diverged is True
    assert diff.first_divergence is not None
    assert diff.first_divergence.index == 1
    assert diff.first_divergence.action_a == {"kind": "click", "target": "#a"}
    assert diff.first_divergence.action_b == {"kind": "click", "target": "#b"}
    assert diff.score_deltas[0].passed_a is True
    assert diff.score_deltas[0].passed_b is False


def test_diff_detects_length_mismatch() -> None:
    header = _header()
    trace_a = Trace(header=header, steps=[_step(0)], footer=_footer())
    trace_b = Trace(header=header, steps=[_step(0), _step(1)], footer=_footer())
    diff = diff_traces(trace_a, trace_b)
    assert diff.diverged is True
    assert diff.first_divergence is not None
    assert diff.first_divergence.index == 1
    assert diff.first_divergence.action_a is None


def _write_full_trace(
    path: Path,
    *,
    task_hash: str = "sha256:deadbeef",
    outcome: Outcome = Outcome.SUCCESS,
    started_at: datetime,
    run_id: str | None = None,
) -> None:
    header = _header(task_hash=task_hash, started_at=started_at, run_id=run_id or path.stem)
    footer = TraceFooter(
        outcome=outcome,
        total_steps=1,
        totals=TraceTotals(
            input_tokens=10, output_tokens=5, agent_latency_ms=10.0, env_latency_ms=2.5
        ),
        ended_at=started_at,
    )
    with TraceWriter(path) as writer:
        writer.write_header(header)
        writer.write_step(_step(0))
        writer.write_footer(footer)


def test_find_last_passing_trace_picks_most_recent_by_started_at(tmp_path: Path) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    target = tmp_path / "target.jsonl"
    _write_full_trace(target, outcome=Outcome.FAILURE, started_at=base)
    older = tmp_path / "older.jsonl"
    _write_full_trace(older, started_at=base - timedelta(hours=2))
    newest = tmp_path / "newest.jsonl"
    _write_full_trace(newest, started_at=base - timedelta(minutes=1))
    newer_but_not_newest = tmp_path / "middle.jsonl"
    _write_full_trace(newer_but_not_newest, started_at=base - timedelta(hours=1))

    result = find_last_passing_trace(target, "sha256:deadbeef")

    assert result == newest


def test_find_last_passing_trace_ignores_different_task_hash(tmp_path: Path) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    target = tmp_path / "target.jsonl"
    _write_full_trace(target, outcome=Outcome.FAILURE, started_at=base)
    other_task = tmp_path / "other_task.jsonl"
    _write_full_trace(
        other_task, task_hash="sha256:othertask", started_at=base - timedelta(minutes=1)
    )
    matching = tmp_path / "matching.jsonl"
    _write_full_trace(matching, started_at=base - timedelta(hours=1))

    result = find_last_passing_trace(target, "sha256:deadbeef")

    assert result == matching


def test_find_last_passing_trace_ignores_failed_and_errored_traces(tmp_path: Path) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    target = tmp_path / "target.jsonl"
    _write_full_trace(target, outcome=Outcome.FAILURE, started_at=base)
    failed = tmp_path / "failed.jsonl"
    _write_full_trace(failed, outcome=Outcome.FAILURE, started_at=base - timedelta(minutes=1))
    errored = tmp_path / "errored.jsonl"
    _write_full_trace(errored, outcome=Outcome.ERROR, started_at=base - timedelta(minutes=2))
    passing = tmp_path / "passing.jsonl"
    _write_full_trace(passing, outcome=Outcome.SUCCESS, started_at=base - timedelta(hours=1))

    result = find_last_passing_trace(target, "sha256:deadbeef")

    assert result == passing


def test_find_last_passing_trace_raises_when_no_passing_runs_at_all(tmp_path: Path) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    target = tmp_path / "target.jsonl"
    _write_full_trace(target, outcome=Outcome.FAILURE, started_at=base)
    failed = tmp_path / "failed.jsonl"
    _write_full_trace(failed, outcome=Outcome.FAILURE, started_at=base - timedelta(minutes=1))

    with pytest.raises(NoPassingTraceFoundError, match="no trace in this directory has"):
        find_last_passing_trace(target, "sha256:deadbeef")


def test_find_last_passing_trace_raises_when_no_passing_runs_for_this_task_hash(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    target = tmp_path / "target.jsonl"
    _write_full_trace(target, outcome=Outcome.FAILURE, started_at=base)
    passing_other_task = tmp_path / "other_task.jsonl"
    _write_full_trace(
        passing_other_task,
        task_hash="sha256:othertask",
        started_at=base - timedelta(minutes=1),
    )

    with pytest.raises(NoPassingTraceFoundError, match="other tasks have passing runs"):
        find_last_passing_trace(target, "sha256:deadbeef")
