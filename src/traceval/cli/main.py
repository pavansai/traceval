"""traceval CLI: run, replay, diff, report.

Kept thin on purpose. Every command below is a small wiring layer over
`runner.run_task`, `trace.diff_traces`, and `scoring.build_report`. Business
logic lives in those modules so it stays testable without going through argv.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import typer

from traceval.agents import Agent
from traceval.agents.live import LiveAgent
from traceval.agents.oracle import OracleAgent
from traceval.agents.replay import ReplayAgent
from traceval.providers import Provider, ResolvedModel
from traceval.providers.anthropic_provider import AnthropicProvider
from traceval.providers.mock_provider import MockProvider
from traceval.providers.openai_provider import OpenAIProvider
from traceval.providers.registry import ModelRegistry
from traceval.runner import run_task
from traceval.scoring import ExactMatchScorer, ModelGradedScorer, RubricScorer, Scorer, build_report
from traceval.scoring.report import TaskSetReport
from traceval.tasks import Task, load_task
from traceval.trace import (
    AgentKind,
    NoPassingTraceFoundError,
    Trace,
    diff_traces,
    find_last_passing_trace,
    read_trace,
)

app = typer.Typer(help="Deterministic task runner and scoring harness for agent rollouts.")


def _build_provider(name: str, registry: ModelRegistry) -> Provider:
    if name == "anthropic":
        return AnthropicProvider(registry)
    if name == "openai":
        return OpenAIProvider(registry)
    if name == "mock":
        return MockProvider()
    raise typer.BadParameter(f"unknown provider: {name!r} (expected anthropic|openai|mock)")


def _build_agent(
    agent_kind: AgentKind,
    task: Task,
    replay_trace: Path | None,
    provider_instance: Provider,
    resolved_model: ResolvedModel,
) -> Agent:
    if agent_kind == AgentKind.ORACLE:
        script_path = task.task_dir / "scripted_trajectory.jsonl"
        if not script_path.exists():
            raise typer.BadParameter(f"oracle agent requires {script_path}")
        return OracleAgent(script_path)
    if agent_kind == AgentKind.REPLAY:
        if replay_trace is None:
            raise typer.BadParameter("--replay-trace is required when --agent replay")
        return ReplayAgent(replay_trace)
    return LiveAgent(provider_instance, resolved_model, goal=task.goal)


def _build_scorers(
    task: Task,
    judge_provider_instance: Provider | None,
    judge_resolved: ResolvedModel | None,
) -> list[Scorer]:
    scorers: list[Scorer] = []
    for config in task.scorers:
        if config.kind == "exact_match":
            scorers.append(ExactMatchScorer())
        elif config.kind == "rubric":
            scorers.append(RubricScorer())
        elif config.kind == "model_graded":
            if judge_provider_instance is None or judge_resolved is None:
                raise typer.BadParameter(
                    f"task {task.id!r} configures a model_graded scorer; "
                    "pass --judge-provider/--judge-model"
                )
            scorers.append(ModelGradedScorer(judge_provider_instance, judge_resolved))
        else:
            raise typer.BadParameter(f"unknown scorer kind: {config.kind!r}")
    return scorers


def _discover_task_dirs(path: Path) -> list[Path]:
    if (path / "task.yaml").exists():
        return [path]
    task_dirs = sorted(p for p in path.iterdir() if p.is_dir() and (p / "task.yaml").exists())
    if not task_dirs:
        raise typer.BadParameter(f"no task.yaml found at {path} or in its subdirectories")
    return task_dirs


def _print_report(report: TaskSetReport) -> None:
    typer.echo(
        f"runs: {report.run_count} "
        f"(success={report.success_count} failure={report.failure_count} "
        f"error={report.error_count})"
    )
    for name in sorted(report.accuracy_by_scorer):
        accuracy = report.accuracy_by_scorer[name]
        mean_score = report.mean_score_by_scorer[name]
        typer.echo(f"  {name}: accuracy={accuracy:.0%} mean_score={mean_score:.3f}")
    typer.echo(
        f"agent latency: p50={report.agent_latency_ms_p50:.1f}ms "
        f"p95={report.agent_latency_ms_p95:.1f}ms"
    )
    typer.echo(
        f"env latency: p50={report.env_latency_ms_p50:.1f}ms p95={report.env_latency_ms_p95:.1f}ms"
    )
    typer.echo(f"tokens: input={report.total_input_tokens} output={report.total_output_tokens}")
    if report.total_cost_usd is not None:
        typer.echo(f"cost: ${report.total_cost_usd:.4f}")
    else:
        typer.echo("cost: unknown (model not in pricing.yaml)")


@app.command()
def run(
    task_path: Path = typer.Argument(
        ..., help="A task directory (contains task.yaml) or a directory of task directories"
    ),
    agent: AgentKind = typer.Option(AgentKind.ORACLE, "--agent", help="oracle|replay|live"),
    provider: str = typer.Option("mock", "--provider"),
    model: str = typer.Option("mock-1", "--model"),
    judge_provider: str | None = typer.Option(None, "--judge-provider"),
    judge_model: str | None = typer.Option(None, "--judge-model"),
    trace_dir: Path = typer.Option(Path("runs"), "--trace-dir"),
    replay_trace: Path | None = typer.Option(
        None, "--replay-trace", help="Required when --agent replay"
    ),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Override task.max_steps"),
    exit_zero_on_error: bool = typer.Option(
        False,
        "--exit-zero-on-error",
        help="Exit 0 even if any task errored (default: exit nonzero on any error)",
    ),
) -> None:
    """Run every task under TASK_PATH and print an accuracy/latency/cost report."""
    registry = ModelRegistry.load()
    provider_instance = _build_provider(provider, registry)
    resolved_model = provider_instance.resolve_model(model)

    judge_provider_instance: Provider | None = None
    judge_resolved: ResolvedModel | None = None
    if judge_model is not None:
        judge_provider_instance = _build_provider(judge_provider or provider, registry)
        judge_resolved = judge_provider_instance.resolve_model(judge_model)

    traces: list[Trace] = []
    untraced_error_count = 0
    for task_dir in _discover_task_dirs(task_path):
        task = load_task(task_dir)
        if max_steps is not None:
            task.max_steps = max_steps
        if task.requires_live_judge and (
            judge_provider_instance is None or isinstance(judge_provider_instance, MockProvider)
        ):
            # A mock judge can only ever rubber-stamp a canned verdict, so
            # running this task against one wouldn't test anything; skip it
            # rather than either faking a pass or erroring the batch on a
            # missing --judge-provider. Not counted in the report at all
            # (no trace exists), unlike an actual error.
            typer.echo(f"{task.id}: skipped (requires a real --judge-provider)")
            continue
        agent_instance = _build_agent(agent, task, replay_trace, provider_instance, resolved_model)
        scorers = _build_scorers(task, judge_provider_instance, judge_resolved)
        # A known run_id up front means the trace path is knowable even if
        # run_task raises, so one task erroring doesn't abort the rest of
        # the batch: run_task always writes a complete trace before
        # re-raising (see docs/architecture.md's runner error-handling
        # section), so we can read that trace back and move on.
        run_id = uuid.uuid4().hex
        trace_path = trace_dir / f"{run_id}.jsonl"
        try:
            trace = run_task(
                task=task,
                agent=agent_instance,
                agent_kind=agent,
                model_under_test=resolved_model,
                scorers=scorers,
                trace_dir=trace_dir,
                judge_model=judge_resolved,
                run_id=run_id,
            )
        except Exception as exc:
            try:
                trace = read_trace(trace_path)
            except FileNotFoundError:
                # The failure happened before run_task ever opened its
                # TraceWriter (e.g. an unknown environment kind), so no trace
                # exists to fall back to. Report and move on regardless.
                typer.echo(
                    f"{task.id}: error ({type(exc).__name__}: {exc}) -> no trace written",
                    err=True,
                )
                untraced_error_count += 1
                continue
            typer.echo(f"{task.id}: error ({type(exc).__name__}: {exc}) -> {trace_path}", err=True)
        else:
            outcome = trace.footer.outcome.value if trace.footer else "unknown"
            typer.echo(f"{task.id}: {outcome} -> {trace_dir / (trace.header.run_id + '.jsonl')}")
        traces.append(trace)

    report = build_report(traces)
    _print_report(report)

    if untraced_error_count > 0:
        typer.echo(f"{untraced_error_count} task(s) errored before a trace could be written.")
    if (report.error_count > 0 or untraced_error_count > 0) and not exit_zero_on_error:
        raise typer.Exit(code=1)


@app.command()
def replay(
    trace_path: Path = typer.Argument(..., help="Trace file to replay"),
    task_path: Path = typer.Argument(..., help="Task directory the trace was produced from"),
    judge_provider: str | None = typer.Option(None, "--judge-provider"),
    judge_model: str | None = typer.Option(None, "--judge-model"),
    trace_dir: Path = typer.Option(Path("runs"), "--trace-dir"),
) -> None:
    """Re-run a task using a previously recorded trace's exact action sequence."""
    task = load_task(task_path)
    original = read_trace(trace_path)
    registry = ModelRegistry.load()

    judge_provider_instance: Provider | None = None
    judge_resolved: ResolvedModel | None = None
    if judge_model is not None:
        judge_provider_instance = _build_provider(judge_provider or "mock", registry)
        judge_resolved = judge_provider_instance.resolve_model(judge_model)

    scorers = _build_scorers(task, judge_provider_instance, judge_resolved)
    trace = run_task(
        task=task,
        agent=ReplayAgent(trace_path),
        agent_kind=AgentKind.REPLAY,
        model_under_test=original.header.model_under_test,
        scorers=scorers,
        trace_dir=trace_dir,
        judge_model=judge_resolved,
    )
    typer.echo(f"replayed -> {trace_dir / (trace.header.run_id + '.jsonl')}")


@app.command()
def diff(
    trace_a: Path = typer.Argument(..., help="First trace (e.g. last passing run)"),
    trace_b: Path | None = typer.Argument(
        None,
        help="Second trace (e.g. a replayed failing run). Omit when using --against-last-passing.",
    ),
    allow_different_task: bool = typer.Option(
        False,
        "--allow-different-task",
        help="Diff traces from different tasks anyway (step alignment is meaningless otherwise)",
    ),
    against_last_passing: bool = typer.Option(
        False,
        "--against-last-passing",
        help="Diff trace_a against the most recent Outcome.SUCCESS trace in its directory "
        "with a matching task_hash, instead of an explicit trace_b.",
    ),
) -> None:
    """Show the structured, step-aligned diff between two traces."""
    a = read_trace(trace_a)

    if against_last_passing:
        if trace_b is not None:
            raise typer.BadParameter("pass either trace_b or --against-last-passing, not both")
        try:
            trace_b = find_last_passing_trace(trace_a, a.header.task_hash)
        except NoPassingTraceFoundError as exc:
            raise typer.BadParameter(str(exc)) from exc
    elif trace_b is None:
        raise typer.BadParameter("trace_b is required unless --against-last-passing is given")

    b = read_trace(trace_b)
    result = diff_traces(a, b, a_path=str(trace_a), b_path=str(trace_b))

    if not result.same_task and not allow_different_task:
        raise typer.BadParameter(
            f"trace_a is task {a.header.task_id!r} ({a.header.task_hash}) but trace_b is "
            f"task {b.header.task_id!r} ({b.header.task_hash}); step-level divergence is "
            "meaningless across different tasks. Pass --allow-different-task to diff anyway."
        )
    if not result.same_seed:
        typer.echo(
            f"WARNING: traces used different seeds (trace_a={a.header.seed}, "
            f"trace_b={b.header.seed}). Divergence may be an expected consequence of the "
            "seed change, not a regression.",
            err=True,
        )

    if result.header_delta.has_changes:
        typer.echo("Header changes (checked before step-level divergence):")
        delta = result.header_delta
        if delta.model_under_test_changed:
            typer.echo(
                f"  model_under_test: {delta.model_under_test_a.model_id} -> "
                f"{delta.model_under_test_b.model_id}"
            )
        if delta.judge_model_changed:
            judge_a = delta.judge_model_a.model_id if delta.judge_model_a else None
            judge_b = delta.judge_model_b.model_id if delta.judge_model_b else None
            typer.echo(f"  judge_model: {judge_a} -> {judge_b}")
        if delta.environment_fingerprint_changed:
            typer.echo(
                f"  environment_fingerprint: {delta.environment_fingerprint_a} -> "
                f"{delta.environment_fingerprint_b}"
            )
        if delta.task_hash_changed:
            typer.echo(f"  task_hash: {delta.task_hash_a} -> {delta.task_hash_b}")
        typer.echo("")

    typer.echo(result.model_dump_json(indent=2))


@app.command()
def report(
    run_dir: Path = typer.Argument(Path("runs"), help="Directory of trace .jsonl files"),
) -> None:
    """Aggregate accuracy, latency, and cost across every trace in RUN_DIR."""
    trace_paths = sorted(run_dir.glob("*.jsonl"))
    if not trace_paths:
        raise typer.BadParameter(f"no trace files found in {run_dir}")
    traces = [read_trace(p) for p in trace_paths]
    _print_report(build_report(traces))


__all__ = ["app"]
