from datetime import UTC, datetime
from pathlib import Path

import pytest

from traceval.providers import ResolvedModel
from traceval.providers.mock_provider import MockProvider
from traceval.scoring import (
    ExactMatchScorer,
    ModelGradedScorer,
    RubricScorer,
    build_report,
)
from traceval.scoring.report import PricingTable
from traceval.scoring.rubric import UnsupportedRubricCheckError
from traceval.tasks import load_task
from traceval.trace import (
    AgentKind,
    Outcome,
    PricingSnapshot,
    ScoreResult,
    Trace,
    TraceFooter,
    TraceHeader,
    TraceStep,
    TraceTotals,
)

EXAMPLE_TASK_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "tasks" / "example_search_task"
)


def _model(model_id: str = "mock-1-20260101") -> ResolvedModel:
    return ResolvedModel(provider="mock", model_id=model_id, alias="mock-1")


def _header(**overrides: object) -> TraceHeader:
    defaults: dict[str, object] = dict(
        run_id="run-1",
        task_id="example_search_task",
        task_hash="sha256:x",
        task_format_version=1,
        seed=1,
        model_under_test=_model(),
        agent_kind=AgentKind.ORACLE,
        environment_fingerprint="sha256:y",
        traceval_version="0.1.0-test",
        started_at=datetime.now(UTC),
    )
    defaults.update(overrides)
    return TraceHeader(**defaults)  # type: ignore[arg-type]


def _step(index: int, action: dict[str, object], elements: dict[str, str]) -> TraceStep:
    return TraceStep(
        index=index,
        observation_hash=f"sha256:o{index}",
        observation={"url": "file:///x", "title": "t", "elements": elements},
        action=action,
        input_tokens=5,
        output_tokens=2,
        agent_latency_ms=8.0,
        env_latency_ms=2.0,
        timestamp=datetime.now(UTC),
    )


def _passing_trace() -> Trace:
    return Trace(
        header=_header(),
        steps=[
            _step(0, {"kind": "type", "target": "#query", "value": "playwright"}, {"#query": ""}),
            _step(1, {"kind": "click", "target": "#search"}, {"#result": "found: playwright"}),
        ],
    )


def test_exact_match_scorer_pass() -> None:
    task = load_task(EXAMPLE_TASK_DIR)
    scorer = ExactMatchScorer()
    result = scorer.score(task, _passing_trace())
    assert result.passed is True
    assert result.score == 1.0


def test_exact_match_scorer_fail() -> None:
    task = load_task(EXAMPLE_TASK_DIR)
    trace = Trace(
        header=_header(),
        steps=[_step(0, {"kind": "click", "target": "#search"}, {"#result": "nope"})],
    )
    scorer = ExactMatchScorer()
    result = scorer.score(task, trace)
    assert result.passed is False
    assert result.score == 0.0


def test_rubric_scorer_all_criteria_pass() -> None:
    task = load_task(EXAMPLE_TASK_DIR)
    scorer = RubricScorer()
    result = scorer.score(task, _passing_trace())
    assert result.passed is True
    assert result.score == 1.0


def test_rubric_scorer_partial_criteria() -> None:
    task = load_task(EXAMPLE_TASK_DIR)
    trace = Trace(
        header=_header(),
        steps=[_step(0, {"kind": "type", "target": "#query", "value": "playwright"}, {})],
    )
    scorer = RubricScorer()
    result = scorer.score(task, trace)
    assert result.passed is False
    assert result.score == 0.5


def test_rubric_scorer_unsupported_check_raises() -> None:
    task = load_task(EXAMPLE_TASK_DIR)
    task.scorers[1].config["criteria"][0]["check"] = "nonsense"
    scorer = RubricScorer()
    with pytest.raises(UnsupportedRubricCheckError):
        scorer.score(task, _passing_trace())


def test_model_graded_scorer_uses_mock_judge_pass() -> None:
    task = load_task(EXAMPLE_TASK_DIR)
    task.scorers.append(
        type(task.scorers[0])(
            kind="model_graded", config={"rubric_prompt": "Did the agent search correctly?"}
        )
    )
    judge = MockProvider(canned_response="PASS\nThe agent typed the query and clicked search.")
    scorer = ModelGradedScorer(judge, _model("mock-judge-1-20260101"))
    result = scorer.score(task, _passing_trace())
    assert result.passed is True
    assert result.details["judge_model_id"] == "mock-judge-1-20260101"


def test_model_graded_scorer_uses_mock_judge_fail() -> None:
    task = load_task(EXAMPLE_TASK_DIR)
    task.scorers.append(
        type(task.scorers[0])(kind="model_graded", config={"rubric_prompt": "grade this"})
    )
    judge = MockProvider(canned_response="FAIL\nMissing steps.")
    scorer = ModelGradedScorer(judge, _model("mock-judge-1-20260101"))
    result = scorer.score(task, _passing_trace())
    assert result.passed is False


def _footer(passed: bool, input_tokens: int = 10, output_tokens: int = 5) -> TraceFooter:
    return TraceFooter(
        outcome=Outcome.SUCCESS if passed else Outcome.FAILURE,
        total_steps=1,
        scores=[ScoreResult(scorer="exact_match", score=1.0 if passed else 0.0, passed=passed)],
        totals=TraceTotals(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            agent_latency_ms=80.0,
            env_latency_ms=20.0,
        ),
        ended_at=datetime.now(UTC),
    )


def test_report_aggregates_accuracy_and_cost(tmp_path: Path) -> None:
    pricing_path = tmp_path / "pricing.yaml"
    pricing_path.write_text(
        "mock:\n"
        "  mock-1-20260101:\n"
        "    input_per_million_usd: 10.0\n"
        "    output_per_million_usd: 20.0\n"
    )
    pricing = PricingTable.load(pricing_path)

    trace_a = Trace(header=_header(), steps=[], footer=_footer(True, 10, 5))
    trace_b = Trace(header=_header(), steps=[], footer=_footer(False, 10, 5))

    report = build_report([trace_a, trace_b], pricing=pricing)
    assert report.run_count == 2
    assert report.accuracy_by_scorer["exact_match"] == 0.5
    assert report.mean_score_by_scorer["exact_match"] == 0.5
    assert report.total_input_tokens == 20
    assert report.total_output_tokens == 10
    # (20/1e6 * 10) + (10/1e6 * 20) = 0.0002 + 0.0002
    assert report.total_cost_usd == pytest.approx(0.0004)


def test_report_cost_unknown_when_model_unpriced(tmp_path: Path) -> None:
    pricing_path = tmp_path / "pricing.yaml"
    pricing_path.write_text("anthropic: {}\n")
    pricing = PricingTable.load(pricing_path)

    trace = Trace(header=_header(), steps=[], footer=_footer(True))
    report = build_report([trace], pricing=pricing)
    assert report.total_cost_usd is None


def test_report_cost_uses_stamped_pricing_snapshot_not_live_pricing_table(
    tmp_path: Path,
) -> None:
    """A trace's header pins the rates that applied when it ran. Replaying it
    after pricing.yaml changes must still report the original cost, not one
    computed from the now-current (different) live table.
    """
    pricing_path = tmp_path / "pricing.yaml"
    pricing_path.write_text(
        "mock:\n"
        "  mock-1-20260101:\n"
        "    input_per_million_usd: 999.0\n"
        "    output_per_million_usd: 999.0\n"
    )
    live_pricing_now = PricingTable.load(pricing_path)

    header = _header(
        pricing_snapshot=PricingSnapshot(input_per_million_usd=10.0, output_per_million_usd=20.0)
    )
    trace = Trace(header=header, steps=[], footer=_footer(True, input_tokens=10, output_tokens=5))

    report = build_report([trace], pricing=live_pricing_now)
    # (10/1e6 * 10) + (5/1e6 * 20) = 0.0001 + 0.0001, from the stamped
    # snapshot, not live_pricing_now's 999.0 rates.
    assert report.total_cost_usd == pytest.approx(0.0002)
