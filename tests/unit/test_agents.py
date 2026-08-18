from datetime import UTC, datetime
from pathlib import Path

import pytest

from traceval.agents.live import LiveAgent
from traceval.agents.oracle import OracleAgent, ScriptedTrajectoryDivergedError
from traceval.agents.replay import ReplayAgent
from traceval.environments import Observation
from traceval.providers import ResolvedModel
from traceval.providers.mock_provider import MockProvider
from traceval.trace import AgentKind, TraceHeader, TraceStep, TraceWriter

EXAMPLE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "tasks"
    / "example_search_task"
    / "scripted_trajectory.jsonl"
)


def test_oracle_agent_replays_scripted_actions() -> None:
    agent = OracleAgent(EXAMPLE_SCRIPT)

    obs1 = Observation(url="file:///x", title="t", elements={"#query": "", "#result": ""})
    step1 = agent.act(obs1, [])
    assert step1.action is not None
    assert step1.action.kind == "type"
    assert step1.action.value == "playwright"
    assert step1.usage.input_tokens == 0
    assert step1.usage.output_tokens == 0

    obs2 = Observation(url="file:///x", title="t", elements={"#query": "playwright", "#result": ""})
    step2 = agent.act(obs2, [])
    assert step2.action is not None
    assert step2.action.kind == "click"

    assert agent.act(obs2, []).action is None


def test_oracle_agent_raises_on_divergence() -> None:
    agent = OracleAgent(EXAMPLE_SCRIPT)
    bad_observation = Observation(
        url="file:///x", title="t", elements={"#query": "unexpected", "#result": ""}
    )
    with pytest.raises(ScriptedTrajectoryDivergedError):
        agent.act(bad_observation, [])


def _write_trace(path: Path) -> None:
    header = TraceHeader(
        run_id="run-1",
        task_id="example_search_task",
        task_hash="sha256:x",
        task_format_version=1,
        seed=1,
        model_under_test=ResolvedModel(provider="mock", model_id="mock-1-x", alias="mock-1"),
        agent_kind=AgentKind.ORACLE,
        environment_fingerprint="sha256:y",
        traceval_version="0.1.0-test",
        started_at=datetime.now(UTC),
    )
    with TraceWriter(path) as writer:
        writer.write_header(header)
        writer.write_step(
            TraceStep(
                index=0,
                observation_hash="sha256:o0",
                action={"kind": "type", "target": "#query", "value": "playwright"},
                timestamp=datetime.now(UTC),
            )
        )
        writer.write_step(
            TraceStep(
                index=1,
                observation_hash="sha256:o1",
                action={"kind": "click", "target": "#search"},
                timestamp=datetime.now(UTC),
            )
        )


def test_replay_agent_replays_recorded_actions(tmp_path: Path) -> None:
    trace_path = tmp_path / "run.jsonl"
    _write_trace(trace_path)

    agent = ReplayAgent(trace_path)
    assert agent.source_task_id == "example_search_task"
    assert agent.source_seed == 1

    obs = Observation(url="file:///x", title="t", elements={})
    step1 = agent.act(obs, [])
    assert step1.action is not None and step1.action.kind == "type"
    step2 = agent.act(obs, [])
    assert step2.action is not None and step2.action.kind == "click"
    assert agent.act(obs, []).action is None


def _live_model() -> ResolvedModel:
    return ResolvedModel(provider="mock", model_id="mock-1-20260101", alias="mock-1")


def test_live_agent_strips_json_language_tagged_code_fence() -> None:
    provider = MockProvider(canned_response='```json\n{"kind": "click", "target": "#x"}\n```')
    agent = LiveAgent(provider, _live_model(), goal="Do the thing.")
    step = agent.act(Observation(url="file:///x", title="t", elements={}), [])
    assert step.action is not None
    assert step.action.kind == "click"
    assert step.action.target == "#x"


def test_live_agent_strips_untagged_code_fence() -> None:
    provider = MockProvider(canned_response='```\n{"kind": "wait", "ms": 0}\n```')
    agent = LiveAgent(provider, _live_model(), goal="Do the thing.")
    step = agent.act(Observation(url="file:///x", title="t", elements={}), [])
    assert step.action is not None
    assert step.action.kind == "wait"


def test_live_agent_strips_fence_around_done() -> None:
    provider = MockProvider(canned_response="```\nDONE\n```")
    agent = LiveAgent(provider, _live_model(), goal="Do the thing.")
    step = agent.act(Observation(url="file:///x", title="t", elements={}), [])
    assert step.action is None


def test_live_agent_parses_unfenced_action_unchanged() -> None:
    provider = MockProvider(canned_response='{"kind": "wait", "ms": 0}')
    agent = LiveAgent(provider, _live_model(), goal="Do the thing.")
    step = agent.act(Observation(url="file:///x", title="t", elements={}), [])
    assert step.action is not None
    assert step.action.kind == "wait"
