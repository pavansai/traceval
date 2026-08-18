#!/usr/bin/env python3
"""Manual, opt-in smoke check against a real Anthropic model and judge.

Exercises the one path nothing else in this repo runs: LiveAgent driving a
real model, and ModelGradedScorer judged by a real model. Deliberately not a
pytest test, so it never runs in CI (CONTRIBUTING.md's "no test may call a
real model provider" rule); this is the escape hatch for when you actually
want to. Run by hand:

    export ANTHROPIC_API_KEY=sk-ant-...
    uv run python scripts/smoke_live.py [--model claude-haiku-4-5]

Never writes ANTHROPIC_API_KEY anywhere; AnthropicProvider reads it lazily
from the environment on first real API call. Prints the trace's real
usage/cost/latency numbers and states plainly whether auth, token-usage
parsing, cost computation, agent latency, and judge-verdict parsing each
worked, since this path has never executed before and any of them could be
broken.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from traceval.agents.live import LiveAgent
from traceval.providers.anthropic_provider import AnthropicProvider
from traceval.providers.registry import ModelRegistry
from traceval.runner import run_task
from traceval.scoring import ModelGradedScorer, build_report
from traceval.tasks import load_task
from traceval.trace import AgentKind

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "tasks" / "feedback_form"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-haiku-4-5", help="Pinned alias from models.yaml")
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY is not set in the environment.", file=sys.stderr)
        return 1

    registry = ModelRegistry.load()
    provider = AnthropicProvider(registry)
    resolved_model = provider.resolve_model(args.model)

    task = load_task(TASK_DIR)
    agent = LiveAgent(provider, resolved_model)
    judge = ModelGradedScorer(provider, resolved_model)

    print(f"task: {task.id}")
    print(f"model: {resolved_model.provider}/{resolved_model.model_id}")
    print("running via LiveAgent (real model acting) + ModelGradedScorer (real judge)...")
    print()

    try:
        trace = run_task(
            task=task,
            agent=agent,
            agent_kind=AgentKind.LIVE,
            model_under_test=resolved_model,
            scorers=[judge],
            trace_dir=REPO_ROOT / "runs",
            judge_model=resolved_model,
        )
    except Exception as exc:
        print(f"run_task RAISED: {type(exc).__name__}: {exc}")
        print()
        print("=== verdict ===")
        print("auth: UNKNOWN, see exception above")
        print("usage parsing: FAIL, run never completed")
        print("cost computation: FAIL, run never completed")
        print("agent latency: FAIL, run never completed")
        print("judge parsing: FAIL, run never completed")
        return 1

    footer = trace.footer
    assert footer is not None
    report = build_report([trace])

    print("=== raw results ===")
    print(f"outcome: {footer.outcome.value}")
    print(f"steps: {len(trace.steps)}")
    print(f"agent input_tokens (trace totals): {footer.totals.input_tokens}")
    print(f"agent output_tokens (trace totals): {footer.totals.output_tokens}")
    print(f"agent_latency_ms total: {footer.totals.agent_latency_ms:.1f}")
    print(f"pricing_snapshot: {trace.header.pricing_snapshot}")
    print(f"total_cost_usd (report): {report.total_cost_usd}")
    print(f"scores: {[(s.scorer, s.passed, s.details) for s in footer.scores]}")
    print(
        "scorer_errors: "
        f"{[(e.scorer, e.error_type, e.error_message) for e in footer.scorer_errors]}"
    )
    if footer.error is not None:
        print(f"error: {footer.error.error_type}: {footer.error.error_message}")
    trace_path = REPO_ROOT / "runs" / f"{trace.header.run_id}.jsonl"
    print(f"trace written to: {trace_path}")
    print()

    has_tokens = footer.totals.input_tokens > 0 and footer.totals.output_tokens > 0
    has_cost = report.total_cost_usd is not None and report.total_cost_usd > 0
    has_latency = footer.totals.agent_latency_ms > 0
    judge_parsed = not footer.scorer_errors

    print("=== verdict ===")
    print("auth: YES, run_task completed without raising")
    print(f"usage parsing populated nonzero agent tokens: {'YES' if has_tokens else 'NO'}")
    print(f"cost computed to a nonzero dollar figure: {'YES' if has_cost else 'NO'}")
    print(f"agent latency came out nonzero: {'YES' if has_latency else 'NO'}")
    print(f"judge returned something the parser accepted: {'YES' if judge_parsed else 'NO'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
