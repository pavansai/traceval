"""Aggregates accuracy, latency, and token cost across a task set's traces.

Cost prefers each trace's stamped `TraceHeader.pricing_snapshot` (the rates
resolved from `pricing.yaml` at run time), so a report built from a replayed
trace keeps reporting the cost that actually applied then, even after
`pricing.yaml` has since changed. Only traces without a snapshot (e.g. from
before this field existed) fall back to a live lookup in the pinned pricing
table (`pricing.yaml`), keyed by provider + the resolved `model_id`, the
same value stamped into the trace header, never an alias. A model absent
from the pricing table (e.g. the `mock` provider with no entry, or a
not-yet-priced model) makes `total_cost_usd` `None` rather than silently
reporting a wrong number.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from traceval.paths import find_repo_file
from traceval.trace import Trace


class ModelPricing(BaseModel):
    input_per_million_usd: float
    output_per_million_usd: float


def find_default_pricing_path(start: Path | None = None) -> Path:
    return find_repo_file("pricing.yaml", start=start)


def _cost_usd(
    *, input_rate: float, output_rate: float, input_tokens: int, output_tokens: int
) -> float:
    return input_tokens / 1_000_000 * input_rate + output_tokens / 1_000_000 * output_rate


class PricingTable:
    def __init__(self, mapping: dict[str, dict[str, dict[str, float]]]) -> None:
        self._mapping = mapping

    @classmethod
    def load(cls, path: Path | None = None) -> PricingTable:
        pricing_path = path or find_default_pricing_path()
        data = yaml.safe_load(pricing_path.read_text(encoding="utf-8")) or {}
        return cls(data)

    def rates(self, provider: str, model_id: str) -> ModelPricing | None:
        entry = self._mapping.get(provider, {}).get(model_id)
        return ModelPricing(**entry) if entry is not None else None

    def cost_usd(
        self, provider: str, model_id: str, input_tokens: int, output_tokens: int
    ) -> float | None:
        rate = self.rates(provider, model_id)
        if rate is None:
            return None
        return _cost_usd(
            input_rate=rate.input_per_million_usd,
            output_rate=rate.output_per_million_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class TaskSetReport(BaseModel):
    run_count: int
    accuracy_by_scorer: dict[str, float]
    mean_score_by_scorer: dict[str, float]
    agent_latency_ms_p50: float
    agent_latency_ms_p95: float
    env_latency_ms_p50: float
    env_latency_ms_p95: float
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float | None


def build_report(traces: list[Trace], pricing: PricingTable | None = None) -> TaskSetReport:
    pricing = pricing or PricingTable.load()

    scorer_pass_counts: dict[str, int] = {}
    scorer_score_totals: dict[str, float] = {}
    scorer_counts: dict[str, int] = {}
    agent_latencies: list[float] = []
    env_latencies: list[float] = []
    total_input = 0
    total_output = 0
    total_cost = 0.0
    any_cost_known = False

    for trace in traces:
        if trace.footer is None:
            continue
        agent_latencies.append(trace.footer.totals.agent_latency_ms)
        env_latencies.append(trace.footer.totals.env_latency_ms)
        total_input += trace.footer.totals.input_tokens
        total_output += trace.footer.totals.output_tokens

        snapshot = trace.header.pricing_snapshot
        if snapshot is not None:
            cost: float | None = _cost_usd(
                input_rate=snapshot.input_per_million_usd,
                output_rate=snapshot.output_per_million_usd,
                input_tokens=trace.footer.totals.input_tokens,
                output_tokens=trace.footer.totals.output_tokens,
            )
        else:
            model = trace.header.model_under_test
            cost = pricing.cost_usd(
                model.provider,
                model.model_id,
                trace.footer.totals.input_tokens,
                trace.footer.totals.output_tokens,
            )
        if cost is not None:
            total_cost += cost
            any_cost_known = True

        for score in trace.footer.scores:
            scorer_counts[score.scorer] = scorer_counts.get(score.scorer, 0) + 1
            scorer_score_totals[score.scorer] = (
                scorer_score_totals.get(score.scorer, 0.0) + score.score
            )
            if score.passed:
                scorer_pass_counts[score.scorer] = scorer_pass_counts.get(score.scorer, 0) + 1

    sorted_agent_latencies = sorted(agent_latencies)
    sorted_env_latencies = sorted(env_latencies)
    return TaskSetReport(
        run_count=len(traces),
        accuracy_by_scorer={
            name: scorer_pass_counts.get(name, 0) / count for name, count in scorer_counts.items()
        },
        mean_score_by_scorer={
            name: scorer_score_totals[name] / scorer_counts[name] for name in scorer_counts
        },
        agent_latency_ms_p50=_percentile(sorted_agent_latencies, 0.50),
        agent_latency_ms_p95=_percentile(sorted_agent_latencies, 0.95),
        env_latency_ms_p50=_percentile(sorted_env_latencies, 0.50),
        env_latency_ms_p95=_percentile(sorted_env_latencies, 0.95),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cost_usd=total_cost if any_cost_known else None,
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, round(fraction * (len(sorted_values) - 1)))
    return sorted_values[index]


__all__ = [
    "ModelPricing",
    "PricingTable",
    "TaskSetReport",
    "build_report",
    "find_default_pricing_path",
]
