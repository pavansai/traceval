"""Scorer protocol: grades a completed (footer-less) trace against its task.

Reuses `traceval.trace.ScoreResult` as the return type, since it's already
what gets stamped into the trace footer, so scorers write directly into the
trace schema rather than a parallel result type.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from traceval.tasks import Task
from traceval.trace import ScoreResult, Trace


@runtime_checkable
class Scorer(Protocol):
    def score(self, task: Task, trace: Trace) -> ScoreResult: ...


def find_scorer_config(task: Task, kind: str) -> dict[str, object]:
    for scorer_config in task.scorers:
        if scorer_config.kind == kind:
            return scorer_config.config
    raise ValueError(f"task {task.id!r} has no {kind!r} scorer configured")


# Imported after find_scorer_config is defined above: each of these submodules
# does `from traceval.scoring import find_scorer_config` at its own top level.
from traceval.scoring.exact_match import ExactMatchScorer  # noqa: E402
from traceval.scoring.model_graded import JudgeResponseError, ModelGradedScorer  # noqa: E402
from traceval.scoring.report import (  # noqa: E402
    PricingTable,
    TaskSetReport,
    build_report,
)
from traceval.scoring.rubric import RubricScorer, UnsupportedRubricCheckError  # noqa: E402

__all__ = [
    "ExactMatchScorer",
    "JudgeResponseError",
    "ModelGradedScorer",
    "PricingTable",
    "RubricScorer",
    "ScoreResult",
    "Scorer",
    "TaskSetReport",
    "UnsupportedRubricCheckError",
    "build_report",
    "find_scorer_config",
]
