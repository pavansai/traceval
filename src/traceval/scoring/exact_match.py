"""ExactMatchScorer: compares the final observation's value at a selector to
the scorer config's `expected` value.

Config (from task.yaml's `scorers: [{kind: exact_match, config: {...}}]`):
    target: selector key into the final step's Observation.elements
    expected: the exact string that selector must equal
"""

from __future__ import annotations

from traceval.scoring import find_scorer_config
from traceval.tasks import Task
from traceval.trace import ScoreResult, Trace


class ExactMatchScorer:
    def score(self, task: Task, trace: Trace) -> ScoreResult:
        config = find_scorer_config(task, "exact_match")
        target = config["target"]
        expected = config["expected"]

        actual = (
            trace.steps[-1].observation.get("elements", {}).get(target) if trace.steps else None
        )
        passed = actual == expected
        return ScoreResult(
            scorer="exact_match",
            score=1.0 if passed else 0.0,
            passed=passed,
            details={"target": target, "expected": expected, "actual": actual},
        )


__all__ = ["ExactMatchScorer"]
