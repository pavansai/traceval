"""RubricScorer: a weighted, deterministic checklist scorer.

No LLM involved: each criterion is a predicate evaluated against the trace's
step history. Distinct from `ModelGradedScorer`, which delegates judgment to
a judge model.

Config (from task.yaml):
    criteria: list of {name, weight, check, ...check-specific fields}
    pass_threshold: float in [0, 1], default 1.0 (require every criterion)

Supported checks:
    action_occurred: {action_kind, target?, value_contains?}
        Passes if any step's action matches action_kind (and target/
        value_contains when given).
"""

from __future__ import annotations

from typing import Any

from traceval.scoring import find_scorer_config
from traceval.tasks import Task
from traceval.trace import ScoreResult, Trace


class UnsupportedRubricCheckError(ValueError):
    pass


class RubricScorer:
    def score(self, task: Task, trace: Trace) -> ScoreResult:
        config = find_scorer_config(task, "rubric")
        criteria: list[dict[str, Any]] = config["criteria"]  # type: ignore[assignment]
        pass_threshold: float = config.get("pass_threshold", 1.0)  # type: ignore[assignment]

        results = [
            (
                criterion["name"],
                float(criterion.get("weight", 1.0)),
                self._evaluate(criterion, trace),
            )
            for criterion in criteria
        ]

        total_weight = sum(w for _, w, _ in results) or 1.0
        earned_weight = sum(w for _, w, ok in results if ok)
        score = earned_weight / total_weight
        passed = score >= pass_threshold

        return ScoreResult(
            scorer="rubric",
            score=score,
            passed=passed,
            details={"criteria": [{"name": n, "weight": w, "passed": ok} for n, w, ok in results]},
        )

    def _evaluate(self, criterion: dict[str, Any], trace: Trace) -> bool:
        check = criterion["check"]
        if check == "action_occurred":
            return self._action_occurred(criterion, trace)
        raise UnsupportedRubricCheckError(f"unsupported rubric check: {check!r}")

    @staticmethod
    def _action_occurred(criterion: dict[str, Any], trace: Trace) -> bool:
        action_kind = criterion["action_kind"]
        target = criterion.get("target")
        value_contains = criterion.get("value_contains")
        for step in trace.steps:
            action = step.action
            if action.get("kind") != action_kind:
                continue
            if target is not None and action.get("target") != target:
                continue
            if value_contains is not None and value_contains not in (action.get("value") or ""):
                continue
            return True
        return False


__all__ = ["RubricScorer", "UnsupportedRubricCheckError"]
