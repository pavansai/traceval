"""ModelGradedScorer: delegates judgment to a judge Provider/model.

The judge is configured and pinned independently of the model under test.
This scorer is constructed with its own `Provider` + `ResolvedModel`, never
reusing whatever model produced the trajectory being graded.

Config (from task.yaml):
    rubric_prompt: grading instructions given to the judge, e.g. "Did the
        agent search for 'playwright' and see a result? Answer PASS or FAIL
        on the first line, then a one-sentence rationale."
"""

from __future__ import annotations

from traceval.providers import Message, Provider, ResolvedModel
from traceval.scoring import find_scorer_config
from traceval.tasks import Task
from traceval.trace import ScoreResult, Trace


class JudgeResponseError(ValueError):
    """Raised when the judge's response doesn't start with PASS or FAIL."""


class ModelGradedScorer:
    def __init__(self, provider: Provider, judge_model: ResolvedModel) -> None:
        self._provider = provider
        self._judge_model = judge_model

    def score(self, task: Task, trace: Trace) -> ScoreResult:
        config = find_scorer_config(task, "model_graded")
        rubric_prompt = config["rubric_prompt"]

        transcript = self._render_transcript(trace)
        prompt = (
            f"{rubric_prompt}\n\n"
            "Respond with PASS or FAIL on the first line, then a one-sentence "
            f"rationale.\n\nTranscript:\n{transcript}"
        )
        response = self._provider.generate(
            self._judge_model, [Message(role="user", content=prompt)]
        )
        verdict, rationale = self._parse_verdict(response.content)

        return ScoreResult(
            scorer="model_graded",
            score=1.0 if verdict else 0.0,
            passed=verdict,
            details={
                "judge_provider": self._judge_model.provider,
                "judge_model_id": self._judge_model.model_id,
                "rationale": rationale,
            },
        )

    @staticmethod
    def _render_transcript(trace: Trace) -> str:
        lines = [
            f"step {step.index}: action={step.action} observation={step.observation}"
            for step in trace.steps
        ]
        return "\n".join(lines) or "(no steps taken)"

    @staticmethod
    def _parse_verdict(content: str) -> tuple[bool, str]:
        lines = content.strip().splitlines()
        if not lines:
            raise JudgeResponseError("judge returned an empty response")
        first_line = lines[0].strip().upper()
        if first_line.startswith("PASS"):
            verdict = True
        elif first_line.startswith("FAIL"):
            verdict = False
        else:
            raise JudgeResponseError(f"judge response did not start with PASS/FAIL: {content!r}")
        rationale = "\n".join(lines[1:]).strip()
        return verdict, rationale


__all__ = ["JudgeResponseError", "ModelGradedScorer"]
