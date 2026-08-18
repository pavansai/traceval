"""LiveAgent: the only agent that calls a real Provider.generate().

Deliberately minimal: a JSON-object-in-content action protocol rather than
native tool calling. This path is never exercised by the test suite (no test
may require an API key); it's covered only by the opt-in
`scripts/smoke_live.py`. Worth revisiting once a second real provider needs
native tool-use.
"""

from __future__ import annotations

import json
import re

from traceval.agents import AgentStep, AgentUsage
from traceval.environments import Action, Observation
from traceval.providers import Message, Provider, ResolvedModel
from traceval.trace import TraceStep

DEFAULT_SYSTEM_PROMPT = (
    "You control a browser via JSON actions. Reply with exactly one JSON object "
    'per turn, e.g. {"kind": "click", "target": "#search"} or '
    '{"kind": "type", "target": "#query", "value": "text"}, or the literal string '
    "DONE when the task is complete. Reply with nothing else."
)

# Real models routinely wrap requested JSON in a markdown code fence
# (```json ... ``` or ``` ... ```) even when told to reply with nothing
# else; this was never exercised before scripts/smoke_live.py first ran
# against a real model and hit it immediately. Only strips a fence that
# wraps the entire response, so it never touches a plain unfenced action.
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _strip_code_fence(content: str) -> str:
    match = _CODE_FENCE_RE.match(content)
    return match.group(1) if match else content


class LiveAgentResponseError(ValueError):
    """Raised when the model's response is neither DONE nor a valid Action."""


class LiveAgent:
    def __init__(
        self,
        provider: Provider,
        resolved_model: ResolvedModel,
        goal: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._provider = provider
        self._resolved_model = resolved_model
        self._goal = goal
        self._system_prompt = system_prompt

    def act(self, observation: Observation, history: list[TraceStep]) -> AgentStep:
        messages = self._build_messages(observation, history)
        response = self._provider.generate(self._resolved_model, messages)
        usage = AgentUsage(input_tokens=response.input_tokens, output_tokens=response.output_tokens)
        content = _strip_code_fence(response.content.strip())
        if content == "DONE":
            return AgentStep(action=None, usage=usage, agent_latency_ms=response.latency_ms)
        try:
            action = Action.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValueError) as exc:
            raise LiveAgentResponseError(
                f"model response was not a valid action or DONE: {content!r}"
            ) from exc
        return AgentStep(action=action, usage=usage, agent_latency_ms=response.latency_ms)

    def _build_messages(self, observation: Observation, history: list[TraceStep]) -> list[Message]:
        transcript = "\n".join(f"step {step.index}: action={step.action}" for step in history)
        observation_text = (
            f"url={observation.url} title={observation.title} elements={observation.elements}"
        )
        content = (
            f"Goal: {self._goal}\n\n"
            f"History:\n{transcript or '(none)'}\n\n"
            f"Current observation:\n{observation_text}"
        )
        return [Message(role="user", content=f"{self._system_prompt}\n\n{content}")]


__all__ = ["DEFAULT_SYSTEM_PROMPT", "LiveAgent", "LiveAgentResponseError"]
