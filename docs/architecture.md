# Architecture

## Protocol contracts

### `Provider` (`src/traceval/providers/__init__.py`)

```python
class Provider(Protocol):
    def resolve_model(self, alias: str) -> ResolvedModel: ...
    def generate(
        self, resolved: ResolvedModel, messages: list[Message], tools: list[dict] | None = None
    ) -> ProviderResponse: ...
```

`resolve_model` looks up an alias (e.g. `"claude-sonnet-5"`) in the pinned
registry (`models.yaml`) and returns a `ResolvedModel` carrying the exact
provider + version string. Traces stamp the `ResolvedModel`, never the alias,
so a run stays reproducible even after `models.yaml` repoints an alias at
a newer pin. The judge model used for model-graded scoring is resolved and
configured completely independently of the model under test; nothing forces
them to share a provider or version.

Implementations: `mock_provider.py` (deterministic, no network, used by
tests and as a judge in integration tests), `anthropic_provider.py` (real,
lazily constructs its client so `resolve_model()` never needs an API key),
`openai_provider.py` (stub: `resolve_model()` works, `generate()` raises
`NotImplementedError`).

### `Environment` (`src/traceval/environments/__init__.py`)

```python
class Environment(Protocol):
    def reset(self, seed: int, fixture: TaskFixture) -> Observation: ...
    def step(self, action: Action) -> StepResult: ...
    def fingerprint(self) -> str: ...
    def close(self) -> None: ...
```

`browser.py` is the only implementation today: a Playwright-driven browser
with a fixed viewport/locale/timezone and `Math.random` replaced by a seeded
PRNG via an init script, so the same seed against the same fixture produces
byte-identical observations. A future desktop environment implements the
same four methods and needs no changes to the runner.

`environment.config.observe_selectors` is what `_observe()` reads into
`Observation.elements` after every step, which is the *only* information a
`live` agent ever gets about the page; there is no DOM/accessibility tree or
screenshot. A task's `observe_selectors` must include every element the
agent needs to know about to act (buttons, inputs), not just the ones a
scorer checks afterward, or the task is structurally unsolvable by a live
agent (see `load_task`'s validation below).
`environment.config.action_timeout_ms` (default 5000) bounds
`click`/`type`/`navigate`; Playwright's own default is 30s, which is fine
for a human but far too long to burn on a single wrong selector guess in an
agent loop.

### `Agent` (`src/traceval/agents/__init__.py`)

```python
class Agent(Protocol):
    def act(self, observation: Observation, history: list[TraceStep]) -> AgentStep: ...
```

`AgentStep` carries `action: Action | None`, `usage: AgentUsage`
(`input_tokens`/`output_tokens`), and `agent_latency_ms`. `action=None` ends
the episode; this is the only completion signal, since a real step always
has an action, so there's no separate "done" flag to keep in sync with it.
Three implementations:

- **`OracleAgent`** replays a task's `scripted_trajectory.jsonl`. Each line
  may carry an `expect` block describing the observation state expected
  before that action; a mismatch raises `ScriptedTrajectoryDivergedError`
  loudly rather than silently drifting. This is the deterministic backbone
  that proves the environment/runner/scoring pipeline works without a model.
  Returns zero `usage`.
- **`ReplayAgent`** replays a previously recorded trace's actions verbatim;
  backs `traceval replay`. Returns zero `usage`.
- **`LiveAgent`** is the only agent that calls a real `Provider.generate()`
  and therefore the only one that needs an API key. It is deliberately
  minimal (a JSON-object-in-content action protocol), populates `usage` and
  `agent_latency_ms` from the `ProviderResponse`, and, because it needs an
  API key, is only exercised in tests via `MockProvider`, never a real one.

**Only `LiveAgent` ever reads `Observation`.** `OracleAgent` and
`ReplayAgent` take the `observation` parameter but never look at it; their
next action comes purely from a pre-written script or a recorded trace, not
from anything the environment reports back. This means a task whose
`observe_selectors` don't show the agent everything it needs to act (the
bug that motivated the `observe_selectors`/scoring-target split above) is
completely invisible under Oracle or Replay, no matter how thoroughly
they're tested: they'll happily run the exact same scripted clicks whether
or not a live agent could ever have discovered those selectors on its own.
Only a `live` run exercises the path that actually depends on
`Observation` content, which is exactly how the `feedback_form` bug went
unnoticed until `scripts/smoke_live.py` first ran against a real model.

### `Scorer` (`src/traceval/scoring/__init__.py`)

```python
class Scorer(Protocol):
    def score(self, task: Task, trace: Trace) -> ScoreResult: ...
```

- **`ExactMatchScorer`** compares the final observation's value at a
  selector to `task.yaml`'s configured `expected` value.
- **`RubricScorer`** is a weighted, deterministic checklist; each criterion
  is a predicate over the trace's step history (currently: "did an action
  with these fields occur"). No LLM involved.
- **`ModelGradedScorer`** is constructed with its own judge `Provider` +
  `ResolvedModel`, sends the trajectory plus a grading prompt to the judge,
  and parses a PASS/FAIL verdict.
- **`build_report`** (`scoring/report.py`) aggregates `ScoreResult`s and
  trace footers across a set of traces into per-scorer accuracy/mean-score,
  agent/env p50/p95 latency, and total token cost (via `pricing.yaml`, keyed by the
  resolved `model_id`; cost is `None`, not silently wrong, for an unpriced
  model). `run_count` includes every trace regardless of outcome;
  `success_count`/`failure_count`/`error_count` break that same total down
  explicitly rather than folding an errored (unscored) task into either
  category or excluding it from `run_count` silently.

## Trace schema (`src/traceval/trace/schema.py`)

Append-only JSONL. One header line, one line per step, one footer line.
`TRACE_SCHEMA_VERSION` is bumped on any breaking field change; `reader.py`
refuses to load a trace whose `schema_version` it doesn't recognize.

- **Header**: `run_id`, `task_id`, `task_hash` (sha256 over `task.yaml` +
  fixture files), `task_format_version`, `seed`, `model_under_test`
  (`ResolvedModel`), `judge_model` (`ResolvedModel | None`), `agent_kind`,
  `environment_fingerprint`, `started_at`.
- **Step**: `index`, `observation_hash` (cheap fingerprint) and the full
  `observation` (so traces are self-contained for scoring and for a human
  diff, not just a hash), `action`, `input_tokens`, `output_tokens`,
  `agent_latency_ms` (time inside `agent.act()`, i.e. the model call for
  `live`), `env_latency_ms` (time inside `environment.step()`), `timestamp`.
- **Footer**: `outcome` (`success` | `failure` | `error`), `total_steps`,
  `scores` (list of `ScoreResult`, only the scorers that completed), `totals`
  (summed tokens and both latencies), `error` (`TraceError | None`:
  `error_type` + `error_message`, set when `reset()`/`step()`/`agent.act()`
  raised before scoring was ever attempted), `scorer_errors` (list of
  `TraceError`, one per `Scorer` that raised; `TraceError.scorer` names it by
  class), `ended_at`.

The writer flushes after every line, so a crashed run leaves a usable
partial trace (header + whatever steps completed) rather than nothing.

**Observations are stored inline and in full** (not just `observation_hash`)
so a trace is self-contained for scoring and for a human diff without
needing to replay the run. That doesn't scale to environments whose
observations are DOM snapshots or screenshots rather than the small
`dict[str, str]` the browser environment's `observe_selectors` produces
today: a task set with many large observations would produce
correspondingly large trace files. Out-of-line storage (write the
observation to a content-addressed blob, stamp its hash into the step) is
the intended fix; not needed for v0.1.0's text-only fixtures.

## Runner (`src/traceval/runner/runner.py`)

`run_task(task, agent, agent_kind, model_under_test, scorers, trace_dir,
judge_model=None)`:

1. `environment.reset(task.seed, fixture)` (the environment seeds itself
   from this: `browser.py` substitutes a seeded PRNG for `Math.random`
   before the page loads), then write the trace header (including
   `environment.fingerprint()`).
2. Loop up to `task.max_steps`: `agent.act(observation, steps)` returns an
   `AgentStep` (`action`, `usage`, `agent_latency_ms`). If `action` is
   `None`, stop; else `environment.step(action)`, append a `TraceStep`
   carrying both the agent's usage/latency and the environment's own
   step latency, continue.
3. Run every configured `Scorer` against the trace-so-far, write the
   footer, close the environment.

If `reset()`, `step()`, or `agent.act()` raises, the runner still writes a
complete trace: a header (using a `"unavailable"` environment fingerprint if
the exception happened before `reset()` returned one) and a footer with
`outcome=error` and the exception's type/message in `TraceFooter.error`, then
re-raises. Scorers are different: each `Scorer.score()` call runs in its own
try/except, so one scorer raising (e.g. an unparseable judge response) never
discards another scorer's result and never stops later scorers from still
running: `scores` keeps every scorer that completed, `scorer_errors` records
every one that didn't (by class name), and `outcome=error` is set if
`scorer_errors` is non-empty even when every other scorer passed, since
incomplete scoring can't be claimed as success or failure. Either way the
runner re-raises afterward, so the failure is both visible in the trace and
not swallowed by the harness.

`cli run`'s batch loop (`src/traceval/cli/main.py`) catches per task: a known
`run_id` is generated before calling `run_task`, so if it raises, the CLI
re-reads the trace `run_task` already wrote (guaranteed complete per above)
and continues to the next task rather than aborting the whole batch. `run`
exits nonzero if any task's outcome was `error` (`--exit-zero-on-error`
restores the old always-exit-0 behavior): a task that never got scored is
not the same as one that failed, and a harness that exits 0 either way would
silently green a CI pipeline on an unscored task.

This is the one place `Environment`, `Agent`, `TraceWriter`, and `Scorer`
are wired together; adding a new environment or agent kind never touches
this file.

## Task / fixture format (`src/traceval/tasks/schema.py`)

A task is a directory with `task.yaml` plus whatever fixture files it
references (`fixture_files: [...]`). `task.yaml` fields: `id`, `seed`,
`environment` (`kind` + kind-specific `config`), `max_steps`, `scorers`
(list of `{kind, config}`), `expected` (used by `exact_match`),
`requires_live_judge` (default `false`; see below). See
`tests/fixtures/tasks/example_search_task/task.yaml` for a complete
example, including the `rubric` scorer's `action_occurred` criteria.

A `model_graded` task set `requires_live_judge: true` when a canned mock
judge verdict would only ever rubber-stamp a pass rather than actually
judging anything (see `tasks/feedback_form/task.yaml`). `cli run` skips such
a task, rather than running and erroring it or faking a pass, whenever the
configured judge is `MockProvider` or no judge was configured at all. The
skip isn't counted in the report at all, since no trace exists for it.

`load_task` validates as well as parses: for the browser environment, every
`exact_match` scorer's `target` must appear in
`environment.config.observe_selectors`, or the task raises
`TaskValidationError` immediately rather than loading successfully and
failing to score forever afterward. `rubric`'s `target` is exempt; it's an
*action* target checked against the trace's action history, not a value
read from an observation, so it isn't part of this contract. This is the
load-bearing half of `TASK_FORMAT_VERSION` 2's clarified contract:
`observe_selectors` is every element the agent needs to see to act, not
just what a scorer happens to check afterward (see the `Environment`
section above).

`compute_task_hash(task)` hashes `task.yaml` plus every referenced fixture
file's contents, stamped into every trace as `task_hash`, so a trace can
be checked against the exact task definition that produced it.

## Adding a task

1. Create `tests/fixtures/tasks/<name>/task.yaml` (copy
   `example_search_task/task.yaml` as a starting point).
2. Add whatever fixture files the environment needs (e.g. a static
   `fixture.html` for the browser environment) and list them in
   `fixture_files`.
3. Write `scripted_trajectory.jsonl`, the oracle agent's script. Each line
   is `{"expect": {...}, "action": {...}}`; `expect.elements` is checked
   against the live observation before the action runs.
4. `uv run traceval run tests/fixtures/tasks/<name> --agent oracle` to
   verify it passes with no model involved.

## Adding a provider

Implement `Provider` (`resolve_model`, `generate`) in
`src/traceval/providers/<name>_provider.py`, and add its models to
`models.yaml` (and, for cost reporting, `pricing.yaml`). Register it in
`cli/main.py`'s `_build_provider()`. See `openai_provider.py` for the
minimal shape of a provider whose `generate()` isn't implemented yet;
`resolve_model()` alone is enough to prove the registry/pinning path works.

## Adding an environment

Implement `Environment` (`reset`, `step`, `fingerprint`, `close`) in
`src/traceval/environments/<name>.py`. Register its `kind` string in
`runner/runner.py`'s `_ENVIRONMENT_FACTORIES`. The constructor takes the
task's `environment.config` dict; nothing else in the runner, agents, or
scoring layers needs to change.
