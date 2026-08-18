# traceval

A deterministic task runner and scoring harness for multi-step agent
rollouts. Each task provides a seeded environment and per-task fixtures; the
agent takes a sequence of observation/action steps; the runner captures the
full trajectory as a structured, append-only trace that can be replayed and
diffed against the last passing run.

- **Deterministic runner**: seeded environments, per-task fixtures, structured
  trace capture. A failed run can be replayed and diffed against the last
  passing run (`traceval replay`, `traceval diff`).
- **Scoring layer**: exact-match, rubric (deterministic checklist), and
  model-graded (LLM judge) scorers, reporting accuracy alongside latency and
  token cost per task set (`traceval report`).
- **No test requires an API key.** An oracle agent replays a scripted
  trajectory and a replay agent replays a recorded trace, so the full test
  suite, including environment and scoring integration tests, runs in CI
  for free.

## Quickstart

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
uv sync
uv run playwright install chromium

uv run traceval run tests/fixtures/tasks/example_search_task --agent oracle
```

This runs the bundled example task with the oracle agent (no model, no API
key), writes a trace to `runs/`, and prints an accuracy/latency/cost report.

### Try replay and diff

```sh
TRACE=$(ls -t runs/*.jsonl | head -1)
uv run traceval replay "$TRACE" tests/fixtures/tasks/example_search_task
uv run traceval diff "$TRACE" "$(ls -t runs/*.jsonl | head -1)"
```

### Running against a real model

```sh
export ANTHROPIC_API_KEY=sk-ant-...
uv run traceval run tests/fixtures/tasks/example_search_task \
  --agent live --provider anthropic --model claude-sonnet-5
```

## Sample run

The only thing that actually demonstrates this harness has evaluated a real
model: all four tasks under `tasks/`, run live against `claude-haiku-4-5`
(Anthropic), scored by a real judge (same model) for the one `model_graded`
task. Run 2026-08-18; total cost **$0.0032**.

```
$ uv run traceval run tasks/ \
    --agent live --provider anthropic --model claude-haiku-4-5 \
    --judge-provider anthropic --judge-model claude-haiku-4-5

feedback_form: success -> runs/6d0d265f63b34389885be35f0b77a86e.jsonl
login_form: failure -> runs/486a623339394955bc1b4679df46a5f3.jsonl
newsletter_signup: error (LiveAgentResponseError: model response was not a
  valid action or DONE: "I'll help you complete this newsletter signup
  task. Let me start by examining the form and filling in the email
  field.\n\n{\"kind\": \"click\", \"target\": \"#email\"}") -> runs/a1e662cd31a84e3fa89586c466b6a814.jsonl
todo_list: failure -> runs/accb8f7b79274fa697a0af820957ac43.jsonl
runs: 4 (success=1 failure=2 error=1)
  exact_match: accuracy=0% mean_score=0.000
  model_graded: accuracy=100% mean_score=1.000
  rubric: accuracy=0% mean_score=0.500
agent latency: p50=3149.4ms p95=4656.9ms
env latency: p50=179.5ms p95=190.2ms
tokens: input=2145 output=203
cost: $0.0032
```

This is pasted as-is, including the failures: a curated all-green run would
prove less. Two real, distinct findings came out of it, both still open:

- **`login_form`/`todo_list` fail, not error: the harness ran correctly, the
  model just had no way to know what to type.** `exact_match` on
  `login_form` requires the exact string `"alice"`; `rubric` on `todo_list`
  requires an item containing `"milk"`. `LiveAgent` never sends the model a
  task-specific goal, only the generic action-protocol instructions plus
  the current page, so a real model reasonably fills in a plausible generic
  value (`"testuser"`, `"Buy groceries"`) instead of the one specific string
  the task happens to check for. `feedback_form` passes because it's graded
  qualitatively (any plausible feedback text), so it never needed this.
  There's currently no field in `task.yaml` for a live agent's goal.
- **`newsletter_signup` errors on a JSON-parsing edge case distinct from
  the markdown-fence bug already fixed**: the model prefixed its JSON
  action with conversational text ("I'll help you complete this... Let me
  start by...") instead of wrapping it in a code fence, so
  `_strip_code_fence` (anchored to match only a fence around the *entire*
  response) correctly left it alone, and `json.loads` failed on the mixed
  prose-plus-JSON string.

A committed example trace (the successful `feedback_form` run above, full
schema, real tokens/cost/judge rationale) lives at
[`examples/`](examples/) so the format is readable without running
anything.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the protocol contracts
(`Provider`, `Environment`, `Agent`, `Scorer`), the trace schema reference,
and guides for adding a task, provider, or environment.

In short: a `Task` (`task.yaml` + fixture files) is run by an `Agent`
(`oracle` | `replay` | `live`) against an `Environment` (currently a
Playwright-driven browser). Every step is written to an append-only JSONL
trace as it happens. Once the episode ends, configured `Scorer`s grade the
trace and the result is written as the trace's footer.

## Development

```sh
uv sync
uv run pytest tests/unit tests/integration -v   # no API key needed
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to add a task, provider, or
environment, and the release process.

## Project layout

```
src/traceval/
  providers/    Provider protocol + anthropic/openai/mock implementations
  environments/ Environment protocol + Playwright browser environment
  agents/       Agent protocol + oracle/replay/live implementations
  tasks/        Task schema, loading, and content hashing
  trace/        Trace schema, append-only writer/reader, diff
  scoring/      Scorer protocol + exact-match/rubric/model-graded + report
  runner/       Orchestrates seed -> env -> agent loop -> trace -> scorers
  cli/          `traceval` command (run/replay/diff/report)
models.yaml     Pinned model-version registry (alias -> exact version string)
pricing.yaml    Pinned per-token pricing for cost reporting
```
