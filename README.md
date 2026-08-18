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
uv run traceval diff "$TRACE" --against-last-passing
```

`--against-last-passing` scans `$TRACE`'s directory for the most recent
`Outcome.SUCCESS` trace with a matching `task_hash` and diffs against that,
so you don't have to track down the right trace file by hand.

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
task. Run 2026-08-18; total cost **$0.0042**.

```
$ uv run traceval run tasks/ \
    --agent live --provider anthropic --model claude-haiku-4-5 \
    --judge-provider anthropic --judge-model claude-haiku-4-5

feedback_form: failure -> runs/5fc02268612948519534b0c25948b9ea.jsonl
login_form: success -> runs/cd5b8258405f4792bd6a71301e44b598.jsonl
newsletter_signup: success -> runs/3c7fe2b671ee49b4806aa51d91ea4be1.jsonl
todo_list: success -> runs/07f2307deaa545d98f764a0d250a2220.jsonl
runs: 4 (success=3 failure=1 error=0)
  exact_match: accuracy=100% mean_score=1.000
  model_graded: accuracy=0% mean_score=0.000
  rubric: accuracy=100% mean_score=1.000
agent latency: p50=2458.6ms p95=4762.0ms
env latency: p50=164.8ms p95=258.0ms
tokens: input=2824 output=274
cost: $0.0042
```

This is pasted as-is, including the failure: a curated all-green run would
prove less. This run is the same four tasks against the same model as the
previous sample above, after two fixes: a required `goal` field on every
task (`TASK_FORMAT_VERSION` 3) and balanced-JSON-object extraction in
`LiveAgent` (see CHANGELOG). Both of the previous run's findings are gone:
`login_form` and `todo_list` now pass once the model was actually told what
the task wanted, and `newsletter_signup` now parses its JSON action correctly
despite the model's conversational preamble.

One new, real, still-open finding came out of this run:

- **`feedback_form` fails on real model behavior, not a harness bug.** The
  judge's rationale: "The agent repeatedly clicked on the feedback textarea
  without ever typing any text into it or clicking the submit button, so no
  positive feedback was entered or submitted." The trace shows exactly that:
  five `click` actions on `#feedback`, the task's `max_steps` limit, never a
  `type` action and never a click on `#submit`. The goal and
  `observe_selectors` are both correct and complete here, so this isn't a
  harness defect, it's the harness doing its job: a live sample this small
  now surfaces real agent-scaffold/model brittleness (the model stalling on
  a click-only loop) instead of a harness bug standing in the way of ever
  seeing it.

A committed example trace (a `feedback_form` success from an earlier run,
before this session's fixes, full schema, real tokens/cost/judge rationale)
lives at [`examples/`](examples/) so the format is readable without running
anything; it predates the fixes above and no longer matches this run's
result for that task.

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
