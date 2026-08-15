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
