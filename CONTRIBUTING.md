# Contributing

## Local dev loop

```sh
uv sync
uv run pytest tests/unit tests/integration -v   # no API key needed, ever
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

All four must pass before opening a PR; CI runs the same four commands with
no repository secrets configured, which is the mechanism that enforces "no
test may require an API key": if a test needed one, CI would fail, not
skip quietly.

## Adding a task, provider, or environment

See [`docs/architecture.md`](docs/architecture.md); it has a short,
concrete walkthrough for each.

## Testing rules

- **No test may call a real model provider or need `ANTHROPIC_API_KEY`.**
  Use `OracleAgent` (scripted trajectory) or `ReplayAgent` (recorded trace)
  instead of `LiveAgent`, and `MockProvider` wherever a `Provider` is
  needed, including as the judge for `ModelGradedScorer` tests.
- `tests/unit/`: fast, no browser, no network. One test file per `src/`
  module it covers.
- `tests/integration/`: real `BrowserEnvironment` (headless Playwright),
  driven by `OracleAgent`, exercising the full runner/scoring/CLI pipeline
  end to end.
- `LiveAgent` and the real `AnthropicProvider`/`OpenAIProvider` are covered
  only by a manual, opt-in smoke check (not part of `pytest`/CI) once one
  exists. See the `## Manual smoke-testing a real model` section below if
  you're adding one.
- **CLI tests must assert against normalized output, not raw `result.output`,
  and must not depend on terminal detection.** typer/rich color and
  line-wrap CLI output based on `NO_COLOR`/`TERM`/`COLUMNS`, which can
  differ between a local dev machine (no TTY, usually plain) and CI
  (color/width can be forced), splitting a flag name across ANSI spans or
  line wraps and breaking a plain substring check that happened to pass
  locally by accident. `tests/conftest.py`'s autouse fixture pins those
  three env vars for every test; every CLI assertion should additionally go
  through `normalize_cli_output()` (strips ANSI, collapses whitespace) in
  `tests/integration/test_end_to_end.py`.

## Manual smoke-testing a real model

There's no `scripts/smoke_live.py` yet. If you're adding real-provider
coverage, keep it out of `pytest`/CI (it needs `ANTHROPIC_API_KEY`) and gate
it behind an explicit script invocation, e.g.:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
uv run traceval run tests/fixtures/tasks/example_search_task \
  --agent live --provider anthropic --model claude-sonnet-5
```

## Adding or bumping a pinned model version

Edit `models.yaml` (and `pricing.yaml` if the per-token price changed). This
is a one-line, reviewable diff; `resolve_model()` never asks a provider to
resolve its own "latest" alias at runtime, so nothing else needs to change.

## Code style

- Ruff (lint + format) and mypy (`strict = true`) are both enforced in CI.
- Prefer editing existing modules over adding new abstractions; see the
  protocol contracts in `docs/architecture.md` before introducing a new one.
- Comments only where the *why* isn't obvious from the code (a hidden
  constraint, a workaround, a non-obvious invariant), not restatements of
  what the code does.

## Branch / PR conventions

- One logical change per PR; keep the diff reviewable.
- Reference the module(s) touched in the PR title (e.g. "scoring: add
  weighted rubric threshold").
- Update `CHANGELOG.md`'s `Unreleased` section for anything user-visible
  (new CLI flag, new scorer kind, trace schema change, breaking config
  change).

## Release process

1. Move the relevant `Unreleased` entries in `CHANGELOG.md` under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading, following [Keep a Changelog].
2. Bump `version` in `pyproject.toml` to match.
3. Commit, then tag: `git tag vX.Y.Z && git push --tags`.
4. Semantic versioning: bump the major version on a `TRACE_SCHEMA_VERSION`
   or `TASK_FORMAT_VERSION` change, or any other break to trace/task
   backward-compatibility; minor for new scorer/provider/environment kinds
   or CLI flags; patch for fixes.

There's no automated publish step yet: this isn't a published package, so
a release is the tag plus the changelog entry, nothing more.

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
