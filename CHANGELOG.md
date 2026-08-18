# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Deterministic task runner: seeded environments, per-task fixtures,
  append-only JSONL trace capture with schema versioning.
- `Provider` protocol with Anthropic (real), OpenAI (stub), and mock
  (deterministic, no network) implementations, plus a pinned model-version
  registry (`models.yaml`).
- `Environment` protocol with a Playwright-driven browser implementation
  (seeded via a substituted `Math.random`, fixed viewport/locale/timezone).
- `Agent` protocol with oracle (scripted trajectory replay), replay
  (recorded trace replay), and live (real model) implementations.
- Scoring layer: exact-match, rubric (deterministic weighted checklist),
  and model-graded (independent judge model) scorers, plus a task-set report
  aggregating accuracy, p50/p95 latency, and token cost (`pricing.yaml`).
- `traceval` CLI: `run`, `replay`, `diff`, `report`.
- `traceval replay` + `traceval diff`: replay a recorded trace's exact
  action sequence and diff it against another trace, step-aligned.
- Full CI (lint, format check, type check, unit + integration tests) with
  no repository secrets. Every test runs via the oracle/replay agents and
  the mock provider, never a real API key.
- `BrowserEnvironment` action timeout (`environment.config.action_timeout_ms`,
  default 5s) applied to click/type/navigate. Previously unset, so every
  action used Playwright's native 30s default; a wrong selector guess in an
  agent loop should fail fast, not burn 30 real seconds finding out.

### Changed

- `TASK_FORMAT_VERSION` bumped to 2: `environment.config.observe_selectors`
  (browser environment) is now unambiguously "every element the agent needs
  to see to act," not just whatever a scorer happens to check afterward.
  `load_task` fails loudly if an `exact_match` scorer's `target` isn't in
  `observe_selectors`, since that task could never pass. All four tasks
  under `tasks/` were missing their submit/add button from
  `observe_selectors`, which Oracle/Replay never surfaced (they never read
  observations to decide anything) and only showed up once `feedback_form`
  ran under `LiveAgent` for the first time.

### Fixed

- `LiveAgent` failed on its first real-model call: it parsed the model's
  response with `json.loads()` directly, but real Claude models routinely
  wrap requested JSON in a markdown code fence (` ```json ... ``` `) even
  when told to reply with nothing else. Because no test may use a real API
  key, this path had never actually executed until `scripts/smoke_live.py`
  first ran against a live model and hit it on turn one. Fixed by stripping
  a fence that wraps the entire response before parsing.
