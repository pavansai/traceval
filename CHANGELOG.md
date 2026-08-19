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
- `traceval diff --against-last-passing`: given one trace path, scans its
  directory for the most recent `Outcome.SUCCESS` trace with a matching
  `task_hash` and diffs against that, instead of requiring an explicit
  second trace path. Candidates are ordered by the header's `started_at`,
  not file mtime, so the result is stable if trace files get copied
  around. Refuses clearly when no candidate exists, and distinguishes
  "no passing runs in this directory at all" from "passing runs exist,
  just not for this task_hash", since those call for different next steps.
- `tasks/unrecoverable_account`: a fifth real task, deliberately
  unwinnable by any agent (see the comment at the top of its `task.yaml`).
  A task set where every task is passable can't distinguish a working
  scorer from a lenient one, so this is a regression signal for the
  harness itself, not a capability being measured. Its actions all
  complete cleanly, so its `Outcome` is always `failure`, never `error`.

### Changed

- `TASK_FORMAT_VERSION` bumped to 3: `task.yaml` now requires a `goal` field,
  a natural-language statement of what the agent is being asked to
  accomplish. `LiveAgent` includes it verbatim in its prompt; `load_task`
  fails loudly if it's missing or blank. Previously a task was only an
  environment plus a scoring rule, so a live model had no way to know what
  the task actually wanted and could only guess from the DOM. `login_form`
  scored a real model 0% for typing a plausible username ("testuser")
  instead of the one the scorer happened to check for ("alice"), which
  measured nothing about the model since the task never said which
  username was required. Task goals are written to state the requirement
  without leaking the exact string a scorer checks (`login_form`'s goal
  says to log in as user alice, not "type alice into #username").
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
- `LiveAgent` failed whenever a real model added any conversational
  preamble before its JSON action (e.g. "I'll help you complete this
  newsletter signup task. Let me start by..."), since the code-fence fix
  above still parsed the (fence-stripped) content as JSON directly, and
  prose before the object isn't valid JSON. Surfaced by the four-task live
  sample run, where `newsletter_signup` errored on exactly this. Fixed by
  scanning for the first balanced `{...}` object in the response instead
  of parsing the whole string; code-fence stripping is kept (still needed
  to recognize a fenced `DONE`, which has no JSON object to find). On
  failure, `LiveAgentResponseError` now includes the model's raw,
  unprocessed response rather than the fence-stripped content, so an
  unparseable response is diagnosable from the trace error alone.
- CI's Node 20 deprecation annotation, by bumping `actions/checkout` v4 to
  v7, `actions/cache` v4 to v6, and `astral-sh/setup-uv` v3 to v10 (pinned
  by commit SHA per astral-sh's own README recommendation). All three had
  fallen behind the Node 24 runner GitHub Actions now uses.
