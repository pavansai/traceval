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
