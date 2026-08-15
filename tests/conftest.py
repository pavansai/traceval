"""Shared pytest fixtures.

CLI output goes through typer/rich, whose color and line-wrap behavior
depends on terminal detection (NO_COLOR, TERM, COLUMNS), not on whether
output is captured by CliRunner. Locally that usually means plain text (no
real TTY); on CI, color and width can be detected differently, splitting a
flag like --allow-different-task across separate ANSI color spans or
wrapping it across lines, either of which breaks a plain substring
assertion on CLI output. Pinning all three here makes CLI output
deterministic everywhere, rather than relying on the local dev machine's
terminal state to happen to produce plain text.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _deterministic_cli_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("COLUMNS", "200")
