"""Tests for BrowserEnvironment's own behavior in isolation (action timeout,
here) rather than a full task run through the runner/agent/scoring stack.
See test_end_to_end.py for those.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from traceval.environments import Action
from traceval.environments.browser import BrowserEnvironment
from traceval.tasks import TaskFixture

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "tasks" / "example_search_task"


def _fixture() -> TaskFixture:
    return TaskFixture(task_dir=FIXTURE_DIR, files={"fixture.html": FIXTURE_DIR / "fixture.html"})


def test_default_action_timeout_is_much_shorter_than_playwright_default() -> None:
    """No action_timeout_ms override: uses BrowserEnvironment's own 5s
    default, not Playwright's native 30s one.
    """
    env = BrowserEnvironment({"fixture_file": "fixture.html"})
    try:
        env.reset(seed=1, fixture=_fixture())
        start = time.monotonic()
        with pytest.raises(PlaywrightTimeoutError):
            env.step(Action(kind="click", target="#does-not-exist"))
        elapsed = time.monotonic() - start
        assert elapsed < 15.0
    finally:
        env.close()


def test_action_timeout_is_configurable() -> None:
    """An explicit, much shorter override actually takes effect, distinct
    from both the 5s default and Playwright's 30s one.
    """
    env = BrowserEnvironment({"fixture_file": "fixture.html", "action_timeout_ms": 300})
    try:
        env.reset(seed=1, fixture=_fixture())
        start = time.monotonic()
        with pytest.raises(PlaywrightTimeoutError):
            env.step(Action(kind="click", target="#does-not-exist"))
        elapsed = time.monotonic() - start
        assert elapsed < 3.0
    finally:
        env.close()
