"""Playwright-driven browser environment.

Deterministic by construction: fixed viewport/locale/timezone, and a seeded
PRNG substituted for `Math.random` via an init script, so two runs with the
same seed against the same fixture produce byte-identical observations. The
fixture page is loaded from disk via a `file://` URL; no HTTP server needed
for static fixtures.
"""

from __future__ import annotations

from typing import Any, cast

from playwright.sync_api import Browser, Locator, Page, ViewportSize, sync_playwright

from traceval.environments import Action, Observation, StepResult
from traceval.environments.fingerprint import compute_browser_fingerprint
from traceval.tasks import TaskFixture

_SEEDED_RANDOM_INIT_SCRIPT = """
(function (seed) {
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  Math.random = mulberry32(seed);
})(%d);
"""


class BrowserEnvironment:
    def __init__(self, config: dict[str, Any]) -> None:
        self._fixture_file: str = config["fixture_file"]
        self._observe_selectors: list[str] = config.get("observe_selectors", [])
        self._viewport: dict[str, int] = config.get("viewport", {"width": 1280, "height": 800})
        self._locale: str = config.get("locale", "en-US")
        self._timezone_id: str = config.get("timezone_id", "UTC")

        self._playwright: Any = None
        self._browser: Browser | None = None
        self._page: Page | None = None
        self._fingerprint: str | None = None

    def reset(self, seed: int, fixture: TaskFixture) -> Observation:
        self.close()
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        context = self._browser.new_context(
            viewport=cast(ViewportSize, self._viewport),
            locale=self._locale,
            timezone_id=self._timezone_id,
        )
        context.add_init_script(_SEEDED_RANDOM_INIT_SCRIPT % seed)
        self._page = context.new_page()

        self._fingerprint = compute_browser_fingerprint(
            browser_name=self._browser.browser_type.name,
            browser_version=self._browser.version,
            viewport=self._viewport,
            locale=self._locale,
            timezone_id=self._timezone_id,
        )

        fixture_path = fixture.path(self._fixture_file).resolve()
        self._page.goto(fixture_path.as_uri())
        return self._observe()

    def step(self, action: Action) -> StepResult:
        page = self._require_page()
        if action.kind == "click":
            page.click(self._require(action.target, "target"))
        elif action.kind == "type":
            page.fill(self._require(action.target, "target"), self._require(action.value, "value"))
        elif action.kind == "navigate":
            page.goto(self._require(action.url, "url"))
        elif action.kind == "wait":
            page.wait_for_timeout(action.ms or 0)
        else:
            raise ValueError(f"unsupported action kind: {action.kind!r}")
        return StepResult(observation=self._observe(), done=False)

    def fingerprint(self) -> str:
        if self._fingerprint is None:
            raise RuntimeError("fingerprint() called before reset()")
        return self._fingerprint

    def close(self) -> None:
        if self._page is not None:
            self._page.close()
            self._page = None
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def _observe(self) -> Observation:
        page = self._require_page()
        elements = {
            selector: self._element_text(page.locator(selector))
            for selector in self._observe_selectors
        }
        return Observation(url=page.url, title=page.title(), elements=elements)

    @staticmethod
    def _element_text(locator: Locator) -> str:
        try:
            return locator.input_value()
        except Exception:
            return locator.text_content() or ""

    def _require_page(self) -> Page:
        if self._page is None:
            raise RuntimeError("environment not reset")
        return self._page

    @staticmethod
    def _require(value: str | None, name: str) -> str:
        if value is None:
            raise ValueError(f"action missing required field: {name!r}")
        return value


__all__ = ["BrowserEnvironment"]
