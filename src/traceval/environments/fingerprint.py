"""Deterministic environment fingerprint.

Identifies the environment *substrate* a trace ran against (browser engine +
version, viewport, locale, timezone) independent of task content (task
content is already covered by `task_hash`). Stamped into every trace header.
"""

from __future__ import annotations

import hashlib


def compute_browser_fingerprint(
    *,
    browser_name: str,
    browser_version: str,
    viewport: dict[str, int],
    locale: str,
    timezone_id: str,
) -> str:
    parts = [
        f"browser={browser_name}",
        f"version={browser_version}",
        f"viewport={viewport.get('width')}x{viewport.get('height')}",
        f"locale={locale}",
        f"timezone={timezone_id}",
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


__all__ = ["compute_browser_fingerprint"]
