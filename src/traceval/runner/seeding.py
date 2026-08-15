"""Seeds Python's stdlib `random` module for the harness process.

`run_task` calls this with `task.seed`, and separately passes the same raw
seed to `Environment.reset(seed, fixture)`. There is no sub-seed derivation
here, just the one process-global `random.seed()` call, in case harness code
itself branches on randomness. Determinism for `oracle` and `replay` runs
follows from this plus the fixed action sequence; `live` runs call a real
model and are not expected to be byte-reproducible.

`random.seed()` mutates process-global state, so this is only safe under the
runner's current single-threaded, one-run-at-a-time execution model. See
"Concurrency" in `docs/architecture.md`.
"""

from __future__ import annotations

import random


def seed_python_random(seed: int) -> None:
    random.seed(seed)


__all__ = ["seed_python_random"]
