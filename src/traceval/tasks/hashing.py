"""Canonical task hash: sha256 over task.yaml plus every referenced fixture file.

Stamped into every trace header as `task_hash`, so a trace can be checked
against the exact task definition that produced it. Edit task.yaml or any
fixture file and the hash changes.
"""

from __future__ import annotations

import hashlib

from traceval.tasks.schema import Task


def compute_task_hash(task: Task) -> str:
    hasher = hashlib.sha256()
    hasher.update((task.task_dir / "task.yaml").read_bytes())
    for name in sorted(task.fixture_files):
        hasher.update(name.encode("utf-8"))
        hasher.update((task.task_dir / name).read_bytes())
    return f"sha256:{hasher.hexdigest()}"


__all__ = ["compute_task_hash"]
