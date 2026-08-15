"""Streaming reader/validator for trace JSONL files."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from traceval.trace.schema import (
    TRACE_SCHEMA_VERSION,
    Trace,
    TraceFooter,
    TraceHeader,
    TraceLine,
    TraceStep,
)


class TraceSchemaError(ValueError):
    """Raised when a trace file has an unsupported schema_version or malformed structure."""


def iter_trace_lines(path: Path) -> Iterator[TraceLine]:
    with path.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            line_type = payload.get("type")
            if line_type == "header":
                header = TraceHeader.model_validate(payload)
                if header.schema_version != TRACE_SCHEMA_VERSION:
                    raise TraceSchemaError(
                        f"{path}: unsupported schema_version {header.schema_version} "
                        f"(reader supports {TRACE_SCHEMA_VERSION})"
                    )
                yield header
            elif line_type == "step":
                yield TraceStep.model_validate(payload)
            elif line_type == "footer":
                yield TraceFooter.model_validate(payload)
            else:
                raise TraceSchemaError(f"{path}:{lineno}: unknown trace line type {line_type!r}")


def read_trace(path: Path) -> Trace:
    header: TraceHeader | None = None
    steps: list[TraceStep] = []
    footer: TraceFooter | None = None
    for line in iter_trace_lines(path):
        if isinstance(line, TraceHeader):
            if header is not None:
                raise TraceSchemaError(f"{path}: multiple header lines")
            header = line
        elif isinstance(line, TraceStep):
            steps.append(line)
        elif isinstance(line, TraceFooter):
            if footer is not None:
                raise TraceSchemaError(f"{path}: multiple footer lines")
            footer = line
    if header is None:
        raise TraceSchemaError(f"{path}: missing header line")
    return Trace(header=header, steps=sorted(steps, key=lambda s: s.index), footer=footer)


__all__ = ["TraceSchemaError", "iter_trace_lines", "read_trace"]
