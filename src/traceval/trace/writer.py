"""Append-only JSONL trace writer.

Flushes after every line so a crashed run still leaves a usable partial trace
on disk (readable up to the last flushed line).
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

from traceval.trace.schema import TraceFooter, TraceHeader, TraceLine, TraceStep


class TraceWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def write(self, line: TraceLine) -> None:
        self._file.write(line.model_dump_json() + "\n")
        self._file.flush()

    def write_header(self, header: TraceHeader) -> None:
        self.write(header)

    def write_step(self, step: TraceStep) -> None:
        self.write(step)

    def write_footer(self, footer: TraceFooter) -> None:
        self.write(footer)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["TraceWriter"]
