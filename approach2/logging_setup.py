"""Stdout/stderr tee for run logging."""

from __future__ import annotations

from typing import TextIO


class Tee:
    """Mirror writes to multiple streams; delegate TTY metadata to the primary stream."""

    def __init__(self, *streams: TextIO):
        self.streams = streams
        self._primary = streams[0]

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        isatty = getattr(self._primary, "isatty", None)
        return bool(isatty()) if callable(isatty) else False

    def fileno(self) -> int:
        return self._primary.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._primary, "encoding", "utf-8")
