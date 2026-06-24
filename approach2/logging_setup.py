"""Stdout/stderr tee for run logging."""

from __future__ import annotations

from typing import TextIO

class Tee(TextIO):
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()
