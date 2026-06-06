from __future__ import annotations

from typing import Protocol

from pypft.tracing.models import TraceFrame


class TraceSink(Protocol):
    def record(self, frame: TraceFrame) -> None: ...


__all__ = ["TraceSink"]
