from __future__ import annotations

from dataclasses import dataclass, field

from pypft.tracing.models import TraceFrame, TransformDirection, TransformTrace


@dataclass(slots=True)
class CollectingTraceSink:
    direction: TransformDirection
    _frames: list[TraceFrame] = field(default_factory=list)

    def record(self, frame: TraceFrame) -> None:
        if frame.direction != self.direction:
            raise ValueError(
                "Trace frame direction does not match the collector "
                "direction."
            )
        self._frames.append(frame)

    def build(self) -> TransformTrace:
        return TransformTrace(
            direction=self.direction,
            frames=tuple(self._frames),
        )


__all__ = ["CollectingTraceSink"]