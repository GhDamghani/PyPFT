from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pypft.tracing import TraceFrame
from pypft.visualization.models import RenderView


class FieldRenderer(Protocol):
    def save_frame(
        self,
        frame: TraceFrame,
        path: Path,
        *,
        view: RenderView,
        gamma: float,
    ) -> None: ...


__all__ = ["FieldRenderer"]
