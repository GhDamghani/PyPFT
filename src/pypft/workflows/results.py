from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypft.tracing import TransformTrace


@dataclass(frozen=True, slots=True)
class TransformWorkflowResult:
    output_path: Path
    trace_path: Path
    manifest_path: Path
    stage_array_paths: dict[str, Path]
    figure_paths: dict[str, tuple[Path, ...]]
    trace: TransformTrace


__all__ = ["TransformWorkflowResult"]
