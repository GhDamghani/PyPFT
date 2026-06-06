from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypft.tracing import TransformDirection
from pypft.visualization import ComplexView, normalize_complex_view


@dataclass(frozen=True, slots=True)
class TransformWorkflowRequest:
    direction: TransformDirection
    input_path: Path
    output_dir: Path
    metadata_path: Path | None = None
    gamma: float = 1.0
    complex_view: ComplexView = "both"
    save_all_views: bool = False
    save_stage_arrays: bool = False
    backend: str | None = None
    dht_implementation: str | None = None

    def __post_init__(self) -> None:
        if self.gamma <= 0.0:
            raise ValueError("Gamma must be positive.")
        object.__setattr__(
            self,
            "complex_view",
            normalize_complex_view(self.complex_view),
        )


__all__ = ["TransformWorkflowRequest"]
