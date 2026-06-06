from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

ComplexView: TypeAlias = Literal["magnitude", "phase", "both"]
RenderView: TypeAlias = Literal["real", "magnitude", "phase"]


def normalize_complex_view(view: str) -> ComplexView:
    normalized = view.lower().strip()
    if normalized == "angular":
        return "phase"
    if normalized in {"magnitude", "phase", "both"}:
        return normalized
    raise ValueError(
        "Complex view must be one of 'magnitude', 'phase', 'both', or "
        "the alias 'angular'."
    )


@dataclass(frozen=True, slots=True)
class FieldRenderSpec:
    complex_view: ComplexView = "both"
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if self.gamma <= 0.0:
            raise ValueError("Gamma must be positive.")


__all__ = [
    "ComplexView",
    "FieldRenderSpec",
    "RenderView",
    "normalize_complex_view",
]
