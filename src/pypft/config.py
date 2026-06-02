from __future__ import annotations

from dataclasses import dataclass

from pypft.core.conventions import DEFAULT_CONVENTIONS, TransformConventions
from pypft.grids import PolarFrequencyGrid, PolarSpatialGrid


@dataclass(frozen=True, slots=True)
class PyPFTConfig:
    dht_implementation: str = "naive"
    enable_batching: bool = False
    backend: str = "cpu"
    conventions: TransformConventions = DEFAULT_CONVENTIONS
    spatial_grid: PolarSpatialGrid | None = None
    frequency_grid: PolarFrequencyGrid | None = None


__all__ = ["PyPFTConfig"]
