from __future__ import annotations

from typing import Literal, Protocol

from pypft.core.typing import ComplexArray
from pypft.grids import PolarFrequencyGrid, PolarSpatialGrid

DHTDirection = Literal["forward", "backward"]


class DHTImplementation(Protocol):
    key: str
    description: str
    supports_batching: bool

    def apply(
        self,
        values: ComplexArray,
        *,
        source_grid: PolarSpatialGrid | PolarFrequencyGrid,
        target_grid: PolarSpatialGrid | PolarFrequencyGrid,
        direction: DHTDirection,
    ) -> ComplexArray: ...


__all__ = ["DHTDirection", "DHTImplementation"]
