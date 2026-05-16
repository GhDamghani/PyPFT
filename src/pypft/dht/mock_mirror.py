from __future__ import annotations

from dataclasses import dataclass

from pypft.core.typing import ComplexArray
from pypft.dht.base import DHTDirection
from pypft.grids import PolarFrequencyGrid, PolarSpatialGrid


@dataclass(frozen=True, slots=True)
class MockMirrorDHTImplementation:
    """Mock DHT that returns the input array unchanged.

    This implementation exists only to exercise the surrounding transform
    pipeline while the real radial transform math is still under development.
    """

    key: str = "mock-mirror"
    description: str = "Mock DHT that mirrors its input without modification."
    supports_batching: bool = True

    def apply(
        self,
        values: ComplexArray,
        *,
        source_grid: PolarSpatialGrid | PolarFrequencyGrid,
        target_grid: PolarSpatialGrid | PolarFrequencyGrid,
        direction: DHTDirection,
    ) -> ComplexArray:
        del source_grid, target_grid, direction
        return values


__all__ = ["MockMirrorDHTImplementation"]
