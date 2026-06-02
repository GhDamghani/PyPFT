from __future__ import annotations

from dataclasses import dataclass

from pypft.core.exceptions import GridMismatchError
from pypft.fields.base import PolarFieldBase
from pypft.grids.polar import (
    PolarAngularModeGrid,
    PolarFrequencyGrid,
    PolarRadialModeGrid,
    PolarSpatialGrid,
)


@dataclass(frozen=True, slots=True, eq=False)
class SpatialSamples(PolarFieldBase[PolarSpatialGrid]):
    """Spatial-domain polar samples tied to a polar spatial grid."""


@dataclass(frozen=True, slots=True, eq=False)
class AngularSpectrum(PolarFieldBase[PolarAngularModeGrid]):
    """Angular Fourier coefficients tied to a polar angular-mode grid."""

    endpoint_grid: PolarSpatialGrid

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.grid.radial_size != self.endpoint_grid.radial_size:
            raise GridMismatchError(
                "AngularSpectrum radial size must match its originating "
                "spatial grid."
            )
        if self.grid.angular_mode_count != self.endpoint_grid.angular_size:
            raise GridMismatchError(
                "AngularSpectrum mode count must match its originating "
                "spatial grid angular size."
            )

    def _extra_init_kwargs(self) -> dict[str, PolarSpatialGrid]:
        return {"endpoint_grid": self.endpoint_grid}


@dataclass(frozen=True, slots=True, eq=False)
class RadialSpectrum(PolarFieldBase[PolarRadialModeGrid]):
    """Radial Hankel-domain coefficients tied to a radial-mode grid."""

    endpoint_grid: PolarFrequencyGrid

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.grid.radial_frequency_size != self.endpoint_grid.radial_size:
            raise GridMismatchError(
                "RadialSpectrum radial frequency size must match its "
                "originating frequency grid."
            )
        if self.grid.angular_mode_count != self.endpoint_grid.angular_size:
            raise GridMismatchError(
                "RadialSpectrum mode count must match its originating "
                "frequency grid angular size."
            )

    def _extra_init_kwargs(self) -> dict[str, PolarFrequencyGrid]:
        return {"endpoint_grid": self.endpoint_grid}


@dataclass(frozen=True, slots=True, eq=False)
class FrequencySamples(PolarFieldBase[PolarFrequencyGrid]):
    """Frequency-domain polar samples tied to a polar frequency grid."""


__all__ = [
    "AngularSpectrum",
    "FrequencySamples",
    "RadialSpectrum",
    "SpatialSamples",
]