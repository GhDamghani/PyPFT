from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True, eq=False)
class RadialSpectrum(PolarFieldBase[PolarRadialModeGrid]):
    """Radial Hankel-domain coefficients tied to a radial-mode grid."""


@dataclass(frozen=True, slots=True, eq=False)
class FrequencySamples(PolarFieldBase[PolarFrequencyGrid]):
    """Frequency-domain polar samples tied to a polar frequency grid."""


__all__ = [
    "AngularSpectrum",
    "FrequencySamples",
    "RadialSpectrum",
    "SpatialSamples",
]