from __future__ import annotations

from dataclasses import dataclass, field
from typing import overload

from pypft.core.exceptions import InvalidFieldOperationError
from pypft.dft import AngularFFT
from pypft.fields import (
    AngularSpectrum,
    FrequencySamples,
    RadialSpectrum,
    SpatialSamples,
)
from pypft.grids import (
    PolarAngularModeGrid,
    PolarFrequencyGrid,
    PolarRadialModeGrid,
    PolarSpatialGrid,
)
from pypft.idft import AngularIFFT


@dataclass(frozen=True, slots=True)
class AngularDFT:
    transform: AngularFFT = field(default_factory=AngularFFT)

    @overload
    def apply(self, values: SpatialSamples) -> AngularSpectrum: ...

    @overload
    def apply(self, values: FrequencySamples) -> RadialSpectrum: ...

    def apply(
        self,
        values: SpatialSamples | FrequencySamples,
    ) -> AngularSpectrum | RadialSpectrum:
        transformed = self.transform.execute(values.asarray())

        if isinstance(values, SpatialSamples):
            return AngularSpectrum(
                data=transformed,
                grid=PolarAngularModeGrid(
                    radial_size=values.grid.radial_size,
                    angular_mode_count=values.grid.angular_size,
                ),
            )

        if isinstance(values, FrequencySamples):
            return RadialSpectrum(
                data=transformed,
                grid=PolarRadialModeGrid(
                    radial_frequency_size=values.grid.radial_size,
                    angular_mode_count=values.grid.angular_size,
                ),
            )

        raise InvalidFieldOperationError(
            "AngularDFT expects SpatialSamples or FrequencySamples."
        )


@dataclass(frozen=True, slots=True)
class AngularIDFT:
    transform: AngularIFFT = field(default_factory=AngularIFFT)

    @overload
    def apply(self, values: AngularSpectrum) -> SpatialSamples: ...

    @overload
    def apply(self, values: RadialSpectrum) -> FrequencySamples: ...

    def apply(
        self,
        values: AngularSpectrum | RadialSpectrum,
    ) -> SpatialSamples | FrequencySamples:
        reconstructed = self.transform.execute(values.asarray())

        if isinstance(values, AngularSpectrum):
            return SpatialSamples(
                data=reconstructed,
                grid=PolarSpatialGrid(
                    radial_size=values.grid.radial_size,
                    angular_size=values.grid.angular_mode_count,
                ),
            )

        if isinstance(values, RadialSpectrum):
            return FrequencySamples(
                data=reconstructed,
                grid=PolarFrequencyGrid(
                    radial_size=values.grid.radial_frequency_size,
                    angular_size=values.grid.angular_mode_count,
                ),
            )

        raise InvalidFieldOperationError(
            "AngularIDFT expects AngularSpectrum or RadialSpectrum."
        )


__all__ = ["AngularDFT", "AngularIDFT"]
