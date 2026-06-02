from __future__ import annotations

from dataclasses import dataclass, field
from typing import overload

import numpy as np

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
    PolarRadialModeGrid,
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
        transformed = _fftshift(
            self.transform.execute(values.asarray()),
            axis=self.transform.axis,
        )

        if isinstance(values, SpatialSamples):
            return AngularSpectrum(
                data=transformed,
                grid=PolarAngularModeGrid(
                    radial_size=values.grid.radial_size,
                    angular_mode_count=values.grid.angular_size,
                ),
                endpoint_grid=values.grid,
            )

        if isinstance(values, FrequencySamples):
            return RadialSpectrum(
                data=transformed,
                grid=PolarRadialModeGrid(
                    radial_frequency_size=values.grid.radial_size,
                    angular_mode_count=values.grid.angular_size,
                ),
                endpoint_grid=values.grid,
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
        reconstructed = self.transform.execute(
            _ifftshift(values.asarray(), axis=self.transform.axis)
        )

        if isinstance(values, AngularSpectrum):
            return SpatialSamples(
                data=reconstructed,
                grid=values.endpoint_grid,
            )

        if isinstance(values, RadialSpectrum):
            return FrequencySamples(
                data=reconstructed,
                grid=values.endpoint_grid,
            )

        raise InvalidFieldOperationError(
            "AngularIDFT expects AngularSpectrum or RadialSpectrum."
        )


def _fftshift(values, *, axis: int):
    module_name = type(values).__module__
    if module_name.startswith("cupy"):
        import cupy as cp

        return cp.fft.fftshift(values, axes=axis)

    return np.fft.fftshift(values, axes=axis)


def _ifftshift(values, *, axis: int):
    module_name = type(values).__module__
    if module_name.startswith("cupy"):
        import cupy as cp

        return cp.fft.ifftshift(values, axes=axis)

    return np.fft.ifftshift(values, axes=axis)


__all__ = ["AngularDFT", "AngularIDFT"]
