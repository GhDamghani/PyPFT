from __future__ import annotations

from dataclasses import dataclass

from pypft.core.typing import ComplexArray
from pypft.dft import AngularFFT
from pypft.dht import DHTImplementation
from pypft.grids import PolarFrequencyGrid, PolarSpatialGrid
from pypft.idft import AngularIFFT


@dataclass(frozen=True, slots=True)
class ForwardPFTPlan:
    angular_transform: AngularFFT
    radial_transform: DHTImplementation
    angular_reconstruction: AngularIFFT

    def execute(
        self,
        values: ComplexArray,
        *,
        spatial_grid: PolarSpatialGrid,
        frequency_grid: PolarFrequencyGrid,
    ) -> ComplexArray:
        angular_modes = self.angular_transform.execute(values)
        radial_modes = self.radial_transform.apply(
            angular_modes,
            source_grid=spatial_grid,
            target_grid=frequency_grid,
            direction="forward",
        )
        return self.angular_reconstruction.execute(radial_modes)


@dataclass(frozen=True, slots=True)
class BackwardPFTPlan:
    angular_transform: AngularFFT
    radial_transform: DHTImplementation
    angular_reconstruction: AngularIFFT

    def execute(
        self,
        values: ComplexArray,
        *,
        frequency_grid: PolarFrequencyGrid,
        spatial_grid: PolarSpatialGrid,
    ) -> ComplexArray:
        angular_modes = self.angular_transform.execute(values)
        radial_modes = self.radial_transform.apply(
            angular_modes,
            source_grid=frequency_grid,
            target_grid=spatial_grid,
            direction="backward",
        )
        return self.angular_reconstruction.execute(radial_modes)


__all__ = ["BackwardPFTPlan", "ForwardPFTPlan"]
