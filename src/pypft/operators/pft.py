from __future__ import annotations

from dataclasses import dataclass, field

from pypft.fields import FrequencySamples, SpatialSamples
from pypft.grids import PolarFrequencyGrid, PolarSpatialGrid
from pypft.operators.angular import AngularDFT, AngularIDFT
from pypft.operators.radial import RadialDHT


@dataclass(frozen=True, slots=True)
class ForwardPFTPlan:
    angular_transform: AngularDFT = field(default_factory=AngularDFT)
    radial_transform: RadialDHT = field(default_factory=RadialDHT)
    angular_reconstruction: AngularIDFT = field(default_factory=AngularIDFT)

    def execute(
        self,
        values: SpatialSamples,
        *,
        frequency_grid: PolarFrequencyGrid,
    ) -> FrequencySamples:
        angular_modes = self.angular_transform.apply(values)
        radial_modes = self.radial_transform.apply(
            angular_modes,
            target_grid=frequency_grid,
        )
        return self.angular_reconstruction.apply(radial_modes)


@dataclass(frozen=True, slots=True)
class BackwardPFTPlan:
    angular_transform: AngularDFT = field(default_factory=AngularDFT)
    radial_transform: RadialDHT = field(default_factory=RadialDHT)
    angular_reconstruction: AngularIDFT = field(default_factory=AngularIDFT)

    def execute(
        self,
        values: FrequencySamples,
        *,
        spatial_grid: PolarSpatialGrid,
    ) -> SpatialSamples:
        radial_modes = self.angular_transform.apply(values)
        angular_modes = self.radial_transform.apply(
            radial_modes,
            target_grid=spatial_grid,
        )
        return self.angular_reconstruction.apply(angular_modes)


__all__ = ["BackwardPFTPlan", "ForwardPFTPlan"]
