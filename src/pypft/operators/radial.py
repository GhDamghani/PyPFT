from __future__ import annotations

from dataclasses import dataclass, field
from typing import overload

from pypft.core.exceptions import InvalidFieldOperationError
from pypft.dht import DHTImplementation, create_dht_implementation
from pypft.fields import AngularSpectrum, RadialSpectrum
from pypft.grids import (
    PolarAngularModeGrid,
    PolarFrequencyGrid,
    PolarRadialModeGrid,
    PolarSpatialGrid,
)


@dataclass(frozen=True, slots=True)
class RadialDHT:
    implementation: DHTImplementation = field(
        default_factory=lambda: create_dht_implementation("naive")
    )

    @overload
    def apply(
        self,
        values: AngularSpectrum,
        *,
        target_grid: PolarFrequencyGrid,
    ) -> RadialSpectrum: ...

    @overload
    def apply(
        self,
        values: RadialSpectrum,
        *,
        target_grid: PolarSpatialGrid,
    ) -> AngularSpectrum: ...

    def apply(
        self,
        values: AngularSpectrum | RadialSpectrum,
        *,
        target_grid: PolarFrequencyGrid | PolarSpatialGrid,
    ) -> RadialSpectrum | AngularSpectrum:
        if isinstance(values, AngularSpectrum) and isinstance(
            target_grid,
            PolarFrequencyGrid,
        ):
            transformed = self.implementation.apply(
                values.asarray(),
                source_grid=PolarSpatialGrid(
                    radial_size=values.grid.radial_size,
                    angular_size=values.grid.angular_mode_count,
                ),
                target_grid=target_grid,
                direction="forward",
            )
            return RadialSpectrum(
                data=transformed,
                grid=PolarRadialModeGrid(
                    radial_frequency_size=target_grid.radial_size,
                    angular_mode_count=target_grid.angular_size,
                ),
            )

        if isinstance(values, RadialSpectrum) and isinstance(
            target_grid,
            PolarSpatialGrid,
        ):
            transformed = self.implementation.apply(
                values.asarray(),
                source_grid=PolarFrequencyGrid(
                    radial_size=values.grid.radial_frequency_size,
                    angular_size=values.grid.angular_mode_count,
                ),
                target_grid=target_grid,
                direction="backward",
            )
            return AngularSpectrum(
                data=transformed,
                grid=PolarAngularModeGrid(
                    radial_size=target_grid.radial_size,
                    angular_mode_count=target_grid.angular_size,
                ),
            )

        raise InvalidFieldOperationError(
            "RadialDHT expects AngularSpectrum with a PolarFrequencyGrid "
            "target or RadialSpectrum with a PolarSpatialGrid target."
        )


__all__ = ["RadialDHT"]
