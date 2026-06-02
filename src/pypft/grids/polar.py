from __future__ import annotations

from dataclasses import dataclass

from pypft.core.exceptions import GridMismatchError, InputShapeError


@dataclass(frozen=True, slots=True)
class PolarSpatialGrid:
    radial_size: int
    angular_size: int

    def __post_init__(self) -> None:
        _validate_grid_sizes(self.radial_size, self.angular_size)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.radial_size, self.angular_size)

    @classmethod
    def infer_from_shape(cls, shape: tuple[int, int]) -> "PolarSpatialGrid":
        radial_size, angular_size = shape
        return cls(radial_size=radial_size, angular_size=angular_size)

    def validate_shape(self, shape: tuple[int, int]) -> None:
        if self.shape != shape:
            raise GridMismatchError(
                f"Spatial grid shape {self.shape!r} does not match runtime "
                f"input shape {shape!r}."
            )


@dataclass(frozen=True, slots=True)
class PolarFrequencyGrid:
    radial_size: int
    angular_size: int

    def __post_init__(self) -> None:
        _validate_grid_sizes(self.radial_size, self.angular_size)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.radial_size, self.angular_size)

    @classmethod
    def infer_from_shape(cls, shape: tuple[int, int]) -> "PolarFrequencyGrid":
        radial_size, angular_size = shape
        return cls(radial_size=radial_size, angular_size=angular_size)

    def validate_shape(self, shape: tuple[int, int]) -> None:
        if self.shape != shape:
            raise GridMismatchError(
                f"Frequency grid shape {self.shape!r} does not match runtime "
                f"input shape {shape!r}."
            )


@dataclass(frozen=True, slots=True)
class PolarAngularModeGrid:
    radial_size: int
    angular_mode_count: int

    def __post_init__(self) -> None:
        _validate_grid_sizes(self.radial_size, self.angular_mode_count)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.radial_size, self.angular_mode_count)

    @property
    def max_angular_order(self) -> int:
        return _validate_and_get_max_angular_order(self.angular_mode_count)

    def validate_shape(self, shape: tuple[int, int]) -> None:
        if self.shape != shape:
            raise GridMismatchError(
                f"Angular-mode grid shape {self.shape!r} does not match runtime "
                f"input shape {shape!r}."
            )


@dataclass(frozen=True, slots=True)
class PolarRadialModeGrid:
    radial_frequency_size: int
    angular_mode_count: int

    def __post_init__(self) -> None:
        _validate_grid_sizes(self.radial_frequency_size, self.angular_mode_count)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.radial_frequency_size, self.angular_mode_count)

    @property
    def max_angular_order(self) -> int:
        return _validate_and_get_max_angular_order(self.angular_mode_count)

    def validate_shape(self, shape: tuple[int, int]) -> None:
        if self.shape != shape:
            raise GridMismatchError(
                f"Radial-mode grid shape {self.shape!r} does not match runtime "
                f"input shape {shape!r}."
            )


@dataclass(frozen=True, slots=True)
class PolarTransformGrids:
    spatial: PolarSpatialGrid
    angular_modes: PolarAngularModeGrid
    radial_modes: PolarRadialModeGrid
    frequency: PolarFrequencyGrid

    @classmethod
    def from_endpoint_grids(
        cls,
        spatial: PolarSpatialGrid,
        frequency: PolarFrequencyGrid,
    ) -> "PolarTransformGrids":
        if spatial.angular_size != frequency.angular_size:
            raise GridMismatchError(
                "Spatial and frequency grids must use the same angular size so "
                "the angular DFT and inverse DFT operate on the same mode count."
            )

        angular_modes = PolarAngularModeGrid(
            radial_size=spatial.radial_size,
            angular_mode_count=spatial.angular_size,
        )
        radial_modes = PolarRadialModeGrid(
            radial_frequency_size=frequency.radial_size,
            angular_mode_count=frequency.angular_size,
        )
        return cls(
            spatial=spatial,
            angular_modes=angular_modes,
            radial_modes=radial_modes,
            frequency=frequency,
        )

    @property
    def angular_size(self) -> int:
        return self.spatial.angular_size

    @property
    def radial_size(self) -> int:
        return self.spatial.radial_size

    @property
    def radial_frequency_size(self) -> int:
        return self.frequency.radial_size


def _validate_grid_sizes(radial_size: int, angular_size: int) -> None:
    if radial_size <= 0 or angular_size <= 0:
        raise InputShapeError(
            "Grid sizes must be positive along the radial and angular axes."
        )


def _validate_and_get_max_angular_order(angular_mode_count: int) -> int:
    if angular_mode_count % 2 == 0:
        raise InputShapeError(
            "The angular mode count must be odd so it can represent symmetric "
            "orders n = -M, ..., M with N_theta = 2M + 1."
        )
    return (angular_mode_count - 1) // 2


__all__ = [
    "PolarAngularModeGrid",
    "PolarFrequencyGrid",
    "PolarRadialModeGrid",
    "PolarSpatialGrid",
    "PolarTransformGrids",
]
