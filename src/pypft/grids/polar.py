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


def _validate_grid_sizes(radial_size: int, angular_size: int) -> None:
    if radial_size <= 0 or angular_size <= 0:
        raise InputShapeError(
            "Grid sizes must be positive along the radial and angular axes."
        )


__all__ = ["PolarFrequencyGrid", "PolarSpatialGrid"]
