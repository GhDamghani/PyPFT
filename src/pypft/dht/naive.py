from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import jv

from pypft.core.exceptions import GridMismatchError, InputShapeError
from pypft.core.typing import ComplexArray
from pypft.dht.base import DHTDirection
from pypft.grids import PolarFrequencyGrid, PolarSpatialGrid


@dataclass(frozen=True, slots=True)
class NaiveDHTImplementation:
    """Legacy-style direct DHT reference implementation.

    The implementation mirrors the historical nested-loop DHT in the pre-0.1
    codebase, but adapts it to the current grid-based stage contract and batch
    handling.
    """

    key: str = "naive"
    description: str = "Legacy-compatible direct DHT reference implementation."
    supports_batching: bool = True

    def apply(
        self,
        values: ComplexArray,
        *,
        source_grid: PolarSpatialGrid | PolarFrequencyGrid,
        target_grid: PolarSpatialGrid | PolarFrequencyGrid,
        direction: DHTDirection,
    ) -> ComplexArray:
        _validate_contract(
            values=values,
            source_grid=source_grid,
            target_grid=target_grid,
            direction=direction,
        )

        values_numpy, restore_output = _as_numpy_with_restorer(values)
        transformed = _apply_direct_dht(
            values_numpy,
            source_radial_size=source_grid.radial_size,
            target_radial_size=target_grid.radial_size,
            angular_size=source_grid.angular_size,
        )
        return restore_output(transformed)


def _validate_contract(
    *,
    values: ComplexArray,
    source_grid: PolarSpatialGrid | PolarFrequencyGrid,
    target_grid: PolarSpatialGrid | PolarFrequencyGrid,
    direction: DHTDirection,
) -> None:
    if direction == "forward":
        if not isinstance(source_grid, PolarSpatialGrid):
            raise GridMismatchError(
                "Forward DHT expects a PolarSpatialGrid source grid."
            )
        if not isinstance(target_grid, PolarFrequencyGrid):
            raise GridMismatchError(
                "Forward DHT expects a PolarFrequencyGrid target grid."
            )
    else:
        if not isinstance(source_grid, PolarFrequencyGrid):
            raise GridMismatchError(
                "Backward DHT expects a PolarFrequencyGrid source grid."
            )
        if not isinstance(target_grid, PolarSpatialGrid):
            raise GridMismatchError(
                "Backward DHT expects a PolarSpatialGrid target grid."
            )

    if values.ndim not in (2, 3):
        raise InputShapeError(
            "NaiveDHTImplementation expects a 2D or batched 3D array."
        )

    runtime_shape = values.shape[-2:]
    source_grid.validate_shape(runtime_shape)

    if source_grid.angular_size != target_grid.angular_size:
        raise GridMismatchError(
            "The radial DHT preserves the angular axis, so the source and "
            "target grids must use the same angular size."
        )

    if source_grid.angular_size % 2 != 0:
        raise InputShapeError(
            "NaiveDHTImplementation requires an even angular size."
        )


def _as_numpy_with_restorer(values: ComplexArray):
    module_name = type(values).__module__
    if module_name.startswith("cupy"):
        import cupy as cp

        return cp.asnumpy(values), lambda result: cp.asarray(
            result,
            dtype=cp.complex128,
        )

    return np.asarray(values, dtype=np.complex128), lambda result: np.asarray(
        result,
        dtype=np.complex128,
    )


def _apply_direct_dht(
    values: np.ndarray,
    *,
    source_radial_size: int,
    target_radial_size: int,
    angular_size: int,
) -> np.ndarray:
    batched_values = np.asarray(values, dtype=np.complex128)
    had_batch_axis = batched_values.ndim == 3
    if not had_batch_axis:
        batched_values = batched_values[None, :, :]

    rho = (0.5 + np.arange(source_radial_size, dtype=np.float64)) / float(
        source_radial_size
    )
    radii = 0.5 * (1.0 + np.arange(target_radial_size, dtype=np.float64))
    orders = np.arange(-(angular_size // 2), angular_size // 2)
    kernel_arguments = np.pi * radii[:, None] * rho[None, :]
    kernel = jv(orders[None, None, :], kernel_arguments[:, :, None])

    scaled_values = batched_values * rho[None, :, None]
    transformed = np.einsum(
        "bsa,tsa->bta",
        scaled_values,
        kernel,
        optimize=True,
        dtype=np.complex128,
    )

    if had_batch_axis:
        return transformed
    return transformed[0]


__all__ = ["NaiveDHTImplementation"]
