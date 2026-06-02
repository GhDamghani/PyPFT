from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from scipy.special import jv

from pypft.core.exceptions import GridMismatchError, InputShapeError
from pypft.core.typing import ComplexArray
from pypft.dht.base import DHTDirection
from pypft.grids import PolarFrequencyGrid, PolarSpatialGrid

_PSEUDOINVERSE_RCOND = 1e-12


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
        if direction == "forward":
            transformed = _apply_direct_dht(
                values_numpy,
                source_radial_size=source_grid.radial_size,
                target_radial_size=target_grid.radial_size,
                angular_size=source_grid.angular_size,
            )
        else:
            transformed = _apply_inverse_dht(
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
    cp = _load_optional_cupy()
    if cp is not None and isinstance(values, cp.ndarray):
        return cp.asnumpy(values), lambda result: cp.asarray(
            result,
            dtype=cp.complex128,
        )

    return np.asarray(values, dtype=np.complex128), lambda result: np.asarray(
        result,
        dtype=np.complex128,
    )


def _load_optional_cupy() -> Any | None:
    try:
        import cupy as cp
    except ImportError:
        return None
    return cp


def _apply_direct_dht(
    values: np.ndarray,
    *,
    source_radial_size: int,
    target_radial_size: int,
    angular_size: int,
) -> np.ndarray:
    batched_values = values
    had_batch_axis = batched_values.ndim == 3
    if not had_batch_axis:
        batched_values = batched_values[None, :, :]

    kernel = _build_weighted_kernel(
        source_radial_size=source_radial_size,
        target_radial_size=target_radial_size,
        angular_size=angular_size,
    )
    transformed = np.einsum(
        "bsa,tsa->bta",
        batched_values,
        kernel,
        optimize=True,
        dtype=np.complex128,
    )

    if had_batch_axis:
        return transformed
    return transformed[0]


def _apply_inverse_dht(
    values: np.ndarray,
    *,
    source_radial_size: int,
    target_radial_size: int,
    angular_size: int,
) -> np.ndarray:
    batched_values = values
    had_batch_axis = batched_values.ndim == 3
    if not had_batch_axis:
        batched_values = batched_values[None, :, :]

    forward_kernel = _build_weighted_kernel(
        source_radial_size=target_radial_size,
        target_radial_size=source_radial_size,
        angular_size=angular_size,
    )
    inverse_kernel = _build_inverse_kernel(forward_kernel)

    transformed = np.einsum(
        "bta,sta->bsa",
        batched_values,
        inverse_kernel,
        optimize=True,
        dtype=np.complex128,
    )

    if had_batch_axis:
        return transformed
    return transformed[0]


def _build_inverse_kernel(
    forward_kernel: np.ndarray,
    *,
    rcond: float = _PSEUDOINVERSE_RCOND,
) -> np.ndarray:
    source_radial_size, target_radial_size, angular_size = (
        forward_kernel.shape
    )
    inverse_kernel = np.empty(
        (target_radial_size, source_radial_size, angular_size),
        dtype=np.complex128,
    )
    truncation_messages: list[str] = []

    for mode_index in range(angular_size):
        mode_inverse, truncation_message = _compute_mode_pseudoinverse(
            forward_kernel[:, :, mode_index],
            mode_index=mode_index,
            rcond=rcond,
        )
        inverse_kernel[:, :, mode_index] = mode_inverse
        if truncation_message is not None:
            truncation_messages.append(truncation_message)

    if truncation_messages:
        warnings.warn(
            "NaiveDHTImplementation inverse kernel truncated singular "
            f"values using rcond={rcond:.1e}; "
            + "; ".join(truncation_messages),
            RuntimeWarning,
            stacklevel=2,
        )

    return inverse_kernel


def _compute_mode_pseudoinverse(
    kernel: np.ndarray,
    *,
    mode_index: int,
    rcond: float,
) -> tuple[np.ndarray, str | None]:
    left_singular_vectors, singular_values, right_h = np.linalg.svd(
        kernel,
        full_matrices=False,
    )
    if singular_values.size == 0:
        return np.empty((kernel.shape[1], kernel.shape[0]), dtype=np.complex128), None

    largest_singular_value = float(singular_values[0])
    if largest_singular_value == 0.0:
        return np.zeros((kernel.shape[1], kernel.shape[0]), dtype=np.complex128), (
            f"mode {mode_index}: retained rank 0/{singular_values.size}, "
            "cutoff 0.000e+00, largest singular value 0.000e+00, "
            "smallest singular value 0.000e+00"
        )

    cutoff = rcond * largest_singular_value
    retained_mask = singular_values > cutoff
    inverse_singular_values = np.zeros_like(singular_values)
    inverse_singular_values[retained_mask] = 1.0 / singular_values[
        retained_mask
    ]
    pseudoinverse = (
        right_h.conj().T * inverse_singular_values[None, :]
    ) @ left_singular_vectors.conj().T

    if np.all(retained_mask):
        return np.asarray(pseudoinverse, dtype=np.complex128), None

    return np.asarray(pseudoinverse, dtype=np.complex128), (
        f"mode {mode_index}: retained rank {int(retained_mask.sum())}/"
        f"{singular_values.size}, cutoff {cutoff:.3e}, largest singular "
        f"value {largest_singular_value:.3e}, smallest singular value "
        f"{float(singular_values[-1]):.3e}"
    )


def _build_weighted_kernel(
    *,
    source_radial_size: int,
    target_radial_size: int,
    angular_size: int,
) -> np.ndarray:
    if angular_size % 2 != 0:
        raise InputShapeError(
            "NaiveDHTImplementation requires an even angular size."
        )

    rho = _normalized_midpoint_samples(source_radial_size)
    radii = _normalized_midpoint_samples(target_radial_size)
    orders = np.arange(-(angular_size // 2), angular_size // 2)
    kernel_arguments = np.pi * radii[:, None] * rho[None, :]
    bessel_kernel = jv(orders[None, None, :], kernel_arguments[:, :, None])
    return bessel_kernel * rho[None, :, None]


def _normalized_midpoint_samples(size: int) -> np.ndarray:
    return (0.5 + np.arange(size, dtype=np.float64)) / float(size)


__all__ = ["NaiveDHTImplementation"]
