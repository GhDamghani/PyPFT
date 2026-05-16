from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pypft.backends import ExecutionBackend
from pypft.core.exceptions import InputShapeError
from pypft.core.typing import ComplexArray


@dataclass(frozen=True, slots=True)
class NormalizedInput:
    values: ComplexArray
    had_batch_axis: bool


def normalize_transform_input(
    values: Any,
    *,
    enable_batching: bool,
    backend: ExecutionBackend,
) -> NormalizedInput:
    array = backend.as_complex128(values)
    if array.ndim == 2:
        _validate_non_empty_shape(array.shape)
        return NormalizedInput(values=array[None, :, :], had_batch_axis=False)
    if array.ndim == 3:
        if not enable_batching:
            raise InputShapeError(
                "Batch inputs require enable_batching=True and must use "
                "a leading batch axis."
            )
        _validate_non_empty_shape(array.shape[1:])
        if array.shape[0] <= 0:
            raise InputShapeError(
                "Batch inputs must include at least one image."
            )
        return NormalizedInput(values=array, had_batch_axis=True)
    raise InputShapeError(
        "PyPFT expects either a 2D array (n_r, n_theta) or a 3D array "
        "(batch, n_r, n_theta)."
    )


def restore_output_shape(values: ComplexArray, *, had_batch_axis: bool):
    if had_batch_axis:
        return values
    return values[0, :, :]


def _validate_non_empty_shape(shape: tuple[int, int]) -> None:
    radial_size, angular_size = shape
    if radial_size <= 0 or angular_size <= 0:
        raise InputShapeError(
            "Transform inputs must have non-zero radial and angular sizes."
        )


__all__ = [
    "NormalizedInput",
    "normalize_transform_input",
    "restore_output_shape",
]
