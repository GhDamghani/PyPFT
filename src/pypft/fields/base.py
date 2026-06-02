from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, Self, TypeVar, cast

import numpy as np
from numpy.typing import DTypeLike

from pypft.core.exceptions import (
    GridMismatchError,
    InputShapeError,
    InvalidFieldOperationError,
)


class _GridShape(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...


class _FieldArray(Protocol):
    ndim: int
    shape: tuple[int, ...]
    dtype: Any
    real: Any
    imag: Any

    def astype(self, dtype: DTypeLike, *, copy: bool = True) -> Any: ...

    def conj(self) -> Any: ...

    def copy(self) -> Any: ...

    def view(self) -> Any: ...

    def __add__(self, other: object) -> Any: ...

    def __sub__(self, other: object) -> Any: ...

    def __mul__(self, other: object) -> Any: ...

    def __truediv__(self, other: object) -> Any: ...

    def __neg__(self) -> Any: ...


GridT = TypeVar("GridT", bound=_GridShape)


@dataclass(frozen=True, slots=True, eq=False)
class PolarFieldBase(Generic[GridT]):
    data: _FieldArray
    grid: GridT

    def __post_init__(self) -> None:
        array = _coerce_field_array(self.data)
        if array.ndim < 2:
            raise InputShapeError(
                f"{type(self).__name__} expects at least two dimensions so "
                "last axes can match the attached polar grid."
            )

        runtime_shape = array.shape[-2:]
        if runtime_shape != self.grid.shape:
            raise GridMismatchError(
                f"{type(self).__name__} grid shape {self.grid.shape!r} does "
                f"not match data trailing shape {runtime_shape!r}."
            )

        object.__setattr__(self, "data", _readonly_view(array))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.data.shape[:-2]

    @property
    def real(self) -> Any:
        return _readonly_view(self.data.real)

    @property
    def imag(self) -> Any:
        return _readonly_view(self.data.imag)

    def copy(self) -> Self:
        return self._wrap(self.data.copy())

    def astype(self, dtype: DTypeLike, *, copy: bool = True) -> Self:
        return self._wrap(self.data.astype(dtype, copy=copy))

    def conj(self) -> Self:
        return self._wrap(self.data.conj())

    def asarray(self, *, copy: bool = False) -> _FieldArray:
        if copy:
            return cast(_FieldArray, self.data.copy())
        return self.data

    def __add__(self, other: object) -> Self:
        other_field = self._require_compatible_field(
            other,
            operation="addition",
        )
        return self._wrap(self.data + other_field.data)

    def __sub__(self, other: object) -> Self:
        other_field = self._require_compatible_field(
            other,
            operation="subtraction",
        )
        return self._wrap(self.data - other_field.data)

    def __mul__(self, other: object) -> Self:
        scalar = _require_scalar(other, stage_name=type(self).__name__)
        return self._wrap(self.data * scalar)

    def __rmul__(self, other: object) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> Self:
        scalar = _require_scalar(other, stage_name=type(self).__name__)
        return self._wrap(self.data / scalar)

    def __neg__(self) -> Self:
        return self._wrap(-self.data)

    def _wrap(self, data: _FieldArray) -> Self:
        return type(self)(data=data, grid=self.grid)

    def _require_compatible_field(
        self,
        other: object,
        *,
        operation: str,
    ) -> Self:
        if not isinstance(other, PolarFieldBase):
            raise InvalidFieldOperationError(
                f"{type(self).__name__} only supports {operation} with "
                f"another "
                f"{type(self).__name__} instance."
            )

        if type(other) is not type(self):
            raise InvalidFieldOperationError(
                f"{type(self).__name__} cannot participate in {operation} "
                f"with "
                f"{type(other).__name__}."
            )

        if self.grid != other.grid:
            raise GridMismatchError(
                f"{type(self).__name__} requires matching grids for "
                f"{operation}."
            )

        return cast(Self, other)


def _coerce_field_array(values: object) -> _FieldArray:
    if hasattr(values, "ndim") and hasattr(values, "shape"):
        return cast(_FieldArray, values)
    return cast(_FieldArray, np.asarray(values))


def _readonly_view(values: Any) -> Any:
    if isinstance(values, np.ndarray):
        readonly_view = values.view()
        readonly_view.setflags(write=False)
        return readonly_view
    return values


def _require_scalar(value: object, *, stage_name: str) -> complex:
    scalar_array = np.asarray(value)
    if scalar_array.ndim != 0:
        raise InvalidFieldOperationError(
            f"{stage_name} only supports scalar multiplication and division."
        )
    return complex(scalar_array.item())


__all__ = ["PolarFieldBase"]
