from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pypft.core.exceptions import BackendUnavailableError
from pypft.core.typing import ComplexArray, FFTNorm


def _load_cupy() -> tuple[Any, Any]:
    try:
        import cupy as cp
        from cupyx.scipy import fft as cupy_fft
    except ImportError as error:
        raise BackendUnavailableError(
            "The GPU backend requires the optional 'gpu' extra with "
            "cupy-cuda12x installed."
        ) from error
    return cp, cupy_fft


@dataclass(frozen=True, slots=True)
class CuPyBackend:
    key: str = "gpu"

    def __post_init__(self) -> None:
        _load_cupy()

    def as_complex128(self, values: Any) -> ComplexArray:
        cp, _ = _load_cupy()
        try:
            return cp.asarray(values, dtype=cp.complex128)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "PyPFT inputs must be numeric and convertible to complex128."
            ) from error

    def fft(
        self,
        values: ComplexArray,
        *,
        axis: int,
        norm: FFTNorm,
    ) -> ComplexArray:
        cp, cupy_fft = _load_cupy()
        transformed = cupy_fft.fft(values, axis=axis, norm=norm)
        return cp.asarray(transformed, dtype=cp.complex128)

    def ifft(
        self,
        values: ComplexArray,
        *,
        axis: int,
        norm: FFTNorm,
    ) -> ComplexArray:
        cp, cupy_fft = _load_cupy()
        transformed = cupy_fft.ifft(values, axis=axis, norm=norm)
        return cp.asarray(transformed, dtype=cp.complex128)


__all__ = ["CuPyBackend"]
