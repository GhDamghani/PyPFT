from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import fft as scipy_fft

from pypft.core.typing import ComplexArray, FFTNorm


@dataclass(frozen=True, slots=True)
class CPUBackend:
    key: str = "cpu"

    def as_complex128(self, values: Any) -> ComplexArray:
        try:
            return np.asarray(values, dtype=np.complex128)
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
        transformed = scipy_fft.fft(values, axis=axis, norm=norm)
        return np.asarray(transformed, dtype=np.complex128)

    def ifft(
        self,
        values: ComplexArray,
        *,
        axis: int,
        norm: FFTNorm,
    ) -> ComplexArray:
        transformed = scipy_fft.ifft(values, axis=axis, norm=norm)
        return np.asarray(transformed, dtype=np.complex128)


__all__ = ["CPUBackend"]
