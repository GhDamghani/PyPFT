from __future__ import annotations

from dataclasses import dataclass, field

from pypft.backends.base import ExecutionBackend, create_backend
from pypft.core.typing import ComplexArray, FFTNorm


@dataclass(frozen=True, slots=True)
class AngularFFT:
    axis: int = -1
    norm: FFTNorm = "backward"
    backend: ExecutionBackend = field(
        default_factory=lambda: create_backend("cpu")
    )

    def execute(self, values: ComplexArray) -> ComplexArray:
        return self.backend.fft(values, axis=self.axis, norm=self.norm)


__all__ = ["AngularFFT", "FFTNorm"]
