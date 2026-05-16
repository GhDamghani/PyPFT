from __future__ import annotations

from typing import Any, Protocol

from pypft.core.exceptions import UnknownBackendError
from pypft.core.typing import ComplexArray, FFTNorm


class ExecutionBackend(Protocol):
    key: str

    def as_complex128(self, values: Any) -> ComplexArray: ...

    def fft(
        self,
        values: ComplexArray,
        *,
        axis: int,
        norm: FFTNorm,
    ) -> ComplexArray: ...

    def ifft(
        self,
        values: ComplexArray,
        *,
        axis: int,
        norm: FFTNorm,
    ) -> ComplexArray: ...


def create_backend(key: str) -> ExecutionBackend:
    if key == "cpu":
        from pypft.backends.cpu import CPUBackend

        return CPUBackend()
    if key == "gpu":
        from pypft.backends.gpu import CuPyBackend

        return CuPyBackend()
    raise UnknownBackendError(f"Unknown execution backend: {key!r}.")


__all__ = ["ExecutionBackend", "create_backend"]
