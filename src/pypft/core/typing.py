from __future__ import annotations

from typing import Any, Literal, Protocol, TypeAlias

FFTNorm: TypeAlias = Literal["backward", "forward", "ortho"]


class ComplexArray(Protocol):
    """Minimal array protocol shared by NumPy and CuPy complex arrays."""

    ndim: int
    shape: tuple[int, ...]

    def __getitem__(self, key: Any) -> Any: ...


__all__ = ["ComplexArray", "FFTNorm"]
