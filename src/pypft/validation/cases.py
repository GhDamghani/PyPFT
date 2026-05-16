from __future__ import annotations

from dataclasses import dataclass

from pypft.core.typing import ComplexArray


@dataclass(frozen=True, slots=True)
class ValidationCase:
    name: str
    values: ComplexArray
    notes: str = ""


__all__ = ["ValidationCase"]
