from __future__ import annotations

import numpy as np

from pypft.core.typing import ComplexArray


def magnitude(values: ComplexArray) -> np.ndarray:
    return np.abs(values)


def phase(values: ComplexArray) -> np.ndarray:
    return np.angle(values)


__all__ = ["magnitude", "phase"]
